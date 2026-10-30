/* Geometry-driven circuit rendering. All motion and horizons come from recorded solves. */
(function () {
  'use strict';

  const mix = (a, b, t) => a + (b - a) * t,
    mod = (a, b) => (a % b + b) % b;
  class CircuitGeometry {
    constructor(track) {
      this.track = track;
      this.s = track.s;
      this.xy = track.xy;
      this.length = track.length;
      // Interpolate tangents as well as positions; segment-wise headings cause visible ticks.
      const count = this.xy.length - 1;
      this.headings = this.xy.map((point, i) => {
        const before = this.xy[(i + count - 1) % count],
          after = this.xy[(i + 1) % count];
        return Math.atan2(after[1] - before[1], after[0] - before[0]);
      });
      for (let i = 1; i < this.headings.length; i++) {
        this.headings[i] = this.headings[i - 1] + mod(this.headings[i] - this.headings[i - 1] + Math.PI, 2 * Math.PI) - Math.PI;
      }
      this.bounds = [Math.min(...this.xy.map(p => p[0])), Math.max(...this.xy.map(p => p[0])), Math.min(...this.xy.map(p => p[1])), Math.max(...this.xy.map(p => p[1]))];
    }
    point(s, n = 0) {
      s = mod(s, this.length);
      let lo = 0,
        hi = this.s.length - 1;
      while (hi - lo > 1) {
        const m = lo + hi >> 1;
        if (this.s[m] > s) hi = m;else lo = m;
      }
      const t = (s - this.s[lo]) / (this.s[hi] - this.s[lo]),
        a = this.xy[lo],
        b = this.xy[hi],
        angle = mix(this.headings[lo], this.headings[hi], t);
      return [mix(a[0], b[0], t) - Math.sin(angle) * n, mix(a[1], b[1], t) + Math.cos(angle) * n, angle];
    }
    states(run, time) {
      const index = Math.min(run.states.length - 1, Math.floor(time / run.dt)),
        next = Math.min(index + 1, run.states.length - 1),
        fraction = Math.min(1, time / run.dt - index);
      return run.states[index].map((x, i) => x.map((v, j) => mix(v, run.states[next][i][j], fraction)));
    }
  }
  class RaceScene {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.mode = 'broadcast';
      this.predictions = true;
      this.focus = 0;
      this.zoom = 1.25;
    }
    setRun(run) {
      this.run = run;
      this.geometry = new CircuitGeometry(run.track);
    }
    resize() {
      const box = this.canvas.getBoundingClientRect(),
        dpr = Math.min(2, window.devicePixelRatio || 1);
      this.width = box.width;
      this.height = box.height;
      if (this.canvas.width !== Math.round(box.width * dpr) || this.canvas.height !== Math.round(box.height * dpr)) {
        this.canvas.width = Math.round(box.width * dpr);
        this.canvas.height = Math.round(box.height * dpr);
      }
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    polygon(points, color, stroke = null, width = 1) {
      const c = this.ctx;
      c.beginPath();
      points.forEach((p, i) => i ? c.lineTo(...p) : c.moveTo(...p));
      c.closePath();
      c.fillStyle = color;
      c.fill();
      if (stroke) {
        c.strokeStyle = stroke;
        c.lineWidth = width;
        c.stroke();
      }
    }
    line(points, color, width = 1, dash = []) {
      if (points.length < 2) return;
      const c = this.ctx;
      c.beginPath();
      points.forEach((p, i) => i ? c.lineTo(...p) : c.moveTo(...p));
      c.strokeStyle = color;
      c.lineWidth = width;
      c.lineJoin = 'round';
      c.lineCap = 'round';
      c.setLineDash(dash);
      c.stroke();
      c.setLineDash([]);
    }
    render(time) {
      if (!this.run) return;
      this.resize();
      const c = this.ctx,
        w = this.width,
        h = this.height,
        g = this.geometry,
        run = this.run,
        states = g.states(run, time),
        ego = states[this.focus];
      const point = g.point(ego[0], ego[1]);
      let center, angle, scale, pitch;
      if (this.mode === 'circuit') {
        const b = g.bounds;
        center = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2];
        angle = 0;
        const margin = run.track.half_width + 8;
        scale = Math.min((w - 320) / (b[1] - b[0] + 2 * margin), (h - 90) / (b[3] - b[2] + 2 * margin));
        scale = Math.max(.04, scale);
        pitch = 1;
      } else {
        const before = g.point(ego[0] - 45),
          ahead = g.point(ego[0] + 65);
        angle = this.mode === 'overhead' ? 0 : Math.atan2(ahead[1] - before[1], ahead[0] - before[0]) - .55;
        const target = g.point(ego[0] + 22, ego[1] * .3);
        center = [mix(point[0], target[0], .6), mix(point[1], target[1], .6)];
        scale = (w < 650 ? Math.max(w / 75, h / 120) : Math.min(w / 155, h / 96)) * this.zoom;
        pitch = this.mode === 'overhead' ? 1 : .72;
      }
      this.scale = scale;
      const cos = Math.cos(angle),
        sin = Math.sin(angle);
      this.project = (p, z = 0) => {
        const x = p[0] - center[0],
          y = p[1] - center[1];
        const vx = x * cos + y * sin,
          vy = -x * sin + y * cos;
        const depth = this.mode === 'broadcast' ? 300 + vy * .694 - z * .72 : 300;
        const factor = 300 / Math.max(35, depth);
        return [w * .54 + scale * vx * factor, h * .55 - scale * (vy * pitch + z * .694) * factor];
      };
      c.fillStyle = '#a9b49c';
      c.fillRect(0, 0, w, h);
      // Stable world-space mowing bands and subtle ground grain.
      if (this.mode !== 'circuit' || scale >= 1.5) {
        for (let x = Math.floor((center[0] - 250) / 16) * 16; x < center[0] + 250; x += 16) this.polygon([[x, center[1] - 300], [x + 8, center[1] - 300], [x + 8, center[1] + 300], [x, center[1] + 300]].map(p => this.project(p)), '#a4b096');
        for (let i = 0; i < 900; i++) {
          const x = Math.floor(center[0] / 300) * 300 - 300 + i * 71.37 % 900,
            y = Math.floor(center[1] / 300) * 300 - 300 + i * 43.71 % 900;
          const p = this.project([x, y]);
          if (p[0] > 0 && p[0] < w && p[1] > 0 && p[1] < h) {
            c.fillStyle = i % 2 ? '#b0baa2' : '#a0ad92';
            c.fillRect(p[0], p[1], 1.1, 1.1);
          }
        }
      }
      const chunks = [];
      let chunk = [];
      for (let i = 0; i < g.s.length; i++) {
        const p = g.xy[i],
          near = this.mode === 'circuit' || Math.hypot(p[0] - center[0], p[1] - center[1]) < 290;
        if (near) chunk.push(i);else if (chunk.length) {
          chunks.push(chunk);
          chunk = [];
        }
      }
      if (chunk.length) chunks.push(chunk);
      const edge = (i, side, extra = 0) => {
        const base = g.point(g.s[i], side * run.track.half_width),
          p = g.xy[i],
          d = Math.hypot(base[0] - p[0], base[1] - p[1]);
        return [base[0] + (base[0] - p[0]) * extra / d, base[1] + (base[1] - p[1]) * extra / d];
      };
      for (const indices of chunks) {
        if (indices.length < 2) continue;
        const strip = (a, b, extra, color) => this.polygon([...indices.map(i => this.project(edge(i, a, extra))), ...indices.slice().reverse().map(i => this.project(edge(i, b, extra)))], color);
        strip(1, -1, 8, '#bcbab0');
        strip(1, -1, 3.2, '#919891');
        strip(1, -1, .7, '#454b4e');
        strip(1, -1, 0, '#53595b');
        if (this.mode !== 'circuit' || scale >= 1.5) {
          for (const side of [-1, 1]) {
            this.line(indices.map(i => this.project(edge(i, side, .45))), '#ecebe4', Math.max(1, scale * .17));
            this.line(indices.map(i => this.project(edge(i, side, 7.5))), '#8a9290', scale * .35);
            this.line(indices.map(i => this.project(edge(i, side, 7.5), .65)), '#dce0dc', scale * .15);
            for (let j = 0; j < indices.length - 1; j++) {
              const i = indices[j],
                k = indices[j + 1];
              if (Math.abs(run.track.curvature[i]) > .003) {
                this.polygon([edge(i, side), edge(k, side), edge(k, side, 1.05), edge(i, side, 1.05)].map(p => this.project(p)), Math.floor(g.s[i] / 4) % 2 ? '#e8e5dc' : '#b95c57');
              }
              if (i % 7 === 0) {
                const p = edge(i, side, 7.5);
                this.line([this.project(p), this.project(p, 1.2)], '#747e7a', Math.max(1, scale * .08));
              }
            }
          }
          // Continuous rubber traces follow the true circuit, not synthetic lane guides.
          for (const offset of [-.8, .8]) this.line(indices.map(i => this.project(g.point(g.s[i], offset))), '#4d5355', scale * .22);
          for (let j = 0; j < indices.length - 1; j += 2) {
            const i = indices[j];
            if (i % 13 === 0) {
              const p = g.point(g.s[i], 0);
              const q = g.point(g.s[i] + 1.2, 0);
              this.line([this.project(p), this.project(q)], '#666c6c', .55);
            }
          }
        }
      }
      // A fixed start/finish stripe makes completed circuits visually explicit.
      const finish = g.point(0);
      if (this.mode === 'circuit' || Math.hypot(finish[0] - center[0], finish[1] - center[1]) < 220) {
        const cells = 12,
          cellWidth = 2 * run.track.half_width / cells;
        for (let row = 0; row < 2; row++) for (let column = 0; column < cells; column++) {
          const a = -run.track.half_width + column * cellWidth,
            b = a + cellWidth;
          this.polygon([[row, a], [row + 1, a], [row + 1, b], [row, b]].map(([s, n]) => this.project(g.point(s, n))), (row + column) % 2 ? '#f2efe6' : '#343b40');
        }
      }
      if (this.predictions && (this.mode !== 'circuit' || scale >= 1.5)) {
        const frame = run.frames[Math.min(run.frames.length - 1, Math.floor(time / run.dt))],
          plan = frame?.plans[this.focus];
        if (plan) {
          for (let j = 0; j < plan.opponents.length; j++) {
            const braking = plan.modes[j] === 'follower_braking';
            this.line(plan.opponents[j].map(p => this.project(g.point(...p))), braking ? '#fb8a9a' : '#ffd166', 1.8, braking ? [2, 6] : [7, 6]);
          }
          if (plan.success) {
            const path = plan.xy.map(p => this.project(g.point(...p)));
            this.line(path, '#123b4966', 7);
            this.line(path, '#45d5fa', 2.6);
            for (let k = 5; k < path.length; k += 5) {
              const p = path[k];
              c.beginPath();
              c.arc(p[0], p[1], 3, 0, Math.PI * 2);
              c.fillStyle = '#b7f3ff';
              c.fill();
            }
          }
        }
      }
      const ordered = states.map((state, id) => ({
        state,
        id,
        p: g.point(state[0], state[1])
      })).sort((a, b) => this.project(a.p)[1] - this.project(b.p)[1]);
      for (const {
        state,
        id,
        p
      } of ordered) {
        if (this.mode === 'circuit' && scale < 1.5) {
          const q = this.project(p);
          c.beginPath();
          c.arc(...q, id === this.focus ? 5 : 4, 0, Math.PI * 2);
          c.fillStyle = run.vehicles[id].color;
          c.fill();
          continue;
        }
        this.car(p, p[2] + state[2], run.vehicles[id], id === this.focus, state[4]);
      }
    }
    car(p, heading, vehicle, selected, steering) {
      const c = this.ctx,
        s = this.scale,
        co = Math.cos(heading),
        si = Math.sin(heading);
      const point = (x, y, z = 0) => this.project([p[0] + co * x - si * y, p[1] + si * x + co * y], z);
      const shape = (vertices, z, color, stroke) => this.polygon(vertices.map(([x, y]) => point(x, y, z)), color, stroke, Math.max(.6, s * .04));
      const rect = (x, y, l, b, z, color) => shape([[x - l / 2, y - b / 2], [x + l / 2, y - b / 2], [x + l / 2, y + b / 2], [x - l / 2, y + b / 2]], z, color);
      const q = point(0, 0);
      if (q[0] < -40 || q[0] > this.width + 40 || q[1] < -40 || q[1] > this.height + 40) return;
      shape([[-2.6, -1.2], [2.1, -1.2], [2.6, -.7], [2.6, 1.3], [-2.4, 1.3]], 0, '#15201f38');
      for (const axle of [-1.5, 1.35]) for (const side of [-1, 1]) {
        rect(axle, side * .89, .79, .34, .16, '#161d21');
        rect(axle, side * .91, .53, .16, .28, '#323b40');
      }
      const body = [[-2.3, -.8], [-1.9, -.94], [1.45, -.94], [2.24, -.65], [2.3, 0], [2.24, .65], [1.45, .94], [-1.9, .94], [-2.3, .8]];
      shape(body, .2, '#23313a');
      const rgb = vehicle.color.slice(1).match(/../g).map(v => parseInt(v, 16)),
        shade = k => '#' + rgb.map(v => Math.round(v * k).toString(16).padStart(2, '0')).join('');
      for (let i = 0; i < body.length; i++) {
        const a = body[i],
          b = body[(i + 1) % body.length];
        this.polygon([point(...a, .2), point(...b, .2), point(...b, .65), point(...a, .65)], shade(i < 4 ? .68 : .82));
      }
      shape(body, .65, vehicle.color, '#25333c');
      rect(-.3, 0, 3.8, .19, .66, '#f0efdf');
      rect(-.3, .29, 3.8, .07, .66, '#e9e8e0');
      shape([[-1.05, -.66], [.5, -.68], [1.14, -.49], [1.14, .49], [.5, .68], [-1.05, .66]], .77, '#182d39');
      shape([[-.75, -.57], [.16, -.57], [.43, -.44], [.43, .44], [.16, .57], [-.75, .57]], 1.08, vehicle.color);
      shape([[.4, -.53], [1.06, -.43], [1.06, .43], [.4, .53]], .83, '#90b4c0');
      this.line([point(.55, -.4, .88), point(.9, .2, .88)], '#d1e4e7', Math.max(.7, s * .04));
      for (const side of [-1, 1]) this.polygon([point(-1.93, side * .65, .65), point(-1.93, side * .65, 1.05), point(-1.7, side * .65, 1.05), point(-1.7, side * .65, .65)], '#25323a');
      rect(-1.93, 0, .3, 2.02, 1.05, '#25323a');
      rect(2.06, 0, .21, 1.82, .29, '#293339');
      for (const side of [-1, 1]) {
        rect(1.9, side * .59, .15, .34, .57, '#fcf5d5');
        rect(-2.16, side * .59, .12, .32, .52, '#ef685e');
        rect(.52, side * 1.03, .28, .16, .72, '#29353d');
      }
      this.line([point(-1.7, -.79, .69), point(1.4, -.79, .69)], shade(1), Math.max(.5, s * .06));
      const label = point(-.2, 0, 1.12);
      c.font = `600 ${Math.max(8, s * .9)}px sans-serif`;
      c.fillStyle = '#fff';
      c.textAlign = 'center';
      c.fillText(vehicle.number, label[0], label[1] + s * .3);
      if (selected) {
        const marker = point(0, 0, 3.5);
        this.polygon([[marker[0] - 4, marker[1] - 6], [marker[0] + 4, marker[1] - 6], [marker[0], marker[1]]], '#ffffff', '#265572', 1);
      }
    }
  }
  window.CircuitGeometry = CircuitGeometry;
  window.RaceScene = RaceScene;
})();
