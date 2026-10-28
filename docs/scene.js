/* Replays recorded states. The orange forecast stays fixed for three seconds. */
(() => {
  const lerp = (a, b, f) => a + (b - a) * f;
  function bracket(grid, value) {
    let lo = 0, hi = grid.length - 1;
    value = Math.max(grid[lo], Math.min(grid[hi], value));
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (grid[mid] <= value) lo = mid; else hi = mid; }
    return [lo, hi, (value - grid[lo]) / (grid[hi] - grid[lo] || 1)];
  }
  function sample(run, time, opponent = null) {
    const [lo, hi, f] = bracket(run.time, time);
    const states = opponent === null ? run.states : run.opponents.map(row => row[opponent]);
    return states[lo].map((v, i) => lerp(v, states[hi][i], f));
  }
  function forecast(run, time, horizon = 3, dt = .25) {
    const origin = Math.floor(Math.max(0, time - 1e-8) / horizon) * horizon;
    const plan = run.plans[Math.round(origin / dt)];
    if (!plan || !plan.success) return null;
    const elapsed = Math.max(0, Math.min(horizon, time - origin));
    const index = Math.min(plan.states.length - 2, Math.floor(elapsed / dt));
    const f = Math.min(1, (elapsed - index * dt) / dt);
    return {plan, origin, elapsed, state: plan.states[index].map((v, i) => lerp(v, plan.states[index + 1][i], f))};
  }
  class DrivingScene {
    constructor(canvas, data) {
      this.canvas = canvas; this.ctx = canvas.getContext('2d'); this.data = data;
      this.view = 'follow'; this.focus = null; this.resize();
      if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => this.resize()).observe(canvas);
    }
    resize() {
      const bounds = this.canvas.getBoundingClientRect(); this.w = bounds.width; this.h = bounds.height;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      this.canvas.width = this.w * ratio; this.canvas.height = this.h * ratio;
      this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0); this.dirty = true;
    }
    project(x, y, z = 0) {
      const dx = x - this.focus[0], dy = y - this.focus[1];
      return [this.w / 2 + (dx * .973 - dy * .231) * this.scale,
        this.h * .52 - (dx * .231 + dy * .973) * this.scale * .9 - z * this.scale * .85];
    }
    poly(points, color) {
      const c = this.ctx; c.beginPath();
      points.forEach((v, i) => { const p = this.project(...v); i ? c.lineTo(...p) : c.moveTo(...p); });
      c.closePath(); c.fillStyle = color; c.fill();
    }
    line(points, color, width = 1, dash = []) {
      const c = this.ctx; c.beginPath();
      points.forEach((v, i) => { const p = this.project(...v); i ? c.lineTo(...p) : c.moveTo(...p); });
      c.strokeStyle = color; c.lineWidth = width; c.setLineDash(dash); c.stroke(); c.setLineDash([]);
    }
    label(x, y, text, color = '#526170', lift = 26) {
      const c = this.ctx, p = this.project(x, y, 1);
      if (p[0] < 0 || p[0] > this.w || p[1] < 0 || p[1] > this.h) return;
      c.font = '14px sans-serif'; const width = c.measureText(text).width + 16;
      c.fillStyle = '#fffffff2'; c.fillRect(p[0] - width / 2, p[1] - lift - 21, width, 26);
      c.fillStyle = color; c.textAlign = 'center'; c.fillText(text, p[0], p[1] - lift - 3);
    }
    world(s, n = 0) {
      const t = this.track, [a, b, f] = bracket(t.s, s);
      const x = lerp(t.xy[a][0], t.xy[b][0], f), y = lerp(t.xy[a][1], t.xy[b][1], f);
      const dx = lerp(t.left[a][0] - t.xy[a][0], t.left[b][0] - t.xy[b][0], f);
      const dy = lerp(t.left[a][1] - t.xy[a][1], t.left[b][1] - t.xy[b][1], f);
      const width = Math.hypot(dx, dy);
      return [x + dx * n / width, y + dy * n / width];
    }
    pose(state) {
      const [x, y] = this.world(state[0], state[1]), p = this.world(state[0] + .1, 0), q = this.world(state[0] - .1, 0);
      return {x, y, heading: Math.atan2(p[1] - q[1], p[0] - q[0]) + state[2]};
    }
    box(x,y,length,width,height,color,heading=0){const sc=Math.cos(heading),sn=Math.sin(heading),p=(u,v,z)=>[x+u*sc-v*sn,y+u*sn+v*sc,z];
      this.poly([p(-length/2,-width/2,0),p(length/2,-width/2,0),p(length/2,-width/2,height),p(-length/2,-width/2,height)],color);
      this.poly([p(length/2,-width/2,0),p(length/2,width/2,0),p(length/2,width/2,height),p(length/2,-width/2,height)],'#526170');
      this.poly([p(-length/2,-width/2,height),p(length/2,-width/2,height),p(length/2,width/2,height),p(-length/2,width/2,height)],color);
      this.poly([p(-length/2,-width/2,height),p(length/2,-width/2,height),p(length/2,width/2,height),p(-length/2,width/2,height)],'#ffffff22');
    }
    car(state,color,ghost=false){const {x,y,heading}=state,sc=Math.cos(heading),sn=Math.sin(heading),p=(u,v,z)=>[x+u*sc-v*sn,y+u*sn+v*sc,z],c=this.ctx;
      c.save();if(ghost)c.globalAlpha=.55;
      this.poly([p(-2.6,-1.1,0),p(2.8,-1.1,0),p(2.8,1.3,0),p(-2.6,1.3,0)],'#26374733');
      for(const u of [-1.4,1.35])for(const v of [-.85,.85])this.poly([p(u-.4,v-.16,.3),p(u+.4,v-.16,.3),p(u+.4,v+.16,.3),p(u-.4,v+.16,.3)],'#24323c');
      this.poly([p(-2.3,-.85,.45),p(1.9,-.85,.45),p(2.3,-.5,.45),p(2.3,.5,.45),p(1.9,.85,.45),p(-2.3,.85,.45)],color);
      this.poly([p(-1.3,-.8,.5),p(1.2,-.75,.5),p(.65,-.62,1.25),p(-1,-.62,1.25)],color);
      this.poly([p(-1,-.62,1.25),p(.65,-.62,1.25),p(.65,.62,1.25),p(-1,.62,1.25)],color);
      this.poly([p(.65,-.62,1.25),p(1.45,-.73,.5),p(1.45,.73,.5),p(.65,.62,1.25)],'#334b61');
      this.poly([p(-.9,-.5,1.27),p(.15,-.5,1.27),p(.15,.5,1.27),p(-.9,.5,1.27)],'#263f53');
      for(const v of [-.62,.62])this.line([p(2.05,v,.55),p(2.23,v*.6,.55)],'#fff4ce',Math.max(1,this.scale*.13));
      if(ghost)this.line([p(-2.8,-1.3,.05),p(2.8,-1.3,.05),p(2.8,1.3,.05),p(-2.8,1.3,.05),p(-2.8,-1.3,.05)],color,1.5,[5,4]);
      c.restore();
    }
    tree(x,y,h){const p=this.project(x,y);if(p[0]<-40||p[0]>this.w+40||p[1]<-40||p[1]>this.h+40)return;
      this.poly([[x-1,y-1,0],[x+3,y-1,0],[x+4,y+1,0],[x,y+2,0]],'#7d8d7929');
      this.poly([[x-1.5,y-1.5,.4],[x+1.5,y-1.5,.4],[x,y,h]],'#9fb29a');
      this.poly([[x+1.5,y-1.5,.4],[x+1.5,y+1.5,.4],[x,y,h]],'#b7c7ac');
      this.poly([[x-1.5,y-1.5,.4],[x-1.5,y+1.5,.4],[x,y,h]],'#8ea186');
    }

    road(state) {
      const t = this.track;
      this.poly([...t.left, ...t.right.slice().reverse()], '#dce2e4');
      this.line(t.left, '#9eaba7', 2); this.line(t.right, '#9eaba7', 2);
      this.line(t.xy, '#ffffff', 2, [10, 14]);
      const start = this.view === 'follow' ? state[0] - 100 : -50;
      const end = this.view === 'follow' ? state[0] + 150 : 600;
      for (let s = Math.floor(start / 12) * 12; s < end; s += 12) {
        for (const n of [-7, 7]) this.line([this.world(s, n), this.world(s + 5, n)], '#ffffff', 4);
      }
      for (let s = Math.floor(start / 32) * 32; s < end; s += 32) {
        this.tree(...this.world(s, 20), 3); this.tree(...this.world(s + 12, -23), 3.5);
      }
      for (let s = Math.floor(start / 50) * 50; s < end; s += 50) if (Math.abs(s - state[0]) > 18) this.label(...this.world(s, -10), `${s} m`, '#64726d', 0);
    }
    render(scenario, run, time, {ghost = true, dt = 1 / 60} = {}) {
      this.track = run.track;
      const state = sample(run, time), pose = this.pose(state), c = this.ctx;
      const target = this.view === 'follow' ? this.world(state[0] + 20, 0) : this.world(240, 0);
      const desired = this.view === 'follow' ? Math.min(this.w / 112, this.h / 55) : Math.min(this.w / 680, this.h / 230);
      if (!this.focus) { this.focus = target; this.scale = desired; }
      const blend = 1 - Math.exp(-dt / .3);
      this.focus = this.focus.map((v, i) => lerp(v, target[i], blend)); this.scale = lerp(this.scale, desired, blend);
      this.moving = Math.hypot(this.focus[0] - target[0], this.focus[1] - target[1]) > .01 || Math.abs(this.scale - desired) > .001;
      c.clearRect(0, 0, this.w, this.h); c.fillStyle = '#f3f6f0'; c.fillRect(0, 0, this.w, this.h);
      this.road(state);
      const index = bracket(run.time, time)[0];
      this.line(run.states.slice(0, index + 1).map(s => this.world(s[0], s[1])).concat([[pose.x, pose.y]]), '#2166ac', 2.5);
      const fixed = forecast(run, time, this.data.horizon, this.data.dt);
      this.lastForecast = fixed; this.lastError = null;
      if (fixed && ghost) {
        this.line(fixed.plan.states.map(s => this.world(s[0], s[1])), '#b44819', 2.5, [8, 5]);
        const predicted = this.pose(fixed.state);
        this.car(predicted, '#c46b39', true); this.label(predicted.x, predicted.y, 'Fixed forecast', '#a54720', 100);
        this.line([[predicted.x, predicted.y, 1.3], [pose.x, pose.y, 1.3]], '#a54720', 2, [3, 3]);
        this.lastError = Math.hypot(predicted.x - pose.x, predicted.y - pose.y);
      }
      for (let i = 0; i < 3; i++) {
        const opponent = this.pose(sample(run, time, i));
        this.car(opponent, ['#72877c', '#7a8193', '#95867e'][i]);
        this.label(opponent.x, opponent.y, `Opponent ${i + 1}`, '#526170', 20);
      }
      this.car(pose, '#2166ac'); this.label(pose.x, pose.y, 'Ego · actual', '#2166ac', -75);
      this.lastState = state; this.dirty = false;
    }
  }
  window.DrivingScene = DrivingScene; window.sampleRun = sample; window.fixedForecast = forecast;
})();
