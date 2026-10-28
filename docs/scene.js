/* Recorded simulation with live MPC plans and separately labeled opponent assumptions. */
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
  function currentPlan(run, time, dt = .25) {
    const index = Math.min(run.plans.length - 1, Math.max(0, Math.floor((time + 1e-9) / dt)));
    return {index, plan: run.plans[index]};
  }
  function interpolatePlan(states, elapsed, dt = .25) {
    const node = Math.min(states.length - 2, Math.max(0, Math.floor(elapsed / dt)));
    const f = Math.max(0, Math.min(1, (elapsed - node * dt) / dt));
    return states[node].map((value, i) => lerp(value, states[node + 1][i], f));
  }
  function opponentAudit(run, time, opponent, horizon = 3, dt = .25) {
    const origin = Math.floor(Math.max(0, time - 1e-8) / horizon) * horizon;
    const plan = run.plans[Math.round(origin / dt)];
    if (!plan || !plan.opponent_predictions?.[opponent]) return null;
    const elapsed = Math.max(0, Math.min(horizon, time - origin));
    return {origin, elapsed, mode: plan.opponent_modes[opponent],
      states: plan.opponent_predictions[opponent],
      state: interpolatePlan(plan.opponent_predictions[opponent], elapsed, dt)};
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
    envelope(state, color) {
      const points = Array.from({length: 49}, (_, i) => {
        const angle = i / 48 * Math.PI * 2;
        return this.world(state[0] + 7 * Math.cos(angle), state[1] + 3 * Math.sin(angle));
      });
      this.poly(points, '#b4481912'); this.line(points, color, 1.2, [4, 5]);
    }
    render(scenario, run, time, {overlay = 'mpc', opponent = 0, probe = 1, envelopes = true, dt = 1 / 60} = {}) {
      this.track = run.track;
      const state = sample(run, time), pose = this.pose(state), c = this.ctx;
      const midpoint = (run.track.s[0] + run.track.s.at(-1)) / 2;
      const target = this.view === 'follow' ? this.world(state[0] + 14, 0) : this.world(midpoint, 0);
      const desired = this.view === 'follow' ? Math.min(this.w / 112, this.h / 55) : Math.min(this.w / (run.track.s.at(-1) - run.track.s[0]), this.h / 230);
      if (!this.focus) { this.focus = target; this.scale = desired; }
      const blend = 1 - Math.exp(-dt / .3);
      this.focus = this.focus.map((v, i) => lerp(v, target[i], blend)); this.scale = lerp(this.scale, desired, blend);
      this.moving = Math.hypot(this.focus[0] - target[0], this.focus[1] - target[1]) > .01 || Math.abs(this.scale - desired) > .001;
      c.clearRect(0, 0, this.w, this.h); c.fillStyle = '#f3f6f0'; c.fillRect(0, 0, this.w, this.h);
      this.road(state);
      const index = bracket(run.time, time)[0];
      this.line(run.states.slice(Math.max(0,index-20), index+1).map(s=>this.world(s[0],s[1])).concat([[pose.x,pose.y]]), '#2166ac77', 2);
      const current = currentPlan(run, time, this.data.dt);
      this.lastPlan = current; this.lastAudit = null; this.lastError = null;
      if (current.plan?.success && overlay === 'mpc') {
        const plan = current.plan;
        this.line(plan.states.map(s => this.world(s[0], s[1])), '#2166ac', 3);
        const future = this.pose(interpolatePlan(plan.states, probe, this.data.dt));
        this.car(future, '#2166ac', true);
        this.label(future.x, future.y, `Ego plan +${probe.toFixed(1)} s`, '#2166ac', -45);
        plan.opponent_predictions.forEach((states, i) => {
          const p = this.pose(interpolatePlan(states, probe, this.data.dt));
          this.line(states.map(s => this.world(s[0], s[1])), '#b4481988', i === opponent ? 2.4 : 1.5, [7,5]);
          if (i === opponent) {
            if (envelopes) this.envelope(interpolatePlan(states, probe, this.data.dt), '#b44819');
            this.car(p, '#c46b39', true);
            this.label(p.x, p.y, `Opponent ${i+1} +${probe.toFixed(1)} s`, '#a54720', 80);
          }
        });
      } else if (overlay === 'audit') {
        const audit = opponentAudit(run, time, opponent, this.data.horizon, this.data.dt);
        this.lastAudit = audit;
        if (audit) {
          const predicted = this.pose(audit.state), actual = this.pose(sample(run, time, opponent));
          this.line(audit.states.map(s => this.world(s[0],s[1])), '#b44819', 2, [7,5]);
          this.car(predicted, '#c46b39', true);
          this.label(predicted.x, predicted.y, `Opponent ${opponent+1} · held assumption`, '#a54720', 85);
          this.line([[predicted.x,predicted.y,1], [actual.x,actual.y,1]], '#b44819', 2, [3,3]);
          this.lastError = Math.hypot(predicted.x-actual.x,predicted.y-actual.y);
        }
      }
      for (let i=0;i<3;i++) {
        const actual = this.pose(sample(run,time,i));
        this.car(actual, ['#72877c','#7a8193','#95867e'][i]);
        this.label(actual.x,actual.y,`Opponent ${i+1} · actual`,'#526170',24);
      }
      this.car(pose, '#2166ac'); this.label(pose.x,pose.y,'Ego · actual','#2166ac',-75);
      this.lastState=state;this.dirty=false;
    }
  }
  window.DrivingScene = DrivingScene; window.sampleRun = sample; window.currentPlan = currentPlan; window.opponentAudit = opponentAudit; window.interpolatePlan = interpolatePlan;
})();
