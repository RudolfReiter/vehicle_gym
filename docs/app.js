/* A recorded experiment viewer: playback never runs a controller in the browser. */
(function () {
  'use strict';

  const $ = id => document.getElementById(id),
    manifest = window.RACE_MANIFEST,
    scene = new window.RaceScene($('scene'));
  let run = null,
    time = 0,
    playing = false,
    last = null,
    request = 0,
    selected = 0,
    plotCache = null,
    uiKey = '';
  const methods = [...new Set(manifest.runs.map(r => r.method))];
  const option = (value, label) => {
    const o = document.createElement('option');
    o.value = value;
    o.textContent = label;
    return o;
  };
  methods.forEach(m => $('method').append(option(m, m)));
  if (manifest.kind === 'hilepp') {
    $('execution').replaceChildren();
    const modes = [...new Set(manifest.runs.map(run => run.execution))];
    modes.forEach(mode => $('execution').append(option(mode, mode === 'sampled' ? 'Sampled policy' : 'Mean action')));
    $('execution').disabled = modes.length < 2;
  }
  [...new Set(manifest.runs.map(r => r.track))].forEach(t => $('track').append(option(t, t === 'Oval' ? 'Oval · repeated laps' : t === 'Sepang' ? 'Sepang · stress test' : t)));
  const outcome = m => m.completed ? 'Completed' : ({
    collision: 'Collision',
    track_exit: 'Track exit',
    solver_failure: 'Solver failure'
  }[m.stop.kind] || m.stop.kind) + (m.stop.vehicles.length ? ' · car ' + m.stop.vehicles.map(i => String(i + 1).padStart(2, '0')).join(', ') : '');
  function actorExecution() {
    return manifest.kind === 'hilepp' ? $('execution').value : null;
  }
  function matchesExecution(entry) {
    return manifest.kind !== 'hilepp' || entry.execution === actorExecution();
  }
  function renderResults() {
    $('results-body').replaceChildren();
    for (const record of manifest.evaluation.filter(row => manifest.kind !== "hilepp" || row.execution === actorExecution())) {
      const row = document.createElement('tr'),
        m = record.metrics;
      for (const value of [record.track, record.method, record.seed, `${m.duration_s.toFixed(1)} s`, `${m.progress_m.toFixed(0)} m · ${m.laps.toFixed(1)} laps`, `${m.rank} / 6`, m.passes, `${m.observed_envelopes[0].speed_excess.toFixed(2)} m/s`, `${m.observed_envelopes[0].lateral_excess.toFixed(2)} m/s²`, outcome(m)]) {
        const cell = document.createElement('td');
        cell.textContent = value;
        row.append(cell);
      }
      $('results-body').append(row);
    }
    if (manifest.kind === 'hilepp') {
      $('results-mode').textContent = actorExecution() === 'sampled' ? '· sampled policies' : '· mean actions';
      $('results-figure').src = `assets/evaluation-${actorExecution()}.svg`;
    }
  }
  renderResults();
  if (manifest.training) {
    $('training-note').textContent = Object.entries(manifest.training).map(([method, t]) => `${method}: ${t.steps.toLocaleString()} training interactions, ${t.gradient_updates.toLocaleString()} SAC updates${t.pretraining ? `, stage seeds ${t.pretraining.training_seed} and ${t.training_seed}` : `, seed ${t.training_seed}`}`).join('. ') + '. Both actors and twin critics use Deep Sets. The selector exposes sampled and mean-action execution from the same checkpoint.';
  }
  function seeds() {
    const current = $('seed').value;
    $('seed').replaceChildren();
    manifest.runs.filter(r => r.method === $('method').value && r.track === $('track').value && matchesExecution(r)).forEach(r => $('seed').append(option(r.seed, r.seed)));
    if ([...$('seed').options].some(o => o.value === current)) $('seed').value = current;
    load();
  }
  async function load() {
    const token = ++request;
    playing = false;
    updatePlay();
    $('load-status').hidden = false;
    $('load-status').textContent = 'Loading recorded race…';
    const record = manifest.runs.find(r => r.track === $('track').value && r.method === $('method').value && String(r.seed) === $('seed').value && matchesExecution(r));
    try {
      const response = await fetch(`data/${record.file}`);
      if (!response.ok) throw new Error(`Replay unavailable (HTTP ${response.status})`);
      const bytes = await response.arrayBuffer();
      let content;
      if (new Uint8Array(bytes)[0] === 31) {
        if (typeof DecompressionStream === 'undefined') throw new Error('This browser needs gzip stream support. Try a current Firefox, Safari or Chromium browser.');
        content = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
      } else content = JSON.parse(new TextDecoder().decode(bytes));
      if (token !== request) return;
      run = content;
      scene.setRun(run);
      time = 0;
      selected = 0;
      scene.focus = 0;
      plotCache = null;
      uiKey = '';
      $('event').replaceChildren(option(0, 'Race start'));
      run.events.filter(event => event.kind === 'pass').forEach(event => $('event').append(option(Math.max(0, event.time - 3), `${event.time.toFixed(1)} s · car ${event.vehicle + 1} passes car ${event.opponent + 1}${event.lap_traffic ? ' · lap traffic' : ''}`)));
      $('focus').replaceChildren();
      run.vehicles.forEach(v => $('focus').append(option(v.id, `${v.number} · ${v.controller}`)));
      $('timeline').max = run.metrics.duration_s;
      $('timeline').value = 0;
      $('download').href = `data/${record.file}`;
      $('circuit-name').textContent = `${run.track.name} · ${(run.track.length / 1000).toFixed(2)} km`;
      $('vehicle-limits').replaceChildren();
      for (const vehicle of run.vehicles) {
        const row = document.createElement('tr'),
          limits = vehicle.limits;
        const range = (a, b) => `${a.toFixed(1)} … ${b.toFixed(1)}`;
        for (const value of [vehicle.number, range(limits.speed_min, limits.speed_max), range(limits.longitudinal_min, limits.longitudinal_max), range(limits.lateral_min, limits.lateral_max)]) {
          const cell = document.createElement('td');
          cell.textContent = value;
          row.append(cell);
        }
        $('vehicle-limits').append(row);
      }
      $('race-summary').textContent = `${run.metrics.laps.toFixed(2)} ego laps · ${run.metrics.total_passes} passes across the field · ${run.metrics.lap_traffic_passes} involving lap traffic`;
      $('protocol').replaceChildren();
      for (const item of [`6 cars`, `Replan every ${run.dt.toFixed(1)} s`, `${run.horizon.toFixed(1)} s horizon`, `${run.planner.intervals} shooting intervals`, `${run.duration_limit_s.toFixed(0)} s time limit`]) {
        const span = document.createElement('span');
        span.textContent = item;
        $('protocol').append(span);
      }
      $('load-status').hidden = true;
      draw();
    } catch (error) {
      if (token === request) $('load-status').textContent = `${error.message}. Serve this page over HTTP using the repository preview command.`;
    }
  }
  function updatePlay() {
    $('play').textContent = playing ? '❚❚ Pause' : '▶ Play';
    $('play').setAttribute('aria-label', playing ? 'Pause recorded race' : 'Play recorded race');
  }
  function focus(id) {
    selected = +id;
    scene.focus = selected;
    $('focus').value = selected;
    draw();
  }
  function fit(canvas) {
    const c = canvas.getContext('2d'),
      box = canvas.getBoundingClientRect(),
      dpr = Math.min(2, window.devicePixelRatio || 1);
    if (canvas.width !== Math.round(box.width * dpr) || canvas.height !== Math.round(box.height * dpr)) {
      canvas.width = Math.round(box.width * dpr);
      canvas.height = Math.round(box.height * dpr);
    }
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [c, box.width, box.height];
  }
  function map(states) {
    const [c, w, h] = fit($('map')),
      g = scene.geometry,
      b = g.bounds,
      pad = 14,
      scale = Math.min((w - 2 * pad) / (b[1] - b[0]), (h - 2 * pad) / (b[3] - b[2]));
    const p = xy => [w / 2 + (xy[0] - (b[0] + b[1]) / 2) * scale, h / 2 - (xy[1] - (b[2] + b[3]) / 2) * scale];
    c.clearRect(0, 0, w, h);
    c.beginPath();
    g.xy.forEach((xy, i) => i ? c.lineTo(...p(xy)) : c.moveTo(...p(xy)));
    c.strokeStyle = '#abb4b4';
    c.lineWidth = 3;
    c.stroke();
    states.forEach((x, i) => {
      const q = p(g.point(x[0], x[1]));
      c.beginPath();
      c.arc(...q, i === selected ? 4.3 : 3.1, 0, 2 * Math.PI);
      c.fillStyle = run.vehicles[i].color;
      c.fill();
      if (i === selected) {
        c.strokeStyle = 'white';
        c.lineWidth = 1.3;
        c.stroke();
      }
    });
  }
  function valueGrid(entries) {
    const grid = document.createElement('div');
    grid.className = 'values';
    for (const [label, value] of entries) {
      const div = document.createElement('div');
      div.className = 'value';
      const span = document.createElement('span');
      span.textContent = label;
      div.append(span, document.createTextNode(value));
      grid.append(div);
    }
    return grid;
  }
  function plots() {
    const maxTime = Math.max(run.dt, run.metrics.duration_s);
    if (!plotCache) plotCache = {};
    for (const kind of ['speed', 'rank']) {
      const [canvasContext, w, h] = fit($(kind + '-plot')),
        left = 32,
        right = w - 8,
        top = 10,
        bottom = h - 23;
      const key = `${selected}/${w}/${h}/${window.devicePixelRatio || 1}`;
      let cache = plotCache[kind];
      if (!cache || cache.key !== key) {
        const background = document.createElement('canvas'),
          dpr = Math.min(2, window.devicePixelRatio || 1);
        background.width = Math.round(w * dpr);
        background.height = Math.round(h * dpr);
        const c = background.getContext('2d');
        c.setTransform(dpr, 0, 0, dpr, 0, 0);
        const ymax = kind === 'speed' ? Math.ceil(Math.max(...run.states.flatMap(states => states.map(state => state[3] * 3.6))) / 20) * 20 : 6;
        const x = t => left + (right - left) * t / maxTime,
          y = v => kind === 'speed' ? bottom - (bottom - top) * v / ymax : top + (bottom - top) * (v - 1) / 5;
        c.clearRect(0, 0, w, h);
        c.font = '11px sans-serif';
        c.fillStyle = '#75818a';
        c.lineWidth = 1;
        for (const tick of kind === 'speed' ? [0, ymax / 2, ymax] : [1, 3, 6]) {
          c.beginPath();
          c.moveTo(left, y(tick));
          c.lineTo(right, y(tick));
          c.strokeStyle = '#e5eaed';
          c.stroke();
          c.textAlign = 'right';
          c.fillText(String(tick), left - 7, y(tick) + 4);
        }
        for (const tick of [0, maxTime / 2, maxTime]) {
          c.textAlign = tick === 0 ? 'left' : tick === maxTime ? 'right' : 'center';
          c.fillText(`${tick.toFixed(0)} s`, x(tick), h - 5);
        }
        for (let id = 0; id < 6; id++) {
          c.beginPath();
          run.states.forEach((states, k) => {
            const value = kind === 'speed' ? states[id][3] * 3.6 : 1 + states.filter((state, j) => j !== id && state[0] > states[id][0]).length;
            const point = [x(k * run.dt), y(value)];
            k ? c.lineTo(...point) : c.moveTo(...point);
          });
          c.lineWidth = id === selected ? 2 : 1.2;
          c.globalAlpha = id === selected ? 1 : .7;
          c.strokeStyle = run.vehicles[id].color;
          c.stroke();
        }
        c.globalAlpha = 1;
        cache = plotCache[kind] = {
          key,
          background
        };
      }
      canvasContext.clearRect(0, 0, w, h);
      canvasContext.drawImage(cache.background, 0, 0, w, h);
      const cursor = left + (right - left) * time / maxTime;
      canvasContext.beginPath();
      canvasContext.moveTo(cursor, top);
      canvasContext.lineTo(cursor, bottom);
      canvasContext.strokeStyle = '#253743';
      canvasContext.lineWidth = 1;
      canvasContext.stroke();
    }
  }
  function parameterPlots() {
    if (manifest.kind !== 'hilepp') return;
    const duration = Math.max(run.dt, run.metrics.duration_s);
    for (const kind of ['lateral', 'weights']) {
      const [target, w, h] = fit($(kind + '-plot')),
        left = 36,
        right = w - 8,
        top = 10,
        bottom = h - 24,
        key = `${selected}/${w}/${h}`;
      let cache = plotCache[kind];
      if (!cache || cache.key !== key) {
        const background = document.createElement('canvas'),
          dpr = Math.min(2, window.devicePixelRatio || 1);
        background.width = Math.round(w * dpr);
        background.height = Math.round(h * dpr);
        const c = background.getContext('2d');
        c.setTransform(dpr, 0, 0, dpr, 0, 0);
        c.font = '11px sans-serif';
        c.fillStyle = '#75818a';
        if (run.frames[0].plans[selected] === null) {
          c.fillText('Direct SAC: no MPC references or cost weights', left, h / 2);
        } else {
          const x = t => left + (right - left) * t / duration,
            y = v => kind === 'lateral' ? bottom - (bottom - top) * (v + 5) / 10 : bottom - (bottom - top) * (Math.log10(v) + 1) / Math.log10(200);
          for (const tick of kind === 'lateral' ? [-5, 0, 5] : [.1, 1, 10]) {
            c.beginPath();
            c.moveTo(left, y(tick));
            c.lineTo(right, y(tick));
            c.strokeStyle = '#e5eaed';
            c.lineWidth = 1;
            c.stroke();
            c.textAlign = 'right';
            c.fillText(String(tick), left - 7, y(tick) + 4);
          }
          for (let series = 0; series < 2; series++) {
            c.beginPath();
            run.frames.forEach((frame, k) => {
              const ref = frame.references[selected];
              const value = kind === 'lateral' ? series === 0 ? ref.lateral : run.states[k][selected][1] : series === 0 ? ref.speed_weight : ref.lateral_weight;
              const p = [x(frame.time), y(value)];
              k ? c.lineTo(...p) : c.moveTo(...p);
            });
            c.strokeStyle = series === 0 ? '#176da5' : '#b78035';
            c.lineWidth = 1.6;
            c.setLineDash(kind === 'lateral' && series === 1 ? [4, 3] : []);
            c.stroke();
            c.setLineDash([]);
          }
          for (const tick of [0, duration / 2, duration]) {
            c.textAlign = tick === 0 ? 'left' : tick === duration ? 'right' : 'center';
            c.fillText(`${tick.toFixed(0)} s`, x(tick), h - 5);
          }
        }
        cache = plotCache[kind] = {
          key,
          background
        };
      }
      target.clearRect(0, 0, w, h);
      target.drawImage(cache.background, 0, 0, w, h);
      const x = left + (right - left) * time / duration;
      target.beginPath();
      target.moveTo(x, top);
      target.lineTo(x, bottom);
      target.lineWidth = 1;
      target.strokeStyle = '#253743';
      target.stroke();
    }
  }
  function draw() {
    if (!run) return;
    scene.render(time);
    const states = scene.geometry.states(run, time),
      frame = run.frames[Math.min(run.frames.length - 1, Math.floor(time / run.dt))],
      plan = frame.plans[selected],
      ref = frame.references[selected],
      car = run.vehicles[selected];
    map(states);
    plots();
    parameterPlots();
    const nextUiKey = `${run.track.name}/${run.method}/${run.seed}/${selected}/${Math.floor(time * 10)}`;
    if (uiKey === nextUiKey) return;
    uiKey = nextUiKey;
    const ordered = states.map((state, id) => ({
      state,
      id
    })).sort((a, b) => b.state[0] - a.state[0]);
    $('standings').replaceChildren();
    ordered.forEach(({
      state,
      id
    }, rank) => {
      const button = document.createElement('button');
      button.className = 'standing' + (id === selected ? ' selected' : '');
      button.type = 'button';
      button.setAttribute('aria-label', `Observe ${run.vehicles[id].name}, position ${rank + 1}`);
      const pos = document.createElement('span');
      pos.textContent = rank + 1;
      const swatch = document.createElement('span');
      swatch.className = 'swatch';
      swatch.style.background = run.vehicles[id].color;
      const name = document.createElement('span');
      name.textContent = `${run.vehicles[id].number} · ${run.vehicles[id].controller}`;
      const gap = document.createElement('span');
      gap.className = 'gap';
      const distance = ordered[0].state[0] - state[0],
        lapsBehind = Math.floor(distance / run.track.length);
      gap.textContent = !rank ? 'Leader' : lapsBehind ? `+${lapsBehind} L · ${(distance % run.track.length).toFixed(0)} m` : `+${distance.toFixed(0)} m`;
      button.append(pos, swatch, name, gap);
      button.onclick = () => focus(id);
      $('standings').append(button);
    });
    $('race-clock').textContent = `${Math.floor(time / 60)}:${(time % 60).toFixed(1).padStart(4, '0')}`;
    $('race-state').textContent = time >= run.metrics.duration_s ? outcome(run.metrics) : `Seed ${run.seed} · Lap ${Math.floor(Math.max(0, states[selected][0]) / run.track.length) + 1}`;
    $('time-label').textContent = `${time.toFixed(1)} / ${run.metrics.duration_s.toFixed(1)} s`;
    $('timeline').value = time;
    $('controller-title').textContent = `${car.number} · ${car.controller}${car.controller === 'HILEPP' ? ' (HILEPP-II)' : ''}`;
    const entries = [['Measured speed', `${(states[selected][3] * 3.6).toFixed(1)} km/h`], ['Force', `${(frame.commands[selected][0] / 1000).toFixed(2)} kN`], ['Steering rate', `${frame.commands[selected][1].toFixed(3)} rad/s`]];
    const physical = run.vehicle_model;
    const speed = states[selected][3];
    entries.push(['Longitudinal acceleration', `${((frame.commands[selected][0] - physical.drag_coefficient * speed * speed - physical.rolling_force_n) / physical.mass_kg).toFixed(2)} m/s²`], ['Lateral acceleration', `${(speed * speed * Math.tan(states[selected][4]) / physical.wheelbase_m).toFixed(2)} m/s²`]);
    if (plan) entries.push(['Target speed', `${(ref.speed * 3.6).toFixed(1)} km/h`], ['Lateral reference', `${ref.lateral.toFixed(2)} m`], ['Weights · speed / lateral', `${ref.speed_weight.toFixed(2)} / ${ref.lateral_weight.toFixed(2)}`]);
    $('controller-values').replaceChildren(valueGrid(entries));
    $('horizon-note').textContent = plan ? `Current solve at t = ${frame.time.toFixed(1)} s · ${plan.solve_ms.toFixed(1)} ms · ${plan.success ? 'accepted' : 'failed'} · obstacle slack ${plan.obstacle_slack.toExponential(1)}. ${frame.applied ? 'Only its first control input is applied.' : 'This step was not applied: the race stopped.'}` : 'Direct SAC acts on force and steering rate. This car has no MPC horizon or obstacle-prediction module.';
    $('opponent-modes').replaceChildren();
    if (plan) plan.opponent_ids.forEach((id, j) => {
      const row = document.createElement('div');
      row.className = 'mode-row';
      const name = document.createElement('span');
      name.textContent = `Car ${run.vehicles[id].number}`;
      const gap = document.createElement('span');
      gap.textContent = `${(plan.opponents[j][0][0] - run.states[Math.min(run.states.length - 1, Math.floor(time / run.dt))][selected][0]).toFixed(1)} m`;
      const mode = document.createElement('span'),
        braking = plan.modes[j] === 'follower_braking';
      mode.className = braking ? 'braking' : 'constant';
      mode.textContent = braking ? `Braking ${run.vehicles[id].limits.longitudinal_min.toFixed(1)} m/s² · behind` : 'Constant speed · ahead / alongside';
      row.append(name, gap, mode);
      $('opponent-modes').append(row);
    });else $('opponent-modes').textContent = 'No MPC predictions for direct SAC. Select an opponent to inspect its controller.';
  }
  $('event').onchange = () => {
    time = +$('event').value;
    draw();
  };
  if (manifest.kind === 'hilepp') $('execution').onchange = () => {
    renderResults();
    seeds();
  };
  $('method').onchange = seeds;
  $('track').onchange = seeds;
  $('seed').onchange = load;
  $('focus').onchange = () => focus($('focus').value);
  $('camera').onchange = () => {
    scene.mode = $('camera').value;
    draw();
  };
  $('zoom').onchange = () => {
    scene.zoom = +$('zoom').value;
    draw();
  };
  $('predictions').onchange = () => {
    scene.predictions = $('predictions').checked;
    draw();
  };
  $('play').onclick = () => {
    if (!run) return;
    if (time >= run.metrics.duration_s) time = 0;
    playing = !playing;
    last = null;
    updatePlay();
    draw();
  };
  $('timeline').oninput = () => {
    time = +$('timeline').value;
    draw();
  };
  $('fullscreen').onclick = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();else await document.querySelector('.race').requestFullscreen();
    } catch {}
  };
  window.addEventListener('resize', draw);
  document.addEventListener('visibilitychange', () => {
    last = null;
  });
  function tick(now) {
    if (playing && run) {
      if (last !== null && !document.hidden) time = Math.min(run.metrics.duration_s, time + Math.min((now - last) / 1000, .08) * +$('rate').value);
      if (time >= run.metrics.duration_s) {
        playing = false;
        updatePlay();
      }
      draw();
    }
    last = now;
    requestAnimationFrame(tick);
  }
  seeds();
  requestAnimationFrame(tick);
  window.RaceViewer = {
    setTime(t) {
      time = Math.max(0, Math.min(run.metrics.duration_s, t));
      draw();
    },
    getState() {
      return {
        time,
        playing,
        selected,
        execution: actorExecution(),
        method: run?.method
      };
    }
  };
})();
