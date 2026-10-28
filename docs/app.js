(() => {
  'use strict';
  const data = window.EXPERIMENT, $ = id => document.getElementById(id);
  const scene = new window.DrivingScene($('scene'), data);
  const names = {overtaking: 'Overtaking · three slower opponents', blocking: 'Blocking · three faster opponents', mixed: 'Mixed traffic · opponents ahead and behind'};
  let scenario = data.scenarios[0], run = scenario.runs[data.kind === 'hilepp' ? 2 : 0];
  let time = 0, playing = false, lastFrame = null;
  $('scenario').innerHTML = data.scenarios.map(s => `<option value="${s.id}">${s.id[0].toUpperCase()+s.id.slice(1)}</option>`).join('');
  function selectRun() {
    run = scenario.runs.find(r => r.id === $('method').value) || scenario.runs[0];
    time = 0; playing = false; lastFrame = null; scene.focus = null;
    $('play').textContent = 'Play'; $('seek').max = run.time.at(-1); $('seek').value = 0;
    $('scene-title').textContent = names[scenario.id];
    $('scene-subtitle').textContent = `${run.name} · seed ${run.seed} · 16 s episode limit`;
    const m = run.metrics;
    $('run-status').textContent = `${m.completed ? 'Episode completed' : 'Episode ended early'} · ${m.duration_s.toFixed(2)} s · ${m.solver_failures} solver failures · ${m.collision ? 'collision recorded' : 'no collision recorded'} · maximum obstacle slack ${Math.max(0,m.max_obstacle_slack).toFixed(4)} / 0.08`;
    drawPlots(); render();
  }
  function setScenario() {
    scenario = data.scenarios.find(s => s.id === $('scenario').value) || data.scenarios[0];
    const previous = run.id;
    $('method').innerHTML = scenario.runs.map(r => `<option value="${r.id}">${r.name}</option>`).join('');
    $('method').value = previous; selectRun();
  }
  function render(dt = 1/60) {
    scene.render(scenario, run, time, {ghost: $('ghost').checked, dt});
    const state = scene.lastState, index = Math.min(run.references.length-1, Math.floor(time/data.dt));
    $('speed-value').textContent = `${state[3].toFixed(1)} m/s`;
    $('reference-value').textContent = `${run.references[index].speed.toFixed(1)} m/s`;
    $('error-value').textContent = scene.lastError === null ? '—' : `${scene.lastError.toFixed(2)} m`;
    $('age-value').textContent = scene.lastForecast ? `${scene.lastForecast.elapsed.toFixed(1)} / 3.0 s` : 'Unavailable';
    $('time').textContent = `${time.toFixed(1)} / ${run.time.at(-1).toFixed(1)} s`;
    $('seek').value = time; $('seek').setAttribute('aria-valuetext', `${time.toFixed(1)} seconds`);
    for (const cursor of document.querySelectorAll('.plot-cursor')) cursor.setAttribute('x1', 44+Math.min(time,16)/16*322), cursor.setAttribute('x2', 44+Math.min(time,16)/16*322);
  }
  function plot(id, lines, low, high, unit) {
    const x = t => 44 + t/16*322, y = v => 153-(v-low)/(high-low)*115;
    let markup = `<svg viewBox="0 0 390 200" role="img" aria-label="${unit} over elapsed time">`;
    for(let i=0;i<4;i++) {
      const v = low+(high-low)*i/3, yy=y(v);
      markup += `<line x1="44" x2="366" y1="${yy}" y2="${yy}" stroke="#e1e7eb"/><text x="37" y="${yy+5}" text-anchor="end" fill="#526170" font-size="14">${v.toFixed(0)}</text>`;
    }
    for (const t of [0,4,8,12,16]) markup += `<text x="${x(t)}" y="178" text-anchor="middle" fill="#526170" font-size="14">${t}</text>`;
    markup += `<text x="44" y="19" fill="#526170" font-size="14">${unit}</text><text x="366" y="198" text-anchor="end" fill="#526170" font-size="14">Time (s)</text>`;
    for(const line of lines) {
      const points=line.values.map(([t,v])=>`${x(t)},${y(v)}`).join(' ');
      markup += `<polyline points="${points}" fill="none" stroke="${line.color}" stroke-width="2.4" ${line.dash?'stroke-dasharray="5 4"':''}/>`;
    }
    markup += '<line class="plot-cursor" x1="44" x2="44" y1="30" y2="153" stroke="#263747" stroke-width="1" opacity=".45"/></svg>';
    $(id).innerHTML=markup;
  }
  function drawPlots() {
    const actual = column => run.time.map((t,i)=>[t,run.states[i][column]]);
    const ref = key => run.time.map((t,i)=>[t,run.references[Math.min(i,run.references.length-1)][key]]);
    plot('speed-plot',[{values:actual(3),color:'#2166ac'},{values:ref('speed'),color:'#b44819',dash:true}],0,36,'Speed (m/s)');
    plot('lateral-plot',[{values:actual(1),color:'#2166ac'},{values:ref('lateral'),color:'#b44819',dash:true}],-6,6,'Lateral position (m)');
    const errors = run.time.map((t,i)=>{const f=window.fixedForecast(run,t,data.horizon,data.dt);return [t,f?Math.hypot(run.states[i][0]-f.state[0],run.states[i][1]-f.state[1]):0]});
    plot('error-plot',[{values:errors,color:'#b44819'}],0,Math.max(2,...errors.map(p=>p[1]))*1.1,'Frenet position error (m)');
  }
  $('scenario').addEventListener('change',setScenario);
  $('method').addEventListener('change',selectRun);
  $('view').addEventListener('change',()=>{scene.view=$('view').value;scene.focus=null;render()});
  $('ghost').addEventListener('change',()=>render());
  $('play').addEventListener('click',()=>{
    if(time>=run.time.at(-1)) {time=0;scene.focus=null;}
    playing=!playing;lastFrame=null;$('play').textContent=playing?'Pause':'Play';render();
  });
  $('reset').addEventListener('click',()=>{time=0;playing=false;scene.focus=null;$('play').textContent='Play';render()});
  $('seek').addEventListener('input',()=>{time=Number($('seek').value);scene.focus=null;render()});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){playing=false;lastFrame=null;$('play').textContent='Play'}});
  function frame(now) {
    const dt = lastFrame === null ? 0 : Math.min((now-lastFrame)/1000,.05);lastFrame=now;
    const wasPlaying = playing;
    if(playing){time=Math.min(run.time.at(-1),time+dt*Number($('rate').value));if(time>=run.time.at(-1)){playing=false;$('play').textContent='Play'}}
    if(wasPlaying||scene.dirty||scene.moving)render(Math.max(dt,1/120));
    requestAnimationFrame(frame);
  }
  window.replayStatus = () => ({scenario:scenario.id,method:run.id,time,playing,view:scene.view,duration:run.time.at(-1)});
  setScenario();requestAnimationFrame(frame);
})();
