(() => {
  'use strict';
  const data = window.EXPERIMENT, $ = id => document.getElementById(id);
  const scene = new window.DrivingScene($('scene'), data);
  const names = {overtaking: 'Overtaking · pass a staggered pack', blocking: 'Blocking · defend against faster challengers', mixed: 'Mixed traffic · pass ahead, defend behind'};
  const modes = {follower_braking: 'Follower: brake-to-stop feasibility assumption', constant_velocity_heading: 'Leader / alongside: constant-speed, measured-heading estimate'};
  const colors = ['#72877c','#7a8193','#95867e'];
  let scenario = data.defense_training ? data.scenarios.find(s=>s.id==='blocking') : data.scenarios[0], run = scenario.runs[data.kind === 'hilepp' ? 2 : 0];
  let time = 0, playing = false, lastFrame = null;
  $('scenario').innerHTML = data.scenarios.map(s=>`<option value="${s.id}">${s.id[0].toUpperCase()+s.id.slice(1)}</option>`).join('');
  $('scenario').value=scenario.id;
  function selectRun() {
    run = scenario.runs.find(r=>r.id===$('method').value)||scenario.runs[0];
    time=0;playing=false;lastFrame=null;scene.focus=null;
    $('play').textContent='Play';$('seek').max=run.time.at(-1);$('seek').value=0;
    $('scene-title').textContent=names[scenario.id];
    $('scene-subtitle').textContent=`${run.name} · seed ${run.seed} · MPC replans every ${data.dt.toFixed(2)} s`;
    const m=run.metrics;
    $('run-status').textContent=`${m.completed?'Episode completed':'Episode ended early'} · ${m.duration_s.toFixed(2)} s · ${m.solver_failures} solver failures · ${m.collision?'collision recorded':'no collision recorded'} · ${m.ego_passes} ego passes · ${m.opponent_passes} opponent passes · ${m.defensive_moves} measured defensive moves`;
    $('interaction-summary').textContent=`Opponents held below their free-running target: ${m.held_up_vehicle_seconds.toFixed(1)} vehicle-seconds. Ego under attack: ${m.under_attack_seconds.toFixed(1)} s. Target speeds: ${run.opponent_target_speeds.map(v=>v.toFixed(1)).join(', ')} m/s; ego limit ${run.ego_speed_limit.toFixed(0)} m/s.`;
    renderEvents();drawPlots();render();
  }
  function setScenario() {
    scenario=data.scenarios.find(s=>s.id===$('scenario').value)||data.scenarios[0];
    const previous=run.id;
    $('method').innerHTML=scenario.runs.map(r=>`<option value="${r.id}">${r.name}</option>`).join('');
    $('method').value=scenario.id==='blocking'&&previous==='learned'&&data.defense_training?(data.defense_reference_limit?'defense_limited':'defense'):previous.startsWith('defense')?'learned':previous;selectRun();
  }
  function render(dt=1/60) {
    const opponent=Number($('opponent').value),overlay=$('overlay').value,probe=Number($('probe').value);
    scene.render(scenario,run,time,{overlay,opponent,probe,envelopes:$('envelopes').checked,dt});
    const state=scene.lastState,index=scene.lastPlan.index,plan=scene.lastPlan.plan;
    $('speed-value').textContent=`${state[3].toFixed(1)} m/s`;
    $('reference-value').textContent=`${run.references[index].speed.toFixed(1)} m/s`;
    const rank=1+[0,1,2].filter(j=>window.sampleRun(run,time,j)[0]>state[0]).length;
    $('rank-value').textContent=`${rank} / 4`;
    $('solve-value').textContent=`t = ${plan.origin.toFixed(2)} s`;
    $('time').textContent=`${time.toFixed(1)} / ${run.time.at(-1).toFixed(1)} s`;
    $('seek').value=time;$('seek').setAttribute('aria-valuetext',`${time.toFixed(1)} seconds`);
    const explanation=overlay==='audit'&&scene.lastAudit
      ? `${modes[scene.lastAudit.mode]}. Held at t=${scene.lastAudit.origin.toFixed(2)} s; age ${scene.lastAudit.elapsed.toFixed(2)} s; separation from the actual opponent at the same time: ${scene.lastError.toFixed(2)} m. MPC continues to replan.`
      : `${modes[plan.opponent_modes[opponent]]}. The blue ego plan and orange obstacle trajectories come from the same solve at t=${plan.origin.toFixed(2)} s. Ghosts mark t=${(plan.origin+probe).toFixed(2)} s.`;
    $('overlay-explanation').textContent=!plan.success&&overlay==='mpc' ? `MPC rejected this solve at t=${plan.origin.toFixed(2)} s (${plan.status}). No control from it was applied; the episode stops here.` : explanation;
    $('probe').disabled=overlay==='audit';
    for(const cursor of document.querySelectorAll('.plot-cursor')) {
      const x=44+Math.min(time,run.episode_limit_s)/run.episode_limit_s*322;
      cursor.setAttribute('x1',x);cursor.setAttribute('x2',x);
    }
  }
  function plot(id,lines,low,high,unit) {
    const duration=run.episode_limit_s,x=t=>44+t/duration*322,y=v=>153-(v-low)/(high-low)*115;
    let markup=`<svg viewBox="0 0 390 200" role="img" aria-label="${unit} over elapsed time">`;
    for(let i=0;i<4;i++) {const v=low+(high-low)*i/3,yy=y(v);markup+=`<line x1="44" x2="366" y1="${yy}" y2="${yy}" stroke="#e1e7eb"/><text x="37" y="${yy+5}" text-anchor="end" fill="#526170" font-size="14">${v.toFixed(0)}</text>`;}
    if(low<0&&high>0)markup+=`<line x1="44" x2="366" y1="${y(0)}" y2="${y(0)}" stroke="#2166ac" stroke-dasharray="3 3"/>`;
    for(const t of [0,duration/4,duration/2,duration*3/4,duration])markup+=`<text x="${x(t)}" y="178" text-anchor="middle" fill="#526170" font-size="14">${t}</text>`;
    markup+=`<text x="44" y="19" fill="#526170" font-size="14">${unit}</text><text x="366" y="198" text-anchor="end" fill="#526170" font-size="14">Time (s)</text>`;
    for(const line of lines)markup+=`<polyline points="${line.values.map(([t,v])=>`${x(t)},${y(v)}`).join(' ')}" fill="none" stroke="${line.color}" stroke-width="2.4" ${line.dash?'stroke-dasharray="5 4"':''}/>`;
    markup+='<line class="plot-cursor" x1="44" x2="44" y1="30" y2="153" stroke="#263747" opacity=".45"/></svg>';
    $(id).innerHTML=markup;
  }
  function drawPlots() {
    const actual=column=>run.time.map((t,i)=>[t,run.states[i][column]]);
    const ref=key=>run.time.map((t,i)=>[t,run.references[Math.min(i,run.references.length-1)][key]]);
    plot('speed-plot',[{values:actual(3),color:'#2166ac'},{values:ref('speed'),color:'#526170',dash:true}],0,40,'Ego speed (m/s)');
    const lateralLines=[{values:actual(1),color:'#2166ac'},{values:ref('lateral'),color:'#526170',dash:true}];
    if(run.raw_lateral_references)lateralLines.push({values:run.time.map((t,i)=>[t,run.raw_lateral_references[Math.min(i,run.raw_lateral_references.length-1)]]),color:'#76548a',dash:true});
    plot('lateral-plot',lateralLines,-6,6,'Lateral position (m)');
    const gaps=[0,1,2].map(j=>({values:run.time.map((t,i)=>[t,run.opponents[i][j][0]-run.states[i][0]]),color:colors[j]}));
    const values=gaps.flatMap(l=>l.values.map(v=>v[1]));
    plot('gap-plot',gaps,Math.min(-10,...values)-5,Math.max(10,...values)+5,'Opponent distance ahead (m)');
    const opponent=Number($('opponent').value);
    plot('opponent-speed-plot',[
      {values:run.time.map((t,i)=>[t,run.opponents[i][opponent][3]]),color:colors[opponent]},
      {values:run.time.map(t=>[t,run.opponent_target_speeds[opponent]]),color:'#526170',dash:true}
    ],15,40,`Opponent ${opponent+1} speed (m/s)`);
  }
  function renderEvents() {
    const labels={ego_pass:'Ego passes',opponent_pass:'Ego is passed by',defensive_move:'Ego covers the line of'};
    $('events').innerHTML=run.events.length?run.events.map((event,i)=>`<button data-event="${i}">${event.time.toFixed(1)} s · ${labels[event.kind]} opponent ${event.opponent+1}</button>`).join(''):'No completed pass or measured defensive move in this episode.';
  }
  $('events').addEventListener('click',event=>{const index=event.target.dataset?.event;if(index===undefined)return;time=Math.max(0,run.events[Number(index)].time-1);playing=false;scene.focus=null;$('play').textContent='Play';render()});
  $('scenario').addEventListener('change',setScenario);$('method').addEventListener('change',selectRun);
  $('view').addEventListener('change',()=>{scene.view=$('view').value;scene.focus=null;render()});
  for(const id of ['overlay','probe','envelopes'])$(id).addEventListener('change',()=>render());
  $('opponent').addEventListener('change',()=>{drawPlots();render()});
  $('play').addEventListener('click',()=>{if(time>=run.time.at(-1)){time=0;scene.focus=null;}playing=!playing;lastFrame=null;$('play').textContent=playing?'Pause':'Play';render()});
  $('reset').addEventListener('click',()=>{time=0;playing=false;scene.focus=null;$('play').textContent='Play';render()});
  $('seek').addEventListener('input',()=>{time=Number($('seek').value);scene.focus=null;render()});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){playing=false;lastFrame=null;$('play').textContent='Play'}});
  function frame(now) {
    const dt=lastFrame===null?0:Math.min((now-lastFrame)/1000,.05);lastFrame=now;const wasPlaying=playing;
    if(playing){time=Math.min(run.time.at(-1),time+dt*Number($('rate').value));if(time>=run.time.at(-1)){playing=false;$('play').textContent='Play'}}
    if(wasPlaying||scene.dirty||scene.moving)render(Math.max(dt,1/120));requestAnimationFrame(frame);
  }
  window.replayStatus=()=>({scenario:scenario.id,method:run.id,time,playing,view:scene.view,duration:run.time.at(-1)});
  setScenario();requestAnimationFrame(frame);
})();
