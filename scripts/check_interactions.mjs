/** Recorded-data and application-state checks; not a browser layout test. */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const data=JSON.parse(fs.readFileSync(path.join(root,'docs/data/experiment.json'),'utf8'));
const context=new Proxy({measureText:t=>({width:String(t).length*8})},{get:(o,k)=>k in o?o[k]:(()=>{}),set:(o,k,v)=>(o[k]=v,true)});
class Element {
  constructor(id){this.id=id;this.value=id==='rate'?'1':'0';this.checked=id==='ghost';this.listeners={};this.attributes={};this.innerHTML='';this.textContent='';}
  addEventListener(event,callback){this.listeners[event]=callback;}
  fire(event){this.listeners[event]?.({target:this});}
  setAttribute(key,value){this.attributes[key]=value;}
  getBoundingClientRect(){return {width:1100,height:510};}
  getContext(){return context;}
}
const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,new Element(id));return nodes.get(id);};
let frame;const handlers={};
const document={getElementById:get,querySelectorAll:()=>[],hidden:false,addEventListener:(event,cb)=>{handlers[event]=cb}};
const window={EXPERIMENT:data,devicePixelRatio:1};
const sandbox={window,document,requestAnimationFrame:cb=>{frame=cb},ResizeObserver:class{observe(){}},Math};
for(const name of ['scene.js','app.js'])vm.runInNewContext(fs.readFileSync(path.join(root,'docs',name),'utf8'),sandbox);
let stamp=0;const tick=()=>frame(stamp+=1000/60);
tick();assert.equal(window.replayStatus().playing,false);assert.equal(window.replayStatus().time,0);
for(const scenario of data.scenarios){
  get('scenario').value=scenario.id;get('scenario').fire('change');
  assert.equal(window.replayStatus().scenario,scenario.id);
  for(const run of scenario.runs){get('method').value=run.id;get('method').fire('change');assert.equal(window.replayStatus().method,run.id);assert.ok(get('speed-plot').innerHTML.includes('<polyline'));}
}
get('seek').value='2.3';get('seek').fire('input');assert.equal(window.replayStatus().time,2.3);
get('view').value='overview';get('view').fire('change');assert.equal(window.replayStatus().view,'overview');
get('play').fire('click');tick();tick();assert.ok(window.replayStatus().time>2.3);
document.hidden=true;handlers.visibilitychange();const paused=window.replayStatus().time;tick();assert.equal(window.replayStatus().playing,false);assert.equal(window.replayStatus().time,paused);
document.hidden=false;get('play').fire('click');for(let i=0;i<1100;i++)tick();
assert.equal(window.replayStatus().time,window.replayStatus().duration);assert.equal(window.replayStatus().playing,false);
for(let i=0;i<10;i++)tick();assert.equal(window.replayStatus().time,window.replayStatus().duration);
get('reset').fire('click');assert.equal(window.replayStatus().time,0);
get('ghost').checked=false;get('ghost').fire('change');assert.equal(get('error-value').textContent,'—');
const scenario=data.scenarios[0],run=scenario.runs[0];
const scene=new window.DrivingScene(get('scene'),data);scene.render(scenario,run,1);
const old=scene.focus.slice();scene.render(scenario,run,1.01,{dt:.01});
assert.ok(Math.hypot(scene.focus[0]-old[0],scene.focus[1]-old[1])<1,'Camera continuity');
assert.equal(window.fixedForecast(run,2.9).origin,0);
assert.equal(window.fixedForecast(run,3).origin,0);
assert.equal(window.fixedForecast(run,3.01).origin,3);
const a=window.sampleRun(run,1),b=window.sampleRun(run,1.25),mid=window.sampleRun(run,1.125);
for(let i=0;i<5;i++)assert.ok(Math.abs(mid[i]-(a[i]+b[i])/2)<1e-10);
console.log('Playback checks passed: all tasks/controllers, charts, seek, camera continuity, fixed forecast, hidden-tab pause, and no automatic loop. Browser layout unverified.');
