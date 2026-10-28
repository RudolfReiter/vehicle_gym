/** Export the actual browser scene renderer as SVG, without a browser dependency.
 * This records its Canvas 2D drawing commands. It is an artifact exporter,
 * not a substitute for browser layout or interaction testing.
 */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
class SVGContext{
  constructor(w,h){this.width=w;this.height=h;this.parts=[];this.defs=[];this.stack=[];this.state={fillStyle:'#000',strokeStyle:'#000',lineWidth:1,globalAlpha:1,font:'10px monospace',textAlign:'left',lineCap:'butt',lineJoin:'miter',dash:[]};}
  clearRect(){this.parts=[];this.defs=[];}
  setTransform(){}
  save(){this.stack.push({...this.state});}
  restore(){this.state=this.stack.pop();}
  setLineDash(d){this.state.dash=d;}
  beginPath(){this.d='';}
  moveTo(x,y){this.d+=`M${x},${y} `;}
  lineTo(x,y){this.d+=`L${x},${y} `;}
  closePath(){this.d+='Z ';}
  ellipse(cx,cy,rx,ry,rotation){
    this.d+=`M${cx+rx},${cy} A${rx},${ry} ${rotation*180/Math.PI} 1 0 ${cx-rx},${cy} A${rx},${ry} ${rotation*180/Math.PI} 1 0 ${cx+rx},${cy} Z `;
  }
  arc(cx,cy,r){this.ellipse(cx,cy,r,r,0);}
  color(value){return typeof value==='object'?`url(#${value.id})`:value;}
  fill(){this.parts.push(`<path d="${this.d}" fill="${this.color(this.state.fillStyle)}" opacity="${this.state.globalAlpha}"/>`);}
  stroke(){const s=this.state;this.parts.push(`<path d="${this.d}" fill="none" stroke="${this.color(s.strokeStyle)}" stroke-width="${s.lineWidth}" stroke-linecap="${s.lineCap}" stroke-linejoin="${s.lineJoin}" stroke-dasharray="${s.dash.join(' ')}" opacity="${s.globalAlpha}"/>`);}
  fillRect(x,y,w,h){this.parts.push(`<rect x="${x}" y="${y}" width="${Math.max(0,w)}" height="${Math.max(0,h)}" fill="${this.color(this.state.fillStyle)}" opacity="${this.state.globalAlpha}"/>`);}
  strokeRect(x,y,w,h){const s=this.state;this.parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${this.color(s.strokeStyle)}" stroke-width="${s.lineWidth}" opacity="${s.globalAlpha}"/>`);}
  fillText(text,x,y){const s=this.state;this.parts.push(`<text x="${x}" y="${y}" fill="${this.color(s.fillStyle)}" font-family="sans-serif" font-size="${parseFloat(s.font)}" text-anchor="${s.textAlign==='center'?'middle':s.textAlign==='right'?'end':'start'}" opacity="${s.globalAlpha}">${esc(text)}</text>`);}
  measureText(t){return {width:String(t).length*parseFloat(this.state.font)*.61};}
  gradient(tag,attrs){const gradient={id:`g${this.defs.length}`,tag,attrs,stops:[],addColorStop(offset,color){this.stops.push({offset,color});}};this.defs.push(gradient);return gradient;}
  createLinearGradient(x1,y1,x2,y2){return this.gradient('linearGradient',`x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"`);}
  createRadialGradient(x1,y1,r1,x2,y2,r2){return this.gradient('radialGradient',`cx="${x2}" cy="${y2}" r="${r2}" fx="${x1}" fy="${y1}"`);}
  output(){const definitions=this.defs.map(g=>`<${g.tag} id="${g.id}" gradientUnits="userSpaceOnUse" ${g.attrs}>${g.stops.map(s=>`<stop offset="${s.offset}" stop-color="${s.color}"/>`).join('')}</${g.tag}>`).join('');return `<svg xmlns="http://www.w3.org/2000/svg" width="${this.width}" height="${this.height}" viewBox="0 0 ${this.width} ${this.height}"><defs>${definitions}</defs>${this.parts.join('')}</svg>`;}
}
for(const key of ['fillStyle','strokeStyle','lineWidth','globalAlpha','font','textAlign','lineCap','lineJoin'])Object.defineProperty(SVGContext.prototype,key,{get(){return this.state[key];},set(v){this.state[key]=v;}});
const data=JSON.parse(fs.readFileSync(path.join(root,'docs/data/experiment.json'),'utf8'));
const sandbox={window:{devicePixelRatio:1},ResizeObserver:class{observe(){}},Math};
vm.runInNewContext(fs.readFileSync(path.join(root,'docs/scene.js'),'utf8'),sandbox);

const defensive = data.defense_training;
const scenario=defensive?data.scenarios.find(s=>s.id==='blocking'):data.scenarios[0];
const run=defensive?scenario.runs.find(r=>r.id===(data.defense_reference_limit?'defense_limited':'defense')):scenario.runs[data.kind==='hilepp'?2:0];
const event=run.events?.find(e=>e.kind==='defensive_move');
const snapshotTime=defensive&&event?event.time:Math.min(4.8,run.time.at(-1));
const out=path.join(root,'artifacts/scenes');fs.mkdirSync(out,{recursive:true});
const context=new SVGContext(1200,620);
const canvas={getContext:()=>context,getBoundingClientRect:()=>({width:1200,height:620})};
const scene=new sandbox.window.DrivingScene(canvas,data);
scene.render(scenario,run,snapshotTime);
context.font='20px sans-serif';context.fillStyle='#202d3b';context.textAlign='left';
context.fillText(data.kind==='hilepp'?'HILEPP · Hierarchical racing policy':'vehicle_gym · Nonlinear predictive planning',30,38);
context.font='16px sans-serif';context.fillStyle='#526170';context.fillText(`${run.name} · seed ${run.seed} · t = ${snapshotTime.toFixed(2)} s`,30,67);
context.fillText('Blue: ego + current plan     Orange: obstacle assumptions     Solid cars: actual traffic',30,585);
fs.writeFileSync(path.join(out,'snapshot.svg'),context.output());
console.log('Exported actual scene renderer to artifacts/scenes/snapshot.svg');
