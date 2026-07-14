"use strict";
// ---------- config ----------
const CATS = ["sink","short_highres","mid","long_lowres","noisy"];
const CATLABEL = {sink:"sink (x0)", short_highres:"recent hi-res", mid:"mid", long_lowres:"far low-res", noisy:"noisy (self)"};
const COL = {sink:"#F5C542", short_highres:"#3B82F6", mid:"#22C7C7", long_lowres:"#9AA0A6", noisy:"#F97316"};
const SEC_PER_CHUNK = 33/24;
const V4METRICS = [
  ["history_share","history share (Σ history)"],
  ["sink_share","sink share"],
  ["short_highres_share","recent hi-res share"],
  ["mid_share","mid share"],
  ["long_lowres_share","far low-res share"],
  ["noisy_share","noisy(self) share"],
  ["entropy","attention entropy (nats)"],
];
const SVGNS="http://www.w3.org/2000/svg";

// ---------- state ----------
const S = { manifest:[], ds:null, prompt:null, metric:"mass", boost:1, step:"avg", v4metric:"history_share" };
const $ = s => document.querySelector(s);
const tip = $("#tip");

const HIST = ["sink","short_highres","mid","long_lowres"];
// ---------- data helpers ----------
// Counterfactual "history amplification": multiply the softmax WEIGHT of every
// history token by f (equivalent to adding log f to every history logit) and
// renormalize. f is exact from the natural per-category mass because all history
// tokens share the same boost, so category sums scale by f. f=1 => natural.
function boostedMass(rec){
  const m = rec.mass;
  if(S.boost===1) return m;
  const hist = HIST.reduce((s,c)=>s+(m[c]||0),0);
  const denom = S.boost*hist + (m.noisy||0);
  const out={};
  HIST.forEach(c=>out[c]=(S.boost*(m[c]||0))/denom);
  out.noisy=(m.noisy||0)/denom;
  return out;
}
function getDist(rec){
  const bm = boostedMass(rec);
  if(S.metric==="mass") return bm;
  // per-token = mass / n_tokens
  const out={}; CATS.forEach(c=>out[c]= (rec.n_tokens[c]>0? bm[c]/rec.n_tokens[c] : 0));
  return out;
}
function recs(filter){
  const out=[];
  for(const r of S.ds.by_step_layer){
    if(r.prompt!==S.prompt) continue;
    if(filter){
      if(filter.chunk!==undefined && r.chunk!==filter.chunk) continue;
      if(filter.layer!==undefined && r.layer!==filter.layer) continue;
      if(filter.step!==undefined && filter.step!=="avg" && r.denoise_step!==filter.step) continue;
    }
    if(S.step!=="avg" && (!filter || filter.step===undefined) && r.denoise_step!==S.step) continue;
    out.push(r);
  }
  return out;
}
function aggDist(list){
  const acc={}; CATS.forEach(c=>acc[c]=0);
  if(!list.length) return acc;
  for(const r of list){ const d=getDist(r); CATS.forEach(c=>acc[c]+=d[c]||0); }
  CATS.forEach(c=>acc[c]/=list.length);
  return acc;
}
function aggPerHead(list){ // natural mass per head, averaged
  const H = (S.ds.meta.num_heads)||40;
  const acc={}; CATS.forEach(c=>acc[c]=new Array(H).fill(0));
  let n=0;
  for(const r of list){ if(!r.per_head_mass) continue; n++; CATS.forEach(c=>{const a=r.per_head_mass[c]||[]; for(let h=0;h<H;h++)acc[c][h]+=(a[h]||0);}); }
  if(n) CATS.forEach(c=>{for(let h=0;h<H;h++)acc[c][h]/=n;});
  return acc;
}
function scalarOf(rec, name){
  if(name==="entropy") return rec.entropy;
  const d=getDist(rec); const total=CATS.reduce((s,c)=>s+(d[c]||0),0)||1e-9;
  const hist=d.sink+d.short_highres+d.mid+d.long_lowres;
  switch(name){
    case "history_share": return hist/total;
    case "sink_share": return d.sink/total;
    case "short_highres_share": return d.short_highres/total;
    case "mid_share": return d.mid/total;
    case "long_lowres_share": return d.long_lowres/total;
    case "noisy_share": return d.noisy/total;
  }
  return 0;
}

// ---------- svg utils ----------
function el(tag,attrs,parent){ const e=document.createElementNS(SVGNS,tag); for(const k in attrs) e.setAttribute(k,attrs[k]); if(parent)parent.appendChild(e); return e; }
function clear(svg){ while(svg.firstChild) svg.removeChild(svg.firstChild); }
function txt(svg,x,y,s,attrs){ const t=el("text",Object.assign({x,y,fill:"#c9ced6","font-size":11},attrs||{}),svg); t.textContent=s; return t; }
function showTip(html,e){ tip.innerHTML=html; tip.style.display="block"; tip.style.left=(e.clientX+12)+"px"; tip.style.top=(e.clientY+12)+"px"; }
function hideTip(){ tip.style.display="none"; }

function viridis(t){ // t in [0,1] -> rgb
  t=Math.max(0,Math.min(1,t));
  const stops=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  const x=t*(stops.length-1); const i=Math.floor(x); const f=x-i;
  const a=stops[i], b=stops[Math.min(i+1,stops.length-1)];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)})`;
}

// value bars (raw magnitudes)
function drawValueBars(svg, dist, subtitle){
  clear(svg); const W=svg.getAttribute("width")*1, Hh=svg.getAttribute("height")*1;
  const pad={l:44,r:14,t:22,b:56}; const iw=W-pad.l-pad.r, ih=Hh-pad.t-pad.b;
  const vals=CATS.map(c=>dist[c]||0); const mx=Math.max(...vals,1e-9);
  txt(svg,pad.l,14,subtitle||"",{fill:"#9aa0a6"});
  const bw=iw/CATS.length*0.62, gap=iw/CATS.length;
  el("line",{x1:pad.l,y1:pad.t+ih,x2:pad.l+iw,y2:pad.t+ih,stroke:"#2a303a"},svg);
  CATS.forEach((c,i)=>{
    const v=dist[c]||0, h=Math.max(1,v/mx*ih), x=pad.l+i*gap+(gap-bw)/2, y=pad.t+ih-h;
    const r=el("rect",{x,y,width:bw,height:h,fill:COL[c],rx:3},svg);
    r.addEventListener("mousemove",e=>showTip(`<b style="color:${COL[c]}">${CATLABEL[c]}</b><br>${S.metric==="mass"?"mass":"per-token"} = ${v.toExponential(3)}`,e));
    r.addEventListener("mouseleave",hideTip);
    txt(svg,x+bw/2,y-4,fmt(v),{ "text-anchor":"middle","font-size":10,fill:"#e6e9ef"});
    txt(svg,x+bw/2,pad.t+ih+14,CATLABEL[c].split(" ")[0],{ "text-anchor":"middle","font-size":10,fill:"#9aa0a6"});
    txt(svg,x+bw/2,pad.t+ih+26,CATLABEL[c].split(" ").slice(1).join(" "),{ "text-anchor":"middle","font-size":9,fill:"#6b7280"});
  });
}
function fmt(v){ if(v===0)return"0"; if(v>=0.01)return v.toFixed(3); return v.toExponential(1); }

// pie (share)
function drawPie(svg, dist){
  clear(svg); const W=svg.getAttribute("width")*1,Hh=svg.getAttribute("height")*1;
  const cx=W/2, cy=Hh/2, r=Math.min(W,Hh)/2-14;
  const total=CATS.reduce((s,c)=>s+(dist[c]||0),0)||1e-9;
  let a0=-Math.PI/2;
  CATS.forEach(c=>{
    const frac=(dist[c]||0)/total; const a1=a0+frac*2*Math.PI;
    const x0=cx+r*Math.cos(a0),y0=cy+r*Math.sin(a0),x1=cx+r*Math.cos(a1),y1=cy+r*Math.sin(a1);
    const large=(a1-a0)>Math.PI?1:0;
    if(frac>1e-4){
      const p=el("path",{d:`M${cx},${cy} L${x0},${y0} A${r},${r} 0 ${large} 1 ${x1},${y1} Z`,fill:COL[c],stroke:"#0e1116","stroke-width":1},svg);
      p.addEventListener("mousemove",e=>showTip(`<b style="color:${COL[c]}">${CATLABEL[c]}</b><br>share = ${(frac*100).toFixed(1)}%`,e));
      p.addEventListener("mouseleave",hideTip);
    }
    a0=a1;
  });
  txt(svg,cx,Hh-2,"share (normalized)",{ "text-anchor":"middle","font-size":10,fill:"#9aa0a6"});
}

// 100% stacked mini bar (vertical)
function drawStackMini(svg, dist, label, sub){
  clear(svg); const W=svg.getAttribute("width")*1,Hh=svg.getAttribute("height")*1;
  const pad={t:6,b:26}; const ih=Hh-pad.t-pad.b, x=8, bw=W-16;
  const total=CATS.reduce((s,c)=>s+(dist[c]||0),0)||1e-9;
  let y=pad.t;
  CATS.forEach(c=>{ const h=(dist[c]||0)/total*ih; if(h>0){ const rr=el("rect",{x,y,width:bw,height:h,fill:COL[c]},svg);
    rr.addEventListener("mousemove",e=>showTip(`<b style="color:${COL[c]}">${CATLABEL[c]}</b><br>${((dist[c]||0)/total*100).toFixed(1)}%`,e));
    rr.addEventListener("mouseleave",hideTip); y+=h; } });
  txt(svg,W/2,Hh-14,label,{ "text-anchor":"middle","font-size":11,fill:"#e6e9ef"});
  if(sub) txt(svg,W/2,Hh-3,sub,{ "text-anchor":"middle","font-size":9,fill:"#6b7280"});
}

// ---------- views ----------
function renderLegend(){
  const l=$("#legend"); l.innerHTML="";
  CATS.forEach(c=>{ const s=document.createElement("span"); s.innerHTML=`<span class="sw" style="background:${COL[c]}"></span>${CATLABEL[c]}`; l.appendChild(s); });
}

function chunkLabel(k){ return `chunk ${k} (~${(k*SEC_PER_CHUNK).toFixed(1)}s)`; }

function renderV1(){
  const chunk=+$("#slChunk").value, layer=+$("#slLayer").value;
  $("#lblChunk").textContent=chunkLabel(chunk);
  $("#lblLayer").textContent=`layer ${layer}/${maxLayer}`;
  const list=recs({chunk,layer,step:S.step});
  const dist=aggDist(list);
  drawValueBars($("#v1bar"),dist,`${S.metric==="mass"?"total mass":"per-token mean"} · ×${S.boost}`);
  drawPie($("#v1pie"),dist);
  const r0=list[0];
  $("#v1meta").innerHTML = r0? `history-share=<b>${(list.reduce((s,r)=>s+r.history_share,0)/list.length).toFixed(3)}</b> · entropy=<b>${(list.reduce((s,r)=>s+r.entropy,0)/list.length).toFixed(2)}</b> nats · n_tokens: `+CATS.map(c=>`${c.split("_")[0]} ${r0.n_tokens[c]}`).join(", ") : "no data";
  // per-head
  drawHeads($("#v1heads"), aggPerHead(list));
  // full map
  drawFullMap(chunk,layer);
}
function drawHeads(svg, ph){
  clear(svg); const W=svg.getAttribute("width")*1,Hh=svg.getAttribute("height")*1;
  const H=(S.ds.meta.num_heads)||40; const pad={l:34,r:10,t:10,b:24}; const iw=W-pad.l-pad.r, ih=Hh-pad.t-pad.b;
  const bw=iw/H*0.8, gap=iw/H;
  // normalize per head to its total (share) so specialization is visible
  for(let h=0;h<H;h++){
    const tot=CATS.reduce((s,c)=>s+(ph[c][h]||0),0)||1e-9;
    let y=pad.t;
    CATS.forEach(c=>{ const frac=(ph[c][h]||0)/tot; const hh=frac*ih; if(hh>0.2){ const x=pad.l+h*gap+(gap-bw)/2;
      const rr=el("rect",{x,y,width:bw,height:hh,fill:COL[c]},svg);
      rr.addEventListener("mousemove",e=>showTip(`head ${h}<br><b style="color:${COL[c]}">${CATLABEL[c]}</b> ${(frac*100).toFixed(1)}%`,e));
      rr.addEventListener("mouseleave",hideTip); y+=hh; } });
  }
  txt(svg,pad.l,Hh-8,"head 0",{ "font-size":10,fill:"#9aa0a6"});
  txt(svg,pad.l+iw-30,Hh-8,`head ${H-1}`,{ "font-size":10,fill:"#9aa0a6"});
  txt(svg,4,pad.t+8,"100%",{ "font-size":9,fill:"#6b7280"});
}
function drawFullMap(chunk,layer){
  const wrap=$("#v1mapwrap"), svg=$("#v1map"); clear(svg);
  const fm=(S.ds.full_maps||[]).find(m=>m.prompt===S.prompt&&m.chunk===chunk&&m.layer===layer);
  if(!fm){ $("#v1mapnote").textContent="(no full query×key map dumped for this cell — dumped only for a few representative cells)"; wrap.style.opacity=.6; return; }
  wrap.style.opacity=1;
  const M=fm.matrix, nq=M.length, nk=M[0].length;
  const W=svg.getAttribute("width")*1,Hh=svg.getAttribute("height")*1;
  const pad={l:8,r:8,t:16,b:34}; const iw=W-pad.l-pad.r, ih=Hh-pad.t-pad.b;
  const cw=iw/nk, ch=ih/nq; let mx=0; M.forEach(row=>row.forEach(v=>mx=Math.max(mx,v)));
  for(let i=0;i<nq;i++)for(let j=0;j<nk;j++){ const v=M[i][j]/(mx||1e-9);
    el("rect",{x:pad.l+j*cw,y:pad.t+i*ch,width:Math.ceil(cw),height:Math.ceil(ch),fill:viridis(Math.pow(v,0.5))},svg); }
  // category boundary overlays on key axis
  const bb=fm.kbin_boundaries;
  CATS.forEach(c=>{ const [s,e]=bb[c]; const x=pad.l+s*cw;
    el("line",{x1:x,y1:pad.t,x2:x,y2:pad.t+ih,stroke:COL[c],"stroke-width":1.5,"stroke-dasharray":"3 2",opacity:.9},svg);
    txt(svg,pad.l+(s+e)/2*cw,Hh-20,c.split("_")[0],{ "text-anchor":"middle","font-size":9,fill:COL[c]});
  });
  txt(svg,pad.l,12,"query (noisy) ↓   key →",{ "font-size":10,fill:"#9aa0a6"});
  $("#v1mapnote").textContent=`head-averaged, pooled ${nq}×${nk}; color=attention (sqrt scaled).`;
}

function renderV2(){
  const wrap=$("#v2sm"); wrap.innerHTML="";
  for(let k=0;k<=maxChunk;k++){
    const list=recs({chunk:k,step:S.step}); // all layers
    const dist=aggDist(list);
    const div=document.createElement("div"); div.className="sm";
    const svg=document.createElementNS(SVGNS,"svg"); svg.setAttribute("width",120); svg.setAttribute("height",180);
    div.appendChild(svg); wrap.appendChild(div);
    drawStackMini(svg,dist,`chunk ${k}`,`~${(k*SEC_PER_CHUNK).toFixed(1)}s · L-avg`);
  }
}

function renderV3(){
  const list=recs({}); // all chunks/layers/steps(respecting step filter)
  const dist=aggDist(list);
  drawValueBars($("#v3bar"),dist,`global mean · ${S.metric==="mass"?"total mass":"per-token"} · ×${S.boost}`);
  drawPie($("#v3pie"),dist);
  const total=CATS.reduce((s,c)=>s+dist[c],0)||1e-9;
  let h=`<table><tr><th class="cat">class</th><th>${S.metric==="mass"?"mass":"per-token"}</th><th>share</th></tr>`;
  CATS.forEach(c=>{ h+=`<tr><td class="cat"><span class="sw" style="background:${COL[c]}"></span> ${CATLABEL[c]}</td><td class="val">${dist[c].toExponential(2)}</td><td class="val">${(dist[c]/total*100).toFixed(1)}%</td></tr>`; });
  h+=`</table><div class="note">${list.length} records averaged</div>`;
  $("#v3tab").innerHTML=h;
}

function renderV4(){
  const svg=$("#v4map"); clear(svg);
  const W=svg.getAttribute("width")*1,Hh=svg.getAttribute("height")*1;
  const pad={l:40,r:16,t:16,b:40}; const iw=W-pad.l-pad.r, ih=Hh-pad.t-pad.b;
  const nC=maxChunk+1, nL=maxLayer+1;
  const cw=iw/nC, ch=ih/nL;
  // build matrix
  const grid=[]; let mn=Infinity,mx=-Infinity;
  for(let l=0;l<=maxLayer;l++){ grid[l]=[]; for(let k=0;k<=maxChunk;k++){
    const list=recs({chunk:k,layer:l,step:S.step}); let v=0;
    if(list.length){ v=list.reduce((s,r)=>s+scalarOf(r,S.v4metric),0)/list.length; }
    grid[l][k]=v; mn=Math.min(mn,v); mx=Math.max(mx,v);
  }}
  const rng=(mx-mn)||1e-9;
  for(let l=0;l<=maxLayer;l++)for(let k=0;k<=maxChunk;k++){
    const v=grid[l][k]; const x=pad.l+k*cw, y=pad.t+l*ch;
    const rr=el("rect",{x,y,width:Math.ceil(cw),height:Math.ceil(ch),fill:viridis((v-mn)/rng)},svg);
    rr.addEventListener("mousemove",e=>showTip(`${chunkLabel(k)} · layer ${l}<br><b>${S.v4metric}</b> = ${v.toFixed(4)}`,e));
    rr.addEventListener("mouseleave",hideTip);
  }
  // axes
  txt(svg,pad.l,12,`${S.v4metric}  (${S.metric}, ×${S.boost})`,{ "font-size":11,fill:"#e6e9ef"});
  for(let k=0;k<=maxChunk;k+=Math.ceil(nC/8)||1) txt(svg,pad.l+k*cw+cw/2,pad.t+ih+14,`${k}`,{ "text-anchor":"middle","font-size":9,fill:"#9aa0a6"});
  txt(svg,pad.l+iw/2,Hh-6,"chunk (time →)",{ "text-anchor":"middle","font-size":10,fill:"#9aa0a6"});
  for(let l=0;l<=maxLayer;l+=Math.ceil(nL/8)||1) txt(svg,pad.l-6,pad.t+l*ch+ch/2+3,`${l}`,{ "text-anchor":"end","font-size":9,fill:"#9aa0a6"});
  txt(svg,12,pad.t+ih/2,"layer",{ "text-anchor":"middle","font-size":10,fill:"#9aa0a6",transform:`rotate(-90 12 ${pad.t+ih/2})`});
  // colorbar
  for(let i=0;i<40;i++){ el("rect",{x:pad.l+iw-8,y:pad.t+i*(ih/40),width:8,height:ih/40+1,fill:viridis(1-i/40)},svg); }
  txt(svg,pad.l+iw-10,pad.t+8,mx.toFixed(2),{ "text-anchor":"end","font-size":9,fill:"#9aa0a6"});
  txt(svg,pad.l+iw-10,pad.t+ih,mn.toFixed(2),{ "text-anchor":"end","font-size":9,fill:"#9aa0a6"});

  // right: per-chunk layer-avg trend
  const bsvg=$("#v4bar"); clear(bsvg);
  const bW=bsvg.getAttribute("width")*1,bH=bsvg.getAttribute("height")*1;
  const bp={l:44,r:14,t:16,b:40}; const biw=bW-bp.l-bp.r,bih=bH-bp.t-bp.b;
  const trend=[]; for(let k=0;k<=maxChunk;k++){ let s=0,n=0; for(let l=0;l<=maxLayer;l++){s+=grid[l][k];n++;} trend.push(s/n); }
  const tmx=Math.max(...trend), tmn=Math.min(...trend,0), tr=(tmx-tmn)||1e-9;
  el("line",{x1:bp.l,y1:bp.t+bih,x2:bp.l+biw,y2:bp.t+bih,stroke:"#2a303a"},bsvg);
  const gap=biw/trend.length, bw=gap*0.6;
  trend.forEach((v,k)=>{ const h=(v-tmn)/tr*bih, x=bp.l+k*gap+(gap-bw)/2, y=bp.t+bih-h;
    const rr=el("rect",{x,y,width:bw,height:Math.max(1,h),fill:viridis((v-tmn)/tr),rx:2},bsvg);
    rr.addEventListener("mousemove",e=>showTip(`${chunkLabel(k)}<br>${S.v4metric}=${v.toFixed(4)}`,e)); rr.addEventListener("mouseleave",hideTip);
    if(k%(Math.ceil(trend.length/8)||1)===0) txt(bsvg,x+bw/2,bp.t+bih+14,`${k}`,{ "text-anchor":"middle","font-size":9,fill:"#9aa0a6"});
  });
  txt(bsvg,bp.l,12,`${S.v4metric} · layer-avg vs chunk`,{ "font-size":10,fill:"#e6e9ef"});
  txt(bsvg,bp.l+biw/2,bH-6,"chunk (time →)",{ "text-anchor":"middle","font-size":10,fill:"#9aa0a6"});
}

// ---------- orchestration ----------
let maxChunk=0, maxLayer=0;
function currentView(){ return document.querySelector(".tabs button.on").dataset.t; }
function renderAll(){
  const v=currentView();
  if(v==="v1") renderV1();
  else if(v==="v2") renderV2();
  else if(v==="v3") renderV3();
  else if(v==="v4") renderV4();
}
function computeBounds(){
  maxChunk=0; maxLayer=0;
  for(const r of S.ds.by_step_layer){ if(r.prompt!==S.prompt)continue; maxChunk=Math.max(maxChunk,r.chunk); maxLayer=Math.max(maxLayer,r.layer); }
  $("#slChunk").max=maxChunk; $("#slLayer").max=maxLayer;
  if(+$("#slChunk").value>maxChunk)$("#slChunk").value=maxChunk;
  if(+$("#slLayer").value>maxLayer)$("#slLayer").value=Math.min(20,maxLayer);
}
function populateSteps(){
  const sel=$("#selStep"); sel.innerHTML="";
  const steps=(S.ds.meta.recorded_steps||[]).slice();
  const o=document.createElement("option"); o.value="avg"; o.textContent="average"; sel.appendChild(o);
  steps.forEach(s=>{ const op=document.createElement("option"); op.value=s; op.textContent="step "+s; sel.appendChild(op); });
  sel.value="avg"; S.step="avg";
}
function populatePrompts(){
  const sel=$("#selPrompt"); sel.innerHTML="";
  const ps=[...new Set(S.ds.by_step_layer.map(r=>r.prompt))];
  ps.forEach(p=>{ const o=document.createElement("option"); o.value=p; o.textContent=p; sel.appendChild(o); });
  S.prompt=ps[0]; sel.value=ps[0];
}
function setMetaPill(){
  const m=S.ds.meta;
  $("#metaPill").textContent=`${m.checkpoint} · ${m.task} · ${m.resolution} · ${m.num_layers}L×${m.num_heads}H · ${m.num_inference_steps} steps`;
  $("#footnote").innerHTML=`<b>History amplify (×f)</b> is a <b>counterfactual</b>: the released weights ship <code>is_amplify_history=false</code>, so the built-in amplification mechanism is inactive and <b>×1.0 = natural</b> is the model's true behavior. Boosting multiplies the softmax weight of every history token by f (= adding log&nbsp;f to each history logit) and renormalizes, illustrating how the mechanism would pull attention toward history. — ${m.notes||""}`;
}
async function loadDataset(file){
  const r=await fetch("data/"+file); S.ds=await r.json();
  populatePrompts(); populateSteps(); setMetaPill(); computeBounds(); renderLegend(); renderAll();
}

function wire(){
  // segmented toggles
  document.querySelectorAll("#segMetric button").forEach(b=>b.onclick=()=>{ document.querySelectorAll("#segMetric button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); S.metric=b.dataset.v; renderAll(); });
  $("#slBoost").oninput=e=>{ S.boost=+e.target.value; $("#lblBoost").textContent = S.boost===1?"×1.0 (natural)":`×${S.boost.toFixed(2)} (counterfactual)`; renderAll(); };
  $("#selStep").onchange=e=>{ S.step=e.target.value==="avg"?"avg":+e.target.value; renderAll(); };
  $("#selPrompt").onchange=e=>{ S.prompt=e.target.value; computeBounds(); renderAll(); };
  $("#slChunk").oninput=renderV1; $("#slLayer").oninput=renderV1;
  // tabs
  document.querySelectorAll("#tabs button").forEach(b=>b.onclick=()=>{
    document.querySelectorAll("#tabs button").forEach(x=>x.classList.remove("on")); b.classList.add("on");
    document.querySelectorAll(".view").forEach(v=>v.classList.remove("on")); $("#"+b.dataset.t).classList.add("on"); renderAll();
  });
  // v4 metric
  const sm=$("#selMetricV4"); V4METRICS.forEach(([v,l])=>{ const o=document.createElement("option"); o.value=v; o.textContent=l; sm.appendChild(o); });
  sm.value=S.v4metric; sm.onchange=e=>{ S.v4metric=e.target.value; renderV4(); };
  // checkpoint
  $("#selCkpt").onchange=e=>{ const d=S.manifest.find(x=>x.name===e.target.value); loadDataset(d.file); };
}

async function boot(){
  wire();
  try{
    const r=await fetch("data/manifest.json"); S.manifest=await r.json();
  }catch(e){ S.manifest=[{name:"Helios-Base",file:"base_full.json"}]; }
  const sel=$("#selCkpt"); S.manifest.forEach(d=>{ const o=document.createElement("option"); o.value=d.name; o.textContent=d.name; sel.appendChild(o); });
  await loadDataset(S.manifest[0].file);
}
boot();
