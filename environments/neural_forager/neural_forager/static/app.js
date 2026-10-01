const $ = id => document.getElementById(id);
let ws, state, running = false, busy = false, fog = false, reconnectTimer;
const directions = ['north', 'east', 'south', 'west'];
const colors = {orange:'#dd8250', lime:'#d8ecaa', ink:'#345e48', muted:'#9ca99b'};
$('actionBars').innerHTML = directions.map(d => `<div class="action-row" id="bar-${d}"><label>${d[0].toUpperCase()+d.slice(1)}</label><div class="bar-track"><div class="bar-fill"></div></div><output>0.00</output></div>`).join('');

function config() { return {seed:Number($('seed').value), agent:$('agent').value, steps:Number($('steps').value), changes:$('changes').checked}; }
function message(type, extra={}) {
  if (busy || ws?.readyState !== WebSocket.OPEN) return;
  busy = true; controls(); ws.send(JSON.stringify({type,...extra}));
}
function connect() {
  ws = new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`);
  ws.onopen = () => { busy=false; $('connection').textContent='Nengo connected'; $('dot').classList.add('ready'); message('reset',{config:config()}); };
  ws.onmessage = event => {
    busy=false;
    const packet=JSON.parse(event.data);
    if(packet.type==='error') { running=false; $('error').textContent=packet.message; $('error').hidden=false; }
    else if(packet.type==='export') {
      const url=URL.createObjectURL(new Blob([JSON.stringify(packet.data,null,2)],{type:'application/json'}));
      const a=document.createElement('a'); a.href=url; a.download=`neural-forager-${packet.data.config.agent}-${packet.data.config.seed}.json`; a.click(); URL.revokeObjectURL(url);
    } else { state=packet.data; $('error').hidden=true; if(state.done) running=false; render(); }
    controls();
    if(running) setTimeout(()=>{if(running&&!busy) message('step');},25);
  };
  ws.onclose = () => {running=false;busy=false;state=null;controls();$('connection').textContent='Disconnected · reconnecting';$('dot').classList.remove('ready');clearTimeout(reconnectTimer);reconnectTimer=setTimeout(connect,2000);};
  ws.onerror = () => { $('connection').textContent='Connection interrupted'; };
}
function controls(){
  const ready=ws?.readyState===WebSocket.OPEN;
  ['step','route','relocate','export'].forEach(id=>$(id).disabled=!ready||!state||busy||running||(state?.done&&id!=='export'));
  $('play').disabled=!ready||!state||state.done||(!running&&busy);
  $('reset').disabled=!ready||busy;
  $('play').textContent=running?'Ⅱ Pause':state?.done?'Run complete':'▶ Run experiment';
}
$('play').onclick=()=>{running=!running;controls();if(running&&!busy)message('step');};
$('step').onclick=()=>message('step');
$('reset').onclick=()=>{running=false;message('reset',{config:config()});};
$('route').onclick=()=>message('route'); $('relocate').onclick=()=>message('food');
$('export').onclick=()=>message('export');
$('view').onclick=()=>{fog=!fog;$('view').textContent=fog?'Observed cells ◉':'Observer view ⊙';drawMaze();};
function canvas(id) {
  const node=$(id), rect=node.getBoundingClientRect(), ratio=window.devicePixelRatio||1;
  node.width=Math.round(rect.width*ratio);node.height=Math.round(rect.height*ratio);
  const ctx=node.getContext('2d');ctx.scale(ratio,ratio);return [ctx,rect.width,rect.height];
}
function circle(ctx,x,y,r,color){ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
function drawMaze(){
  if(!state)return;const [ctx,w,h]=canvas('maze'),world=state.world,n=world.size;
  const cell=Math.min(w/n,h/n),ox=(w-cell*n)/2,oy=(h-cell*n)/2;
  const visited=new Set(state.brain.seen.map(p=>p.join(','))),known=new Set();
  state.brain.seen.concat([world.position]).forEach(([x,y])=>[[x,y],[x-1,y],[x+1,y],[x,y-1],[x,y+1]].forEach(p=>known.add(p.join(','))));
  for(let y=0;y<n;y++)for(let x=0;x<n;x++){
    const seen=known.has(`${x},${y}`),wall=world.walls[y][x];
    ctx.fillStyle=fog&&!seen?'#1a322b':wall?'#20352e':visited.has(`${x},${y}`)?'#75906b':'#506b55';
    ctx.fillRect(ox+x*cell+1.5,oy+y*cell+1.5,cell-3,cell-3);
    if(visited.has(`${x},${y}`)&&!wall)circle(ctx,ox+(x+.5)*cell,oy+(y+.5)*cell,1.6,'#789578');
  }
  world.blocked.forEach(([a,b])=>{
    if(fog&&!visited.has(a.join(','))&&!visited.has(b.join(',')))return;
    const mx=ox+(a[0]+b[0]+1)*cell/2,my=oy+(a[1]+b[1]+1)*cell/2;
    ctx.strokeStyle=colors.orange;ctx.lineWidth=3;ctx.beginPath();
    if(a[0]===b[0]){ctx.moveTo(mx-cell*.38,my);ctx.lineTo(mx+cell*.38,my);}else{ctx.moveTo(mx,my-cell*.38);ctx.lineTo(mx,my+cell*.38);}ctx.stroke();
  });
  const [hx,hy]=world.home;ctx.strokeStyle='#87a18c';ctx.lineWidth=1.5;ctx.strokeRect(ox+(hx+.32)*cell,oy+(hy+.32)*cell,cell*.36,cell*.36);
  const [fx,fy]=world.food;
  if(!fog||fx===world.position[0]&&fy===world.position[1]){
    const x=ox+(fx+.5)*cell,y=oy+(fy+.5)*cell;circle(ctx,x,y,cell*.23,'#dd825025');circle(ctx,x,y,cell*.11,colors.orange);
    ctx.strokeStyle=colors.orange;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,y-cell*.21);ctx.lineTo(x,y-cell*.31);ctx.stroke();
  }
  const [px,py]=world.position,ax=ox+(px+.5)*cell,ay=oy+(py+.5)*cell;
  circle(ctx,ax,ay,cell*.29,'#d8ecaa18');circle(ctx,ax,ay,cell*.19,colors.lime);
  const d=directions.indexOf(state.action),offset=[[0,-1],[1,0],[0,1],[-1,0]][d]||[0,-1];circle(ctx,ax+offset[0]*cell*.09,ay+offset[1]*cell*.09,2,'#345e48');
}
function drawActivity(){
  const [ctx,w,h]=canvas('activity');
  const data=state.brain.activity;
  if(!data?.length){ctx.fillStyle='#758177';ctx.font='11px sans-serif';ctx.fillText(state.brain.neurons?'Run or step to measure neural activity.':'No spiking population in this controller.',0,35);return;}
  const n=data.length,cell=Math.min(w/n,h/n),ox=(w-cell*n)/2;
  for(let y=0;y<n;y++)for(let x=0;x<n;x++){
    const v=Math.min(1,data[y][x]/100);ctx.fillStyle=`rgb(${Math.round(231-v*145)},${Math.round(235-v*102)},${Math.round(223-v*132)})`;
    ctx.fillRect(ox+x*cell+1,y*cell+1,cell-2,cell-2);
  }
}
function drawTrials(){
  const [ctx,w,h]=canvas('trials'),left=38,right=w-30,top=22,bottom=h-30,items=state.trials,limit=state.config.trial_limit;
  ctx.font='9px monospace';ctx.textAlign='right';
  [0,limit/2,limit].forEach(v=>{const y=bottom-(bottom-top)*v/limit;ctx.strokeStyle='#e4e7db';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillStyle='#859080';ctx.fillText(v,left-9,y+3);});
  if(!items.length){ctx.textAlign='center';ctx.font='12px Georgia';ctx.fillStyle='#8a9484';ctx.fillText('The first trial is still unfolding.',(left+right)/2,h/2);return;}
  const count=Math.max(10,items.length-1),x=i=>left+(right-left)*i/count,y=v=>bottom-(bottom-top)*v/limit;
  ctx.strokeStyle='#b6c7ae';ctx.lineWidth=1.5;ctx.beginPath();items.forEach((t,i)=>i?ctx.lineTo(x(i),y(t.length)):ctx.moveTo(x(i),y(t.length)));ctx.stroke();
  ctx.setLineDash([3,4]);ctx.strokeStyle='#9aaa8f';ctx.beginPath();items.forEach((t,i)=>{if(t.mixed)return;i?ctx.lineTo(x(i),y(t.optimal)):ctx.moveTo(x(i),y(t.optimal));});ctx.stroke();ctx.setLineDash([]);
  items.forEach((t,i)=>{circle(ctx,x(i),y(t.length),3.5,t.success?colors.orange:'#a5afa7');});
  ctx.fillStyle='#859080';ctx.textAlign='left';ctx.fillText('TRIAL '+Math.max(1,state.summary.trials-items.length+1),left,h-9);ctx.textAlign='right';ctx.fillText(state.summary.trials,right,h-9);
}
function render(){
  const {world,brain,summary,config:c}=state;
  $('foodCount').textContent=world.collected;$('coverage').innerHTML=`${Math.round(summary.coverage*100)}<em>%</em>`;
  $('success').textContent=summary.trials?Math.round(summary.success_rate*100)+'%':'—';$('trialCount').textContent=`${summary.trials} completed trials`;
  $('neurons').textContent=brain.neurons?brain.neurons.toLocaleString():'—';$('engineLabel').textContent=brain.neurons?'LIF spiking neurons':'non-neural baseline';
  $('phase').textContent={learn:'DISCOVERY',reroute:'ROUTE CHANGED',relocate:'FOOD RELOCATED'}[world.phase];
  $('stepLabel').textContent=`STEP ${String(world.step).padStart(4,'0')} / ${c.steps}`;
  $('action').textContent=state.action||'waiting';$('td').textContent=brain.td_error.toFixed(3);$('weights').textContent=brain.weight_change.toFixed(5);
  directions.forEach((d,i)=>{const row=$('bar-'+d);row.classList.toggle('chosen',state.action===d);row.querySelector('.bar-fill').style.width=`${Math.max(0,Math.min(100,(brain.utilities[i]+1.5)/3*100))}%`;row.querySelector('output').textContent=brain.utilities[i].toFixed(2);});
  $('schedule').textContent=c.changes?`Passage change at step ${Math.floor(c.steps/3).toLocaleString()}. Food relocation at step ${Math.floor(c.steps*2/3).toLocaleString()}.`:'Scheduled changes off. You can still intervene manually.';
  const events=[{step:0,message:c.agent==='frozen'?'Frozen weights. Learning disabled for this control.':`Fresh ${c.agent} controller. Local observations only.`},...world.events];
  if(state.done)events.push({step:world.step,message:'Run complete. Export results or start a new seed.'});
  $('events').replaceChildren(...events.slice(-4).map(e=>{const div=document.createElement('div');div.className='log-item';const tag=document.createElement('span');tag.textContent=String(e.step).padStart(4,'0');const p=document.createElement('p');p.textContent=e.message;div.append(tag,p);return div;}));
  drawMaze();drawActivity();drawTrials();
}
window.addEventListener('resize',()=>{if(state)render();});connect();
