// Model proxy plus declarative phased tasks. No legacy task runners.
import fs from 'node:fs';
import path from 'node:path';
import {createCore} from './core.mjs';
import {inprocFetch} from './inproc.mjs';
import {resolveProxyToken,expandHome,pworkerHome,serverRecordFile,readServerRecord} from './settings.mjs';
import {loadLayers,ensureUserHome} from './config.mjs';
import {createPworkerClient} from './client.mjs';
import {Pworker,validateTask} from './pworker/task.mjs';
import {jobStore} from './pworker/jobs.mjs';
import {httpFetch} from './http-fetch.mjs';
import {createLocalLlmHost,OWNER_WAIT_MS} from './local-llm-host/host.mjs';
import {createLocalLlmRoutes} from './local-llm-host/routes.mjs';

// One proxy per Pworker home (all limits live in it; two proxies would double the effective provider rate). `pworker serve` records
// {pid, host, port} in <home>/server.json; a second serve for the same home refuses while that server answers, and clients without an
// explicit URL or port use the recorded port.
export {serverRecordFile,readServerRecord};
const alive=pid=>{try{process.kill(pid,0);return true;}catch(e){return e.code==='EPERM';}};
/** The live proxy of this home ({pid, host, port}) or null: its process exists and its /health answers on the recorded port. */
export async function liveServer(home,{fetchImpl=httpFetch}={}){
  const r=readServerRecord(home);
  if(!r||!alive(r.pid))return null;
  const host=!r.host||r.host==='0.0.0.0'||r.host==='::'?'127.0.0.1':r.host;
  try{const res=await fetchImpl(`http://${host.includes(':')?`[${host}]`:host}:${r.port}/health`,{signal:AbortSignal.timeout(1500)});await res.text().catch(()=>{});return res.ok?r:null;}catch{return null;}
}

/**
 * The request tags of a task submission (the headers of POST /v1/tasks): purpose, run, priority, cache and client name. A task's model
 * calls carry them, so the submitter's budget, run accounting, priority class and purpose rules apply to its phases; without a purpose
 * header the calls keep the server's own purpose `pworker:tasks`.
 */
export function taskTags(headers={}) {
  const get=name=>{const v=headers[name];return typeof v==='string'&&v.trim()?v.trim().slice(0,200):null;};
  const tags={purpose:get('x-pworker-purpose'),run:get('x-pworker-run'),priority:get('x-pworker-priority'),cache:get('x-pworker-cache'),client:get('x-client-name')};
  if(tags.priority&&!['interactive','normal','background'].includes(tags.priority))tags.priority=null;
  if(tags.cache&&!['use','strict','record','off'].includes(tags.cache))tags.cache=null;
  const set=Object.fromEntries(Object.entries(tags).filter(([,v])=>v!=null));
  return Object.keys(set).length?set:null;
}

export function createOps({config,inproc,home}) {
  const store=jobStore(home);
  const waiters=new Map();
  const abort=new AbortController();let shuttingDown=false;
  const client=createPworkerClient({url:'http://pworker.local',fetchImpl:inproc,purpose:'pworker:tasks',autostart:false,env:{}});
  const worker=new Pworker({client,config,signal:abort.signal,onProgress:event=>{
    if(shuttingDown)return;
    try{store.update(event.id,{...event,pid:process.pid});}catch(error){console.error(`task ${event.id}: progress not saved: ${error.message}`);}
    for(const wake of waiters.get(event.id)??[])wake();
    waiters.delete(event.id);
  }});
  let scheduled=false;
  function schedule() {
    if(scheduled)return;
    scheduled=true;
    setImmediate(()=>{scheduled=false;worker.flush().catch(error=>console.error(`task flush failed: ${error?.message??error}`));});
  }
  function view(id) {
    const row=store.view(id);
    if(!row)return null;
    return {...row,kind:'task',status:row.status==='completed'?'finished':row.status,log:[],log_total:0};
  }
  /** `tags` {purpose, run, priority, cache, client}: the submitter's request tags, given to every model call of the task (and kept for a recovery). */
  function start(kind,args,tags=null) {
    if(kind!=='task')throw new Error('Only declarative phased tasks are supported');
    if(!args||Object.keys(args).some(k=>!['task','input','currentWorkingDirectory'].includes(k)))throw new Error('Only task, input and currentWorkingDirectory are accepted');
    validateTask(args.task);
    const task=structuredClone(args.task);
    const cwd=args.currentWorkingDirectory??args.input?.currentWorkingDirectory??null;
    const directory=cwd?fs.realpathSync(path.resolve(cwd)):null;
    if(directory&&!fs.statSync(directory).isDirectory())throw new Error('Working directory is not a directory');
    const row=store.create({request:task,input:args.input??{},currentWorkingDirectory:directory});
    store.update(row.id,{pid:process.pid,executionOwner:'server',...(tags?{tags}:{})});
    worker.enqueue(task,args.input??{},{id:row.id,currentWorkingDirectory:directory,tags});
    schedule();
    return view(row.id);
  }
  async function wait(id,seconds=0) {
    const row=view(id);
    if(!row)return null;
    if(['queued','running','waiting'].includes(row.status)&&seconds>0)await new Promise(resolve=>{
      const wake=()=>{clearTimeout(timer);waiters.get(id)?.delete(wake);resolve();};
      const timer=setTimeout(wake,Math.min(seconds,60)*1000);
      if(!waiters.has(id))waiters.set(id,new Set());
      waiters.get(id).add(wake);
    });
    return view(id);
  }
  function recover(){
    for(const row of store.list({raw:true})){
      if(!['waiting','queued'].includes(row.status))continue;
      if(row.pid){try{process.kill(row.pid,0);continue;}catch(e){if(e.code!=='ESRCH')continue;}}
      const checkpoint=row.checkpoint;
      if(!checkpoint&&!(row.status==='queued'&&row.executionOwner==='server'))continue;
      try{
        const task=checkpoint?.task??row.request;validateTask(task);
        worker.enqueue(task,row.input,{id:row.id,currentWorkingDirectory:row.currentWorkingDirectory,checkpoint,tags:row.tags??null});
        store.update(row.id,{pid:process.pid,executionOwner:'server',waitingReason:'Recovered pending phase',error:null});
      }catch(error){store.update(row.id,{status:'failed',error:`Checkpoint recovery refused: ${error.message}`});}
    }
    if(worker.pending.length)schedule();
  }
  function close(){
    if(shuttingDown)return;
    shuttingDown=true;abort.abort();
    for(const row of store.list({raw:true}))if(row.executionOwner==='server'&&row.pid===process.pid&&['waiting','queued'].includes(row.status))store.update(row.id,{pid:null,waitingReason:'Waiting for worker restart'});
  }
  const cancel=id=>{const row=store.read(id);if(!row)return null;worker.cancel(id);return view(id);};
  return {start,wait,get:view,list:()=>store.list().slice(0,50).map(r=>view(r.id)),store,recover,close,cancel};
}

/** What listens on `host:port`: 'pworker' when its /health has the shape of ours, else 'other' (another program, or nothing that answers). */
export async function identifyListener(host,port,{fetchImpl=httpFetch}={}){
  const target=!host||host==='0.0.0.0'||host==='::'?'127.0.0.1':host;
  try{
    const res=await fetchImpl(`http://${target.includes(':')?`[${target}]`:target}:${port}/health`,{signal:AbortSignal.timeout(1500)});
    const body=res.ok?await res.json().catch(()=>null):null;
    if(body&&body.ok===true&&typeof body.uptime_s==='number'&&body.tiers!==undefined&&body.upstreams!==undefined)return 'pworker';
    await res.body?.cancel?.().catch(()=>{});
  }catch{}
  return 'other';
}

/** `localLlmOptions` are seams of the local-llm host (lib/local-llm-host/host.mjs), for tests. */
export async function createPworkerServer({config,env=process.env,coreOptions={},localLlmOptions={}}={}) {
  const http=await import('node:http');
  const dataDir=expandHome(config.dataDir??path.join(pworkerHome(env),'data'));
  const home=config.taskHome??path.dirname(dataDir);
  let ops;
  const routes=async(req,res,url,{sendJson,readJson})=>{
    const p=url.pathname;
    // The engine's admin routes (/v1/local-llm/*); they answer before the task routes and before the request admission.
    const engine=await engineRoutes(req,res,url,{sendJson});
    if(engine!==false)return engine;
    if(req.method==='POST'&&/^\/v1\/tasks\/[^/]+\/cancel$/.test(p)){
      try{const row=ops.cancel(decodeURIComponent(p.split('/')[3]));return row?sendJson(res,200,row):sendJson(res,404,{error:{type:'not_found',message:'No such task'}});}
      catch(error){return sendJson(res,400,{error:{type:'invalid_task_id',message:error.message}});}
    }
    if(/^\/v1\/(?:lambdas|run|jobs|calls)(?:\/|$)/.test(p))return sendJson(res,410,{error:{type:'unsupported_task_format',message:'Legacy execution and call APIs were removed. Submit a declarative phase map to /v1/tasks.'}});
    if(req.method==='POST'&&p==='/v1/tasks') {
      try {
        const body=await readJson(req);
        if(!body||typeof body!=='object'||Array.isArray(body))throw new Error('A task request object is required');
        const op=ops.start('task',body,taskTags(req.headers));
        return sendJson(res,202,{id:op.id,kind:'task',status:op.status,phase:op.phase});
      } catch(error){return sendJson(res,400,{error:{type:'invalid_task',message:error.message}});}
    }
    if(req.method==='GET'&&p==='/v1/ops')return sendJson(res,200,{data:ops.list()});
    if(req.method==='GET'&&p.startsWith('/v1/ops/')) {
      try {
        const row=await ops.wait(decodeURIComponent(p.slice('/v1/ops/'.length)),Math.max(0,Math.min(Number(url.searchParams.get('wait'))||0,60)));
        return row?sendJson(res,200,row):sendJson(res,404,{error:{type:'not_found',message:'No such task'}});
      } catch(error){return sendJson(res,400,{error:{type:'invalid_task_id',message:error.message}});}
    }
    return false;
  };
  // One local-llm host per Pworker home; nothing is created or bound until serve() acquires the home or the engine is first used. The home
  // is the server's own (config.taskHome, else the parent of its data folder), so a server built for a scratch folder never reaches the
  // real ~/.pworker, whatever its `env` says; for `pworker serve` it is the Pworker home (data folder = <home>/data).
  const localLlm=createLocalLlmHost({config,home,env,log:line=>console.error(line),...localLlmOptions});
  const engineRoutes=createLocalLlmRoutes({host:localLlm,gate:()=>Object.values(core.gates)[0]??null});
  const core=createCore({config,env,dataDir,proxyToken:resolveProxyToken(config,env),localLlm,...coreOptions,routes});
  const inproc=inprocFetch(core.handle);
  ops=createOps({config,inproc,home});
  const server=http.createServer((req,res)=>{core.handle(req,res).catch(()=>{try{res.destroy();}catch{}});});
  server.requestTimeout=0;server.headersTimeout=30000;server.keepAliveTimeout=5000;
  return {core,ops,server,localLlm,handle:core.handle,fetch:inproc,
    address:()=>server.address(),
    listen:(port,host='127.0.0.1')=>new Promise((resolve,reject)=>{
      server.once('error',reject);server.listen(port,host,()=>{server.off('error',reject);core.boot();ops.recover();resolve(server.address());});
    }),
    close:async()=>{
      ops.close();
      server.close();server.closeAllConnections?.();
      // The engine is drained before close() returns, so the process that exits next leaves no runner and no ownership behind.
      await Promise.all([...Object.values(core.starters).map(s=>s.stop?.()),localLlm.drain()]);
    },
  };
}

/** `localLlmOptions` and `ownerWaitMs` are seams for tests (see createPworkerServer and the local-llm host). */
export async function serve({port=null,host=null,project=null,env=process.env,log=console.log,localLlmOptions={},ownerWaitMs=OWNER_WAIT_MS}={}) {
  const home=ensureUserHome(env);
  for(const file of home.written)log(`wrote ${file}`);
  const other=await liveServer(home.home);
  if(other){log(`a Ploinky Workers proxy for ${home.home} already runs (pid ${other.pid}, port ${other.port}); one proxy per home keeps the provider limits shared`);return {already:true,...other};}
  const {config,layers}=loadLayers({project,env});
  const server=await createPworkerServer({config,env,localLlmOptions:{log:line=>log(line),...localLlmOptions}});
  const p=Number(port??env.PWORKER_PORT??config.server?.port??18080),h=host??config.server?.host??'127.0.0.1';
  // Ownership of the home's engine, and the reaping of runners an earlier process left behind, come first: a predecessor that is
  // still draining is waited for (OWNER_WAIT_MS). Another pworker of this home means this one must not run; any other refusal
  // leaves the proxy up with the engine unavailable, and the refusal is what every engine use reports.
  try{await server.localLlm.acquire({waitMs:ownerWaitMs});}
  catch(error){
    if(error?.code==='owner_held'){await server.close();log(`a Ploinky Workers proxy for ${home.home} already runs the local models (${error.message})`);return {already:true};}
    log(`the local-llm engine is unavailable: ${error?.message??error}`);
  }
  try{await server.listen(p,h);}catch(error){
    await server.close();
    if(error.code==='EADDRINUSE'){
      if(await identifyListener(h,p)==='pworker'){log(`a Ploinky Workers proxy already listens on ${h}:${p}`);return {already:true};}
      log(`port ${p} on ${h} is taken by another program, not by a Ploinky Workers proxy; stop it or choose another port with --port`);
      return {conflict:true};
    }
    throw error;
  }
  const address=server.address?.()??null,actual=typeof address?.port==='number'?address.port:p;
  const recordFile=serverRecordFile(home.home);
  fs.writeFileSync(recordFile,JSON.stringify({pid:process.pid,host:h,port:actual,startedAt:new Date().toISOString()})+'\n',{mode:0o600});
  process.on('exit',()=>{try{if(readServerRecord(home.home)?.pid===process.pid)fs.unlinkSync(recordFile);}catch{}});
  log(`Ploinky Workers listening on http://${h}:${actual} (layers: ${layers.join(' < ')})`);
  for(const signal of ['SIGINT','SIGTERM'])process.on(signal,async()=>{await server.close();process.exit(0);});
  return server;
}
