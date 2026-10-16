import './home.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import {createPworkerServer} from '../lib/server.mjs';
import {createPworkerClient} from '../lib/client.mjs';

async function fixture() {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'pworker-phase-server-'));
  const config={dataDir:path.join(root,'data'),taskHome:root,providers:{},tiers:{},policy:{allowedPurposes:['test:*','pworker:*'],untaggedDailyMax:100}};
  const server=await createPworkerServer({config,env:{}});
  const address=await server.listen(0);
  const url=`http://127.0.0.1:${address.port}`;
  return {root,config,server,url,client:createPworkerClient({url,purpose:'test:phase-server',autostart:false}),
    async close(){await server.close();fs.rmSync(root,{recursive:true,force:true});}};
}

test('HTTP executes only declared phases and persists IDs, working directory and results',async()=>{
  const t=await fixture();
  try {
    const task={begin:{tier:null,code:'this.upper=this.input.toUpperCase(); this.next("save")'},
      save:{tier:null,code:'await this.writeFile("output.txt",this.upper); this.end(await this.readFile("output.txt"))'}};
    const started=await t.client.task(task,{input:{input:'hello'},currentWorkingDirectory:t.root,wait:false});
    const done=await t.client.waitOp(started.id,{pollSeconds:1});
    assert.equal(done.status,'finished',done.error);assert.equal(done.result.value,'HELLO');
    assert.equal(done.steps,2);assert.equal(done.currentWorkingDirectory,fs.realpathSync(t.root));
    assert.equal(fs.readFileSync(path.join(t.root,'output.txt'),'utf8'),'HELLO');
    const persisted=JSON.parse(fs.readFileSync(path.join(t.root,'jobs',started.id+'.json'),'utf8'));
    assert.equal(persisted.result.value,'HELLO');assert.equal(persisted.status,'completed');
    assert.equal((await t.client.ops()).data[0].id,started.id);
  }finally{await t.close();}
});

test('all retired execution endpoints refuse requests and old task bodies fail validation',async()=>{
  const t=await fixture();
  try {
    for(const endpoint of ['/v1/lambdas','/v1/lambdas/demo','/v1/run','/v1/jobs','/v1/calls']) {
      for(const method of ['GET','POST']){
        const r=await fetch(t.url+endpoint,{method,...(method==='POST'?{headers:{'content-type':'application/json'},body:'{}'}:{})});
        assert.equal(r.status,410,endpoint);assert.equal((await r.json()).error.type,'unsupported_task_format');
      }
    }
    for(const body of [{instructions:'do it'},{task:{start:'begin',phases:{begin:{tier:null}}}},
      {task:{begin:{tier:null,code:'result => result'}}},{task:{begin:{tier:null}},attachments:[]}]) {
      const r=await fetch(t.url+'/v1/tasks',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
      assert.equal(r.status,400);assert.equal((await r.json()).error.type,'invalid_task');
    }
    assert.equal(t.server.ops.list().length,0);
    assert.throws(()=>t.server.ops.start('lambda',{}),/Only declarative/);
  }finally{await t.close();}
});

test('server can retrieve task results after restart',async()=>{
  const t=await fixture();
  let restarted;
  try {
    const done=await t.client.task({begin:{tier:null,code:'this.end(42)'}},{pollSeconds:1});
    await t.server.close();
    restarted=await createPworkerServer({config:t.config,env:{}});
    const addr=await restarted.listen(0);
    const client=createPworkerClient({url:`http://127.0.0.1:${addr.port}`,purpose:'test:restart',autostart:false});
    assert.equal((await client.op(done.id)).result.value,42);
  }finally{if(restarted)await restarted.close();await t.close();}
});

test('one proxy per home: a second serve for the same home refuses and clients find the recorded port',async()=>{
  const home=fs.mkdtempSync(path.join(os.tmpdir(),'pworker-one-proxy-')),env={PWORKER_HOME:home};
  const {serve,readServerRecord}=await import('../lib/server.mjs');
  const {createPworkerClient}=await import('../lib/client.mjs');
  const first=await serve({port:0,env,project:false,log:()=>{}});
  try{
    const record=readServerRecord(home);
    assert.equal(record.pid,process.pid);assert.ok(record.port>0);
    const second=await serve({port:0,env,project:false,log:()=>{}});
    assert.equal(second.already,true);assert.equal(second.port,record.port);
    const client=createPworkerClient({purpose:'test:one-proxy',autostart:false,env:{PWORKER_HOME:home}});
    assert.equal(client.url,`http://127.0.0.1:${record.port}`);
  }finally{await first.close();fs.rmSync(home,{recursive:true,force:true});}
});

test('a task submitted over HTTP sends its model calls with the submitter\'s purpose, run and priority; without them, pworker:tasks',async()=>{
  const http=await import('node:http');
  const seen=[];
  const up=http.createServer((req,res)=>{let b='';req.on('data',c=>b+=c);req.on('end',()=>{seen.push(req.headers);res.writeHead(200,{'content-type':'application/json'});res.end(JSON.stringify({choices:[{message:{content:'ok'},finish_reason:'stop'}],usage:{prompt_tokens:1,completion_tokens:1}}));});});
  await new Promise(r=>up.listen(0,'127.0.0.1',r));
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'pworker-task-tags-'));
  const config={dataDir:path.join(root,'data'),taskHome:root,defaultUpstream:'stub',upstreams:{stub:{baseUrl:`http://127.0.0.1:${up.address().port}`,noKey:true,formats:{openai:'/v1/chat/completions'}}},
    tiers:{small:[{upstream:'stub',model:'M'}]},policy:{allowedPurposes:['test:*','pworker:*'],untaggedDailyMax:100}};
  const server=await createPworkerServer({config,env:{}});
  const address=await server.listen(0);
  const url=`http://127.0.0.1:${address.port}`;
  try{
    const task={begin:{tier:'small',template:'Say ok.',code:'this.end(result)'}};
    const tagged=createPworkerClient({url,purpose:'test:caller',run:'run-42',priority:'background',autostart:false});
    const done=await tagged.task(task,{pollSeconds:1});
    assert.equal(done.status,'finished',done.error);assert.equal(done.result.value,'ok');
    const plain=await fetch(url+'/v1/tasks',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({task,input:{}})}).then(r=>r.json());
    assert.equal((await tagged.waitOp(plain.id,{pollSeconds:1})).status,'finished');
    const log=fs.readdirSync(config.dataDir).filter(f=>f.startsWith('requests-')).flatMap(f=>fs.readFileSync(path.join(config.dataDir,f),'utf8').split('\n').filter(Boolean).map(l=>JSON.parse(l)));
    const upstream=log.filter(r=>r.endpoint==='/v1/chat/completions'||r.tier==='small');
    assert.ok(upstream.some(r=>r.purpose==='test:caller'&&r.run==='run-42'),JSON.stringify(upstream.map(r=>[r.purpose,r.run])));
    assert.ok(upstream.some(r=>r.purpose==='pworker:tasks'),'a task without tags keeps the server purpose');
    assert.equal(JSON.parse(fs.readFileSync(path.join(root,'jobs',done.id+'.json'),'utf8')).tags.run,'run-42','the tags are kept with the task for a recovery');
  }finally{await server.close();up.close();fs.rmSync(root,{recursive:true,force:true});}
});

test('close releases the listener and resolves when the home was removed under the server',async()=>{
  const t=await fixture();
  const {port}=t.server.address();
  try {
    fs.rmSync(t.root,{recursive:true,force:true});
    await t.server.close();
    const probe=net.createServer();
    await new Promise((resolve,reject)=>{probe.once('error',reject);probe.listen(port,'127.0.0.1',resolve);});
    await new Promise(resolve=>probe.close(resolve));
  }finally{t.server.server.close();t.server.server.closeAllConnections?.();fs.rmSync(t.root,{recursive:true,force:true});}
});
