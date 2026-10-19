import './home.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateTask} from '../lib/pworker/task.mjs';
import * as sandbox from '../lib/sandbox.mjs';

test('legacy module sandbox entrypoint is absent',()=>{
  assert.equal(sandbox.runModule,undefined);
});

test('documented task declarations are valid strict JSON phase maps',()=>{
  let count=0;
  for(const name of ['PWORKER.md','docs/task-format-migration.md','docs/concepts.html','docs/batching.html']){
    const source=fs.readFileSync(new URL('../'+name,import.meta.url),'utf8');
    const blocks=[...source.matchAll(/```json\n([\s\S]*?)```|<pre><code>(\{[\s\S]*?)<\/code><\/pre>/g)];
    for(const block of blocks){
      const task=JSON.parse(block[1]??block[2]);
      if(name==='PWORKER.md' && (Object.hasOwn(task,'providers')||Object.hasOwn(task,'localLlm')))continue; // Configuration, not a task example.
      validateTask(task);count++;
    }
  }
  assert.ok(count>=5,`Expected task examples, found ${count}`);
});
