// A stand-in for the proxy process, run by the real-process launcher test (test/local-launcher.test.mjs): a real starter spawns the
// fixture server, the runner's PID is printed, and the process then waits. The test kills it with SIGKILL and checks the server lives on.
import { createLocalStarter } from '../lib/local.mjs';

const [serverScript, port, logFile] = process.argv.slice(2);
const starter = createLocalStarter({ name: 'fixture', baseUrl: `http://127.0.0.1:${port}`,
  start: { script: serverScript, requireFreeGpu: false, startTimeoutMs: 10000, logFile, args: ['--lifetime-ms', '40000'] } }, { gpuPids: () => [] });
if (!(await starter.ensure())) { console.error(JSON.stringify(starter.status().last_refusal)); process.exit(2); }
console.log(JSON.stringify({ runnerPid: starter.status().pid }));
setTimeout(() => process.exit(0), 40000); // never outlives a failed test
