#!/usr/bin/env node
// Live smoke test of the standalone engine: a real `pworker serve` on a fresh home, a real runner install, a real (tiny) model on a
// real llama.cpp, one chat through a tier. It downloads a runner (up to ~0.8 GB for the CUDA build) and a model (~0.5 GB), starts
// processes and binds a port, so it is opt-in and is not part of `npm test`:
//
//   PWORKER_LIVE_ENGINE=1 node test/live/engine-smoke.mjs --out DIR [--port N] [--runner llama.cpp|llama.cpp-cpu]
//       [--model ID] [--expect-gpu auto|yes|no] [--home DIR] [--keep-home] [--timeout-min N]
//   PWORKER_LIVE_ENGINE=1 node test/live/engine-smoke.mjs --out DIR --probe ready-kill
//   PWORKER_LIVE_ENGINE=1 node test/live/engine-smoke.mjs --out DIR --probe record-kill [--at after-spawn|after-intent]
//
// Steps (default run): record the environment (architecture, GPU, git revision), serve on a fresh short home, install the runner,
// run the model until ready, read the launch record, assert the runner listens on a Unix socket under <home>/r and has no TCP
// listener, send one chat through a tier at 200 with `x-pworker-model: local-llm/...`, record the runner's offload lines (and
// fail when a GPU was expected and the model is not fully offloaded), run `ldd` on every ELF file of the runnable copy (none
// `not found`; with a GPU libcuda must resolve too), stop the model, and assert that no runner process, socket or launch record is
// left, then stop the proxy and assert no process of the home remains.
//
// Probes (crash recovery; each exits 3 "inconclusive" instead of passing when the setup it needs did not happen):
//   ready-kill   SIGKILL the serve process while the model is ready, check the orphaned runner is alive, run `pworker start`, and
//                require that the reaper ended that runner (its log line, the runner gone, socket and record removed).
//   record-kill  start serve with PWORKER_LOCAL_LLM_CRASH_AT=<at> (default after-spawn), run the model so serve kills itself
//                at the launch-record transition, check the state it left (an intent record and, after-spawn, a live child that has
//                no pid in the record), run `pworker start` and require no process, socket or record of the home left.
//
// Output: DIR/steps.json (the step log, written after every step), DIR/serve-*.log, DIR/runner-log-lines.txt, DIR/ldd.json.
// Exit codes: 0 every step passed; 1 a step failed; 2 refused or bad usage; 3 inconclusive. The Hugging Face token and the key of a
// run are never written: every text goes through scrub().

import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { strictLibraryCheck } from '../../lib/local-llm-host/elf.mjs';

const ROOT = fileURLToPath(new URL('../..', import.meta.url));
const BIN = path.join(ROOT, 'bin', 'pworker.mjs');
const DEFAULT_MODEL = 'qwen2.5-0.5b-instruct-q4_k_m';
const TIER = 'smoke';
const CRASH_ENV = 'PWORKER_LOCAL_LLM_CRASH_AT';
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function refuse(message, code = 2) {
    process.stderr.write(`${message}\n`);
    process.exit(code);
}

if (process.env.PWORKER_LIVE_ENGINE !== '1') {
    refuse('Refusing to run: this test downloads a runner and a model, starts processes and binds a port. Set PWORKER_LIVE_ENGINE=1 to run it.');
}
if (process.platform !== 'linux') refuse(`Refusing to run on ${process.platform}: the standalone engine is Linux-only in this release.`);

function parseArgs(argv) {
    const out = { flags: new Set(), values: {} };
    for (let index = 0; index < argv.length; index += 1) {
        const arg = argv[index];
        if (!arg.startsWith('--')) refuse(`Unexpected argument ${arg}`);
        const name = arg.slice(2);
        if (['keep-home'].includes(name)) out.flags.add(name);
        else if (['out', 'port', 'runner', 'model', 'expect-gpu', 'home', 'probe', 'at', 'timeout-min'].includes(name) && argv[index + 1] !== undefined) out.values[name] = argv[++index];
        else refuse(`Unknown or incomplete option ${arg}`);
    }
    return out;
}
const args = parseArgs(process.argv.slice(2));
const outDir = args.values.out ? path.resolve(args.values.out) : refuse('Usage: PWORKER_LIVE_ENGINE=1 node test/live/engine-smoke.mjs --out DIR [--probe ready-kill|record-kill]');
const probe = args.values.probe ?? null;
if (probe !== null && !['ready-kill', 'record-kill'].includes(probe)) refuse(`Unknown probe ${probe}; use ready-kill or record-kill.`);
const crashAt = args.values.at ?? 'after-spawn';
if (!['after-spawn', 'after-intent'].includes(crashAt)) refuse(`Unknown --at ${crashAt}; use after-spawn or after-intent.`);
const model = args.values.model ?? DEFAULT_MODEL;
const timeoutMs = Number(args.values['timeout-min'] ?? 40) * 60_000;

// ----- secrets and the step log ---------------------------------------------------------------------------------------------

const secrets = new Set();
if (process.env.HF_TOKEN) secrets.add(process.env.HF_TOKEN);
/** `text` without any secret value and without the value of `--api-key`. */
function scrub(text) {
    let result = String(text);
    for (const secret of secrets) if (secret) result = result.split(secret).join('***');
    return result.replace(/(--api-key[ =])\S+/g, '$1***').replace(/(Bearer )[A-Za-z0-9._~+/=-]{8,}/g, '$1***');
}
const tail = (text, max = 3000) => { const s = scrub(text); return s.length > max ? `...${s.slice(-max)}` : s; };

fs.mkdirSync(outDir, { recursive: true });
const record = { startedAt: new Date().toISOString(), probe, environment: null, steps: [], ok: null };
class StepFailed extends Error {}
class Inconclusive extends StepFailed {}
const saveRecord = () => fs.writeFileSync(path.join(outDir, 'steps.json'), `${scrub(JSON.stringify(record, null, 2))}\n`);

/** Runs one named step; records its outcome and detail; throws StepFailed (after recording) when `fn` throws. */
async function step(name, fn) {
    const entry = { name, startedAt: new Date().toISOString(), ok: false };
    record.steps.push(entry);
    const t0 = Date.now();
    try {
        entry.detail = (await fn()) ?? null;
        entry.ok = true;
    } catch (error) {
        entry.error = scrub(error?.message ?? error);
        if (error instanceof Inconclusive) entry.inconclusive = true;
        throw error instanceof StepFailed ? error : new StepFailed(entry.error);
    } finally {
        entry.ms = Date.now() - t0;
        saveRecord();
    }
}
const need = (condition, message) => { if (!condition) throw new StepFailed(message); };
const inconclusive = (message) => { throw new Inconclusive(message); };

// ----- environment ------------------------------------------------------------------------------------------------------------

function run(command, commandArgs, options = {}) {
    const result = spawnSync(command, commandArgs, { encoding: 'utf8', timeout: 30_000, ...options });
    return { status: result.status, stdout: (result.stdout ?? '').trim(), stderr: (result.stderr ?? '').trim(), error: result.error };
}
const smi = run('nvidia-smi', ['--query-gpu=name,driver_version', '--format=csv']);
const gpuPresent = !smi.error && smi.status === 0 && smi.stdout.split('\n').length > 1;
const runnerId = args.values.runner ?? (gpuPresent ? 'llama.cpp' : 'llama.cpp-cpu');
const gpuSetting = args.values['expect-gpu'] ?? 'auto';
if (!['auto', 'yes', 'no'].includes(gpuSetting)) refuse(`Unknown --expect-gpu ${gpuSetting}; use auto, yes or no.`);
const expectGpu = gpuSetting === 'auto' ? (runnerId === 'llama.cpp' && gpuPresent) : gpuSetting === 'yes';

const ownsHome = !args.values.home;
const home = fs.realpathSync(args.values.home ?? fs.mkdtempSync('/tmp/pws-'));
const port = Number(args.values.port ?? await new Promise((resolve) => {
    const server = net.createServer();
    server.listen(0, '127.0.0.1', () => { const { port: free } = server.address(); server.close(() => resolve(free)); });
}));
const layout = { root: path.join(home, 'local-llm'), sockets: path.join(home, 'r') };
layout.launch = path.join(layout.root, 'state', 'launch');
layout.runLog = path.join(layout.root, 'logs', 'runner.log');
layout.run = path.join(layout.root, 'run');
const base = `http://127.0.0.1:${port}`;

const childEnv = (extra = {}) => {
    // PWORKER_AUTOSTART=0: a CLI command never restarts a proxy this script has killed on purpose (only `pworker start` does).
    const env = { ...process.env, PWORKER_HOME: home, PWORKER_PORT: String(port), PWORKER_AUTOSTART: '0', ...extra };
    delete env.NODE_TEST_CONTEXT;
    delete env.PWORKER_URL;
    delete env.PWORKER_TOKEN;
    if (!Object.hasOwn(extra, CRASH_ENV)) delete env[CRASH_ENV];
    return env;
};

// ----- /proc helpers (Linux) --------------------------------------------------------------------------------------------------

const readStartTime = (pid) => {
    try {
        const text = fs.readFileSync(`/proc/${pid}/stat`, 'utf8');
        const rest = text.slice(text.lastIndexOf(')') + 1).trim().split(/\s+/);
        return rest[0] === 'Z' || rest[0] === 'X' ? null : rest[19];
    } catch { return null; }
};
const readCmdline = (pid) => { try { return fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').filter(Boolean); } catch { return null; } };
/** Whether `pid` is still the process that started at `startTime` (a reused pid is not). */
const same = (pid, startTime) => startTime !== null && readStartTime(pid) === startTime;
/** The pids (not this process) whose command line contains `needle`. */
function pidsWith(needle) {
    const found = [];
    for (const name of fs.readdirSync('/proc')) {
        if (!/^\d+$/.test(name) || Number(name) === process.pid) continue;
        const argv = readCmdline(name);
        if (argv && argv.some((part) => part.includes(needle))) found.push(Number(name));
    }
    return found;
}
function runnerIdentity(launch) {
    const argv = readCmdline(launch.pid);
    const alive = same(launch.pid, launch.startTime) && Boolean(argv) && argv.some((part, index) => part === launch.socketPath && argv[index - 1] === '--host');
    return { alive, argv };
}
function rememberKey(argv) {
    const index = argv ? argv.indexOf('--api-key') : -1;
    if (index >= 0 && argv[index + 1]) secrets.add(argv[index + 1]);
}
function readLaunchRecords() {
    let names = [];
    try { names = fs.readdirSync(layout.launch); } catch { return []; }
    return names.filter((name) => /^[0-9a-f]{8}\.json$/.test(name)).map((name) => JSON.parse(fs.readFileSync(path.join(layout.launch, name), 'utf8')));
}
const socketsLeft = () => { try { return fs.readdirSync(layout.sockets).filter((name) => name.endsWith('.sock')); } catch { return []; } };

/** The inodes of the sockets a process holds, and the inodes of TCP listeners and of Unix sockets by path. */
function socketInodes(pid) {
    const inodes = new Set();
    for (const fd of fs.readdirSync(`/proc/${pid}/fd`)) {
        try {
            const match = /^socket:\[(\d+)\]$/.exec(fs.readlinkSync(`/proc/${pid}/fd/${fd}`));
            if (match) inodes.add(match[1]);
        } catch { /* closed meanwhile */ }
    }
    return inodes;
}
function tcpListeners(inodes) {
    const listening = [];
    for (const table of ['/proc/net/tcp', '/proc/net/tcp6']) {
        let text = '';
        try { text = fs.readFileSync(table, 'utf8'); } catch { continue; }
        for (const line of text.split('\n').slice(1)) {
            const columns = line.trim().split(/\s+/);
            if (columns.length > 9 && columns[3] === '0A' && inodes.has(columns[9])) listening.push(`${table.split('/').pop()} ${columns[1]}`);
        }
    }
    return listening;
}
function unixSocketPaths(inodes) {
    const paths = [];
    for (const line of fs.readFileSync('/proc/net/unix', 'utf8').split('\n').slice(1)) {
        const columns = line.trim().split(/\s+/);
        if (columns.length >= 8 && inodes.has(columns[6])) paths.push(columns[7]);
    }
    return paths;
}

// ----- processes we start ------------------------------------------------------------------------------------------------------

/** The CLI, as a child process; resolves { code, signal, stdout, stderr } (scrubbed, tail only). */
function pw(commandArgs, { input = null, env = childEnv(), waitMs = timeoutMs } = {}) {
    return new Promise((resolve) => {
        const child = spawn(process.execPath, [BIN, ...commandArgs], { env, stdio: ['pipe', 'pipe', 'pipe'] });
        let stdout = '', stderr = '';
        child.stdout.on('data', (chunk) => { stdout += chunk; });
        child.stderr.on('data', (chunk) => { stderr += chunk; });
        const timer = setTimeout(() => child.kill('SIGKILL'), waitMs);
        child.on('close', (code, signal) => { clearTimeout(timer); resolve({ code, signal, stdout: tail(stdout), stderr: tail(stderr) }); });
        child.stdin.end(input ?? '');
    });
}

const started = [];   // { pid, startTime, what }: processes this script started, for the teardown
let serve = null;     // { child, pid, startTime, exited: Promise<{code, signal}> }

function startServe(label, env) {
    const log = fs.createWriteStream(path.join(outDir, `serve-${label}.log`));
    const child = spawn(process.execPath, [BIN, 'serve', '--port', String(port)], { env, stdio: ['ignore', 'pipe', 'pipe'] });
    for (const stream of [child.stdout, child.stderr]) stream.on('data', (chunk) => log.write(scrub(chunk.toString('utf8'))));
    const exited = new Promise((resolve) => child.on('exit', (code, signal) => { log.end(); resolve({ code, signal }); }));
    const startTime = readStartTime(child.pid);
    started.push({ pid: child.pid, startTime, what: `serve ${label}` });
    serve = { child, pid: child.pid, startTime, exited, label };
    return serve;
}

async function waitFor(what, test, { waitMs = 60_000, everyMs = 250 } = {}) {
    const deadline = Date.now() + waitMs;
    for (;;) {
        const value = await test();
        if (value) return value;
        if (Date.now() > deadline) throw new StepFailed(`Timed out after ${Math.round(waitMs / 1000)} s waiting for ${what}`);
        await sleep(everyMs);
    }
}

async function health() {
    try {
        const res = await fetch(`${base}/health`, { signal: AbortSignal.timeout(2000) });
        return res.ok ? await res.json() : null;
    } catch { return null; }
}
const serverPid = () => { try { return JSON.parse(fs.readFileSync(path.join(home, 'server.json'), 'utf8')).pid; } catch { return null; } };

async function waitForServe(expectedPid) {
    await waitFor(`the proxy of pid ${expectedPid} to answer`, async () => (await health()) && serverPid() === expectedPid, { waitMs: 90_000 });
}

async function engineStatus() {
    const res = await fetch(`${base}/v1/local-llm/status`, { signal: AbortSignal.timeout(10_000) });
    need(res.ok, `GET /v1/local-llm/status answered ${res.status}`);
    return res.json();
}

// ----- the steps ---------------------------------------------------------------------------------------------------------------

function gitFacts() {
    const sha = run('git', ['-C', ROOT, 'rev-parse', 'HEAD']);
    const dirty = run('git', ['-C', ROOT, 'status', '--porcelain']);
    return { sha: sha.stdout || null, dirty: dirty.stdout.length > 0 };
}

async function stepEnvironment() {
    await step('environment', () => {
        record.environment = {
            machine: run('uname', ['-m']).stdout, kernel: run('uname', ['-r']).stdout, node: process.version,
            gpu: gpuPresent ? smi.stdout : null, nvidiaSmi: gpuPresent ? 'present' : 'absent',
            git: gitFacts(), runner: runnerId, model, expectGpu, home, port,
            ldd: run('ldd', ['--version']).stdout.split('\n')[0] ?? null,
        };
        need(!(gpuSetting === 'yes' && !gpuPresent), 'a GPU was expected (--expect-gpu yes) but nvidia-smi did not answer');
        need(!(args.values.home && fs.readdirSync(home).length > 0), `--home ${home} is not empty; use a fresh folder`);
        return record.environment;
    });
}

async function stepPrepare() {
    await step('prepare the home', () => {
        const probePath = path.join(layout.sockets, '0123abcd.sock');
        need(Buffer.byteLength(probePath) <= 107, `the home ${home} is too long for a socket path (${Buffer.byteLength(probePath)} > 107 bytes); use a shorter --home`);
        fs.writeFileSync(path.join(home, 'config.json'), `${JSON.stringify({ tiers: { [TIER]: [{ upstream: 'local-llm', model, runner: runnerId }] } }, null, 2)}\n`, { mode: 0o600 });
        return { socketPathBytes: Buffer.byteLength(probePath) };
    });
}

async function stepServe(label, extraEnv = {}) {
    await step(`serve (${label})`, async () => {
        const proc = startServe(label, childEnv(extraEnv));
        await waitForServe(proc.pid);
        return { pid: proc.pid, port };
    });
}

async function stepInstall() {
    await step(`runner install ${runnerId}`, async () => {
        const result = await pw(['models', 'runner', 'install', runnerId], { waitMs: timeoutMs });
        need(result.code === 0, `exit ${result.code ?? result.signal}: ${result.stderr || result.stdout}`);
        return { output: result.stdout.split('\n').slice(-3).join('\n') };
    });
}

async function stepRunToReady() {
    await step(`run ${model} on ${runnerId} until ready`, async () => {
        const result = await pw(['models', 'run', model, '--runner', runnerId], { waitMs: timeoutMs });
        need(result.code === 0, `exit ${result.code ?? result.signal}: ${result.stderr || result.stdout}`);
        return { output: result.stdout.split('\n').slice(-3).join('\n') };
    });
}

/** The one launch record of the running model, with its identity checked. */
async function stepLaunchRecord() {
    let launch = null;
    await step('launch record and runner identity', () => {
        const records = readLaunchRecords();
        need(records.length === 1, `expected one launch record, found ${records.length}`);
        launch = records[0];
        need(launch.phase === 'spawned' && Number.isInteger(launch.pid) && launch.startTime, `the record is not complete: ${JSON.stringify({ ...launch, socketPath: undefined })}`);
        need(launch.socketPath.startsWith(`${layout.sockets}${path.sep}`), `the socket ${launch.socketPath} is not under ${layout.sockets}`);
        const identity = runnerIdentity(launch);
        need(identity.alive, 'the runner of the record is not alive with that pid, start time and --host');
        rememberKey(identity.argv);
        return { startId: launch.startId, pid: launch.pid, startTime: launch.startTime, socketPath: launch.socketPath, argv: scrub(identity.argv.join(' ')).split(' ').slice(0, 40).join(' ') };
    });
    return launch;
}

async function stepSocketOnly(launch) {
    await step('the runner listens on a Unix socket under <home>/r and has no TCP listener', () => {
        const inodes = socketInodes(launch.pid);
        const tcp = tcpListeners(inodes);
        need(tcp.length === 0, `the runner has TCP listeners: ${tcp.join(', ')}`);
        const unix = unixSocketPaths(inodes);
        need(unix.includes(launch.socketPath), `the runner holds no Unix socket at ${launch.socketPath} (it holds ${JSON.stringify(unix)})`);
        const stat = fs.lstatSync(launch.socketPath);
        need(stat.isSocket(), `${launch.socketPath} is not a socket`);
        const dir = fs.statSync(layout.sockets);
        need((dir.mode & 0o777) === 0o700, `${layout.sockets} has mode ${(dir.mode & 0o777).toString(8)}, not 700`);
        return { socketInodes: inodes.size, tcpListeners: 0, unixPaths: unix.map((p) => path.basename(p)) };
    });
}

async function stepChat() {
    await step('one chat through the tier at 200 from local-llm', async () => {
        const res = await fetch(`${base}/v1/chat/completions`, {
            method: 'POST', headers: { 'content-type': 'application/json', 'x-pworker-purpose': 'pworker:smoke' },
            body: JSON.stringify({ model: TIER, max_tokens: 24, temperature: 0, messages: [{ role: 'user', content: 'Say hello in three words.' }] }),
            signal: AbortSignal.timeout(300_000),
        });
        const text = await res.text();
        need(res.status === 200, `status ${res.status}: ${tail(text, 500)}`);
        const servedBy = res.headers.get('x-pworker-model') ?? '';
        need(servedBy.startsWith('local-llm/'), `x-pworker-model is ${JSON.stringify(servedBy)}, not local-llm/...`);
        const body = JSON.parse(text);
        const content = body?.choices?.[0]?.message?.content;
        need(typeof content === 'string' && content.length > 0, 'the answer has no content');
        return { status: res.status, xPworkerModel: servedBy, content: tail(content, 160), usage: body.usage ?? null };
    });
}

async function stepOffload() {
    await step(expectGpu ? 'the model is offloaded to the GPU' : 'where the model runs (no GPU expected)', async () => {
        const status = await engineStatus();
        const report = status.runnerReport ?? {};
        let lines = [];
        try { lines = fs.readFileSync(layout.runLog, 'utf8').split('\n'); } catch { /* the in-memory lines below */ }
        if (!lines.length) lines = (status.logs ?? []).map((entry) => entry.line);
        const wanted = lines.filter((line) => /offloaded \d+\/\d+ layers|using device|ggml_cuda_init|CUDA\d|load_backend|Vulkan\d/.test(line)).map((line) => scrub(line).slice(0, 300));
        fs.writeFileSync(path.join(outDir, 'runner-log-lines.txt'), `${wanted.join('\n')}\n`);
        const offloaded = report.offloaded ?? null;
        const detail = { profile: status.profile ?? null, device: report.device ?? null, offloaded, modelMiB: report.modelMiB ?? null, logLines: wanted.length };
        if (expectGpu) {
            need(offloaded && offloaded.of > 0 && offloaded.layers === offloaded.of, `a GPU was expected but the model is not fully offloaded: ${JSON.stringify(offloaded)}`);
            need(/^CUDA\d/.test(report.device ?? ''), `a GPU was expected but the runner reports the device ${JSON.stringify(report.device)}`);
            need(wanted.some((line) => /offloaded \d+\/\d+ layers/.test(line)), 'the runner log has no "offloaded N/N layers" line');
        } else {
            need(!(offloaded && offloaded.layers > 0), `no GPU was expected but ${offloaded?.layers} layers are offloaded`);
        }
        return detail;
    });
}

async function stepLdd() {
    await step('ldd on every ELF file of the runnable copy', () => {
        const parent = path.join(layout.run, runnerId);
        const versions = fs.readdirSync(parent).filter((name) => !name.startsWith('.'));
        need(versions.length === 1, `expected one runnable copy under ${parent}, found ${JSON.stringify(versions)}`);
        const dir = path.join(parent, versions[0]);
        const result = strictLibraryCheck(dir);
        fs.writeFileSync(path.join(outDir, 'ldd.json'), `${JSON.stringify({ dir, ...result }, null, 2)}\n`);
        need(result.ok, result.problems.join('; '));
        // On a GPU host the driver library must resolve too: only a machine without a driver may miss it.
        if (expectGpu) need(result.driverOnly.length === 0, `libcuda.so.1 does not resolve for: ${result.driverOnly.join(', ')}`);
        return { dir, elfFiles: result.checked, driverOnly: result.driverOnly.length };
    });
}

async function stepStopAndClean(launch) {
    await step('stop the model: no runner, socket or record is left', async () => {
        const result = await pw(['models', 'stop'], { waitMs: 120_000 });
        need(result.code === 0, `exit ${result.code ?? result.signal}: ${result.stderr || result.stdout}`);
        await waitFor('the runner to exit', () => !same(launch.pid, launch.startTime), { waitMs: 60_000 });
        need(pidsWith(launch.socketPath).length === 0, `a process still names ${path.basename(launch.socketPath)}`);
        await waitFor('the launch record and the socket to be removed', () => readLaunchRecords().length === 0 && socketsLeft().length === 0, { waitMs: 30_000 });
        return { runnerGone: true, socketFileGone: !fs.existsSync(launch.socketPath) };
    });
}

async function stepStopProxy() {
    await step('stop the proxy: nothing of the home is left running', async () => {
        const result = await pw(['stop'], { waitMs: 90_000 });
        need(result.code === 0, `exit ${result.code ?? result.signal}: ${result.stderr || result.stdout}`);
        if (serve) await Promise.race([serve.exited, sleep(60_000)]);
        const left = [...started.filter((entry) => same(entry.pid, entry.startTime)).map((entry) => entry.pid), ...pidsWith(`${layout.sockets}${path.sep}`)];
        need(left.length === 0, `processes of the home are left: ${[...new Set(left)].join(', ')}`);
        return { left: 0 };
    });
}

// ----- the probes --------------------------------------------------------------------------------------------------------------

async function killServeHard() {
    need(serve && same(serve.pid, serve.startTime), 'the serve process is not the one this script started');
    process.kill(serve.pid, 'SIGKILL');
    return Promise.race([serve.exited, sleep(10_000).then(() => null)]);
}

async function probeReadyKill() {
    await stepServe('first');
    await stepInstall();
    await stepRunToReady();
    const launch = await stepLaunchRecord();
    await step('SIGKILL the serve process: the orphaned runner must be alive (else the probe proves nothing)', async () => {
        const exit = await killServeHard();
        need(exit !== null, 'serve did not exit after SIGKILL');
        const identity = runnerIdentity(launch);
        if (!identity.alive) inconclusive('the runner died with its proxy, so there is no orphan to reap');
        return { serveSignal: exit.signal, orphan: { pid: launch.pid, startTime: launch.startTime }, record: readLaunchRecords().length };
    });
    await step('pworker start: the reaper ends the orphan', async () => {
        const result = await pw(['start'], { waitMs: 120_000 });
        need(result.code === 0, `pworker start exit ${result.code ?? result.signal}: ${result.stderr || result.stdout}`);
        const newPid = serverPid();
        need(newPid && newPid !== serve.pid, `the recorded proxy ${newPid} is not a new process`);
        started.push({ pid: newPid, startTime: readStartTime(newPid), what: 'serve (started by pworker start)' });
        await waitFor('the orphan to be gone', () => !runnerIdentity(launch).alive, { waitMs: 30_000 });
        const log = fs.readFileSync(path.join(home, 'logs', 'pworker.log'), 'utf8');
        const evidence = new RegExp(`ended the orphaned runner ${launch.startId} \\(pid ${launch.pid}\\)`).exec(log);
        need(evidence, `the proxy log has no "ended the orphaned runner ${launch.startId} (pid ${launch.pid})" line`);
        need(pidsWith(launch.socketPath).length === 0, 'a process still names the orphan\'s socket');
        need(!fs.existsSync(launch.socketPath), 'the orphan\'s socket file is still there');
        need(readLaunchRecords().length === 0, 'a launch record is left');
        return { newProxyPid: newPid, reaperLine: scrub(evidence[0]) };
    });
    await stepStopProxy();
}

async function probeRecordKill() {
    await stepServe('crashing', { [CRASH_ENV]: crashAt });
    await stepInstall();
    let crashed = null;
    await step(`run ${model}: serve kills itself ${crashAt}`, async () => {
        const result = await pw(['models', 'run', model, '--runner', runnerId], { waitMs: timeoutMs });
        need(result.code !== 0, 'models run succeeded although serve was to crash');
        crashed = await Promise.race([serve.exited, sleep(30_000).then(() => null)]);
        if (!crashed || crashed.signal !== 'SIGKILL') inconclusive(`serve did not die by SIGKILL at ${crashAt} (${JSON.stringify(crashed)}); the hook did not fire`);
        return { cliExit: result.code, serveSignal: crashed.signal };
    });
    let leftover = null;
    await step(`the state the crash left (${crashAt})`, () => {
        const records = readLaunchRecords();
        need(records.length === 1, `expected one launch record, found ${records.length}`);
        need(records[0].phase === 'intent', `the record is in phase ${records[0].phase}, not intent`);
        leftover = records[0];
        const children = pidsWith(leftover.socketPath);
        if (crashAt === 'after-spawn') {
            if (children.length !== 1) inconclusive(`expected the unrecorded child (one process naming the socket), found ${children.length}`);
            for (const pid of children) { rememberKey(readCmdline(pid)); started.push({ pid, startTime: readStartTime(pid), what: 'runner left by the crash' }); }
        } else if (children.length !== 0) {
            need(false, `a process exists although the crash came before the spawn: ${children.join(', ')}`);
        }
        return { phase: records[0].phase, startId: leftover.startId, unrecordedChildren: children.length };
    });
    await step('pworker start: no unidentifiable child is left', async () => {
        const result = await pw(['start'], { waitMs: 120_000 });
        need(result.code === 0, `pworker start exit ${result.code ?? result.signal}: ${result.stderr || result.stdout}`);
        const newPid = serverPid();
        need(newPid && newPid !== serve.pid, `the recorded proxy ${newPid} is not a new process`);
        started.push({ pid: newPid, startTime: readStartTime(newPid), what: 'serve (started by pworker start)' });
        await waitFor('the child to be gone', () => pidsWith(leftover.socketPath).length === 0, { waitMs: 30_000 });
        const log = fs.readFileSync(path.join(home, 'logs', 'pworker.log'), 'utf8');
        const reaped = crashAt === 'after-spawn' ? /ended the orphaned runner/.test(log) : /named no live runner; removed/.test(log);
        need(reaped, `the proxy log has no reaper line for ${leftover.startId}`);
        need(pidsWith(layout.sockets).length === 0, 'a process of the home still names the socket directory');
        need(readLaunchRecords().length === 0, 'a launch record is left');
        need(socketsLeft().length === 0, 'a socket file is left');
        return { newProxyPid: newPid };
    });
    await stepStopProxy();
}

async function mainRun() {
    await stepServe('main');
    await stepInstall();
    await stepRunToReady();
    const launch = await stepLaunchRecord();
    await stepSocketOnly(launch);
    await stepChat();
    await stepOffload();
    await stepLdd();
    await stepStopAndClean(launch);
    await stepStopProxy();
}

// ----- teardown ----------------------------------------------------------------------------------------------------------------

/** Signals a process this script started, only while it still is that process (pid and start time are read again right before). */
function signalIfSame(entry, signal) {
    if (!same(entry.pid, entry.startTime)) return false;
    try { process.kill(entry.pid, signal); return true; } catch { return false; }
}
async function teardown() {
    const victims = started.filter((entry) => same(entry.pid, entry.startTime));
    for (const entry of victims) signalIfSame(entry, 'SIGTERM');
    const deadline = Date.now() + 30_000;
    while (Date.now() < deadline && victims.some((entry) => same(entry.pid, entry.startTime))) await sleep(200);
    const stubborn = victims.filter((entry) => same(entry.pid, entry.startTime));
    for (const entry of stubborn) signalIfSame(entry, 'SIGKILL');
    // A runner that survived and that no record led to: found by its `--host` socket in this home's socket folder, identity read again before the signal.
    for (const pid of pidsWith(`${layout.sockets}${path.sep}`)) {
        const entry = { pid, startTime: readStartTime(pid) };
        if (entry.startTime !== null) signalIfSame(entry, 'SIGKILL');
    }
    if (ownsHome && !args.flags.has('keep-home') && home.startsWith('/tmp/pws-') && home.length > '/tmp/pws-'.length) {
        fs.rmSync(home, { recursive: true, force: true });
    }
    return { signalled: victims.map((entry) => entry.what) };
}

// ----- main --------------------------------------------------------------------------------------------------------------------

let exitCode = 0;
try {
    await stepEnvironment();
    await stepPrepare();
    if (probe === 'ready-kill') await probeReadyKill();
    else if (probe === 'record-kill') await probeRecordKill();
    else await mainRun();
} catch (error) {
    exitCode = error instanceof Inconclusive ? 3 : 1;
    if (!(error instanceof StepFailed)) {
        record.steps.push({ name: 'unexpected error', ok: false, error: scrub(error?.stack ?? error) });
    }
} finally {
    try { await step('teardown (only what this script started)', teardown); } catch { exitCode ||= 1; }
    record.ok = exitCode === 0 && record.steps.every((entry) => entry.ok);
    record.finishedAt = new Date().toISOString();
    if (!record.ok && exitCode === 0) exitCode = 1;
    saveRecord();
    process.stdout.write(`${record.steps.map((entry) => `${entry.ok ? 'ok    ' : entry.inconclusive ? 'INCONC' : 'FAILED'} ${entry.name}${entry.ok ? '' : `: ${entry.error}`}`).join('\n')}\n`);
    process.stdout.write(`${record.ok ? 'PASSED' : exitCode === 3 ? 'INCONCLUSIVE' : 'FAILED'}: ${path.join(outDir, 'steps.json')}\n`);
    process.exit(exitCode);
}
