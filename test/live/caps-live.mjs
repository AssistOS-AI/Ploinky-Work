#!/usr/bin/env node
// Live end-to-end test of the pworker hardware caps (`localLlm.caps`) and the engine work around them, on real processes: a real
// `pworker serve` on a fresh home per run, a real runner and a real (tiny) model, real HTTP, real `ps`. It downloads the pinned runner
// (11.3 MiB on Apple Silicon, up to ~0.8 GB for the CUDA build) and the model (0.46 GiB), starts processes and binds a port, so it is
// opt-in and is not part of `npm test`:
//
//   PWORKER_LIVE_CAPS=1 node test/live/caps-live.mjs --out DIR [--home DIR] [--port N] [--keep-home] [--only L1,L2,...] [--model ID]
//       [--tmp DIR] [--settle-min N] [--timeout-min N]
//
//   npm run test:live:caps -- --out DIR      (the same, with PWORKER_LIVE_CAPS=1 set in your shell)
//
// One home serves the whole run, so the runner and the model download once. Caps are fixed for the life of a `serve` process, so each
// case that needs other caps restarts `serve` with another project configuration (PWORKER_CONFIG). Every cap value is derived from
// the preview estimates of THIS machine (test/live/caps-derive.mjs), never from numbers measured elsewhere.
//
// Profile (what the machine runs): macOS arm64 Metal (`llama.cpp`); Linux with an answering nvidia-smi the dedicated CUDA build
// (`llama.cpp`); any other Linux the CPU build (`llama.cpp-cpu`). Model: qwen2.5-0.5b-instruct-q4_k_m.
//
// Cases (a case records pass, fail, skip or inconclusive, each check with its evidence):
//   L1  ram + threads   a threads cap below the profile default reaches the runner's argv; status prints the caps; a chat answers 200
//   L2  count caps      maxContext/maxParallel/threads: the default is clamped, explicit values above a cap are refused naming the cap
//                       (CLI and HTTP 400 invalid_params), and a refused replace leaves the running runner alone
//   L3  ram too small   a ram cap below the need: preview and run are refused as incompatible, no launch record, no runner
//   L4  invalid caps    ram 0, gpuMemory "lots", an unknown key (and gpuMemory on macOS): status says invalid_caps, HTTP 503, serve stays up
//   L5  fit context     mode fit lowers a default ctx that does not fit; the run launches with it; an explicit ctx is refused
//   L6  fit layers      dedicated only: fit lowers nGpuLayers; the estimate's RAM grows as layers drop
//   L7  ourUse          while a model runs under a tight ram cap, a preview's available memory is lower by about the runner's RSS
//   L8  gpuMemory share dedicated only: below the GPU need refused, above admitted, and nvidia-smi shows the runner within the cap + 10 %
//   L9  RSS reader      readRunnerMemory equals `ps -o rss` of the runner within 10 %
//   L10 guard backstop  guardSample/guardBreach with the real reader and a ram below the live RSS answer { stop } twice; the runner is not killed
//   L11 identity (G5)   with no caps: no fit, no limits.budget, no capped values in the argv
// Run order: baseline (no caps, which is L11), L1, L2, L3, L4, L5, L6, L7 with L9 and L10, L8.
//
// Safety: every pworker invocation sets PWORKER_HOME and PWORKER_CONFIG (and PWORKER_PORT, PWORKER_AUTOSTART=0, no PWORKER_URL or
// PWORKER_TOKEN), and has a subcommand: a bare `pworker` would auto-start a proxy in ~/.pworker. The run asserts that ~/.pworker was not
// created (or, if it existed, not changed). The port is free and never 18080. Only processes this script started are signalled: serve
// by `pworker stop`, the model by `models stop`, a survivor by pid and start time. Every `--api-key` value is scrubbed from everything
// this script stores or prints.
//
// Output: DIR/steps.json (written after every check), DIR/serve-*.log, DIR/commands.log, DIR/config/*.json.
// Exit codes: 0 every selected case passed or skipped; 1 a case or the cleanup failed; 2 refused or bad usage; 3 inconclusive.

import { execFileSync, spawn, spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineCapLimits, parseCaps } from '../../lib/local-llm-host/caps.mjs';
import { createRunnerMemoryReader } from '../../lib/local-llm-host/runnerMemory.mjs';
import { detectGpu } from './gpu-detect.mjs';
import {
    FIT_CONTEXT_FLOOR, MIB, availableNowBytes, halvedContext, gpuCapAdmitting, insufficientNowCause, parseRunnerArgv, planGpuFitContext, planGpuFitLayers,
    planOurUseCap, planRamFit, ramAdmitMargin, ramNeedOf, settleFailedCase, within,
} from './caps-derive.mjs';

const ROOT = fileURLToPath(new URL('../..', import.meta.url));
const BIN = path.join(ROOT, 'bin', 'pworker.mjs');
const DEFAULT_MODEL = 'qwen2.5-0.5b-instruct-q4_k_m';
const CASE_IDS = Object.freeze(['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11']);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function refuse(message, code = 2) {
    process.stderr.write(`${message}\n`);
    process.exit(code);
}

if (process.env.PWORKER_LIVE_CAPS !== '1') {
    refuse('Refusing to run: this test downloads a runner and a model, starts processes and binds a port. Set PWORKER_LIVE_CAPS=1 to run it.');
}
if (!['linux', 'darwin'].includes(process.platform)) refuse(`Refusing to run on ${process.platform}: the caps suite needs Linux or macOS.`);
if (process.platform === 'darwin' && process.arch !== 'arm64') refuse(`Refusing to run on darwin-${process.arch}: the caps suite covers Apple Silicon (Metal) on macOS.`);

function parseArgs(argv) {
    const out = { flags: new Set(), values: {} };
    for (let index = 0; index < argv.length; index += 1) {
        const arg = argv[index];
        if (!arg.startsWith('--')) refuse(`Unexpected argument ${arg}`);
        const name = arg.slice(2);
        if (name === 'keep-home') out.flags.add(name);
        else if (['out', 'home', 'port', 'only', 'model', 'tmp', 'settle-min', 'timeout-min'].includes(name) && argv[index + 1] !== undefined) out.values[name] = argv[++index];
        else refuse(`Unknown or incomplete option ${arg}`);
    }
    return out;
}
const args = parseArgs(process.argv.slice(2));
const outDir = args.values.out ? path.resolve(args.values.out) : refuse('Usage: PWORKER_LIVE_CAPS=1 node test/live/caps-live.mjs --out DIR [--home DIR] [--port N] [--keep-home] [--only L1,L2,...] [--model ID]');
const only = args.values.only === undefined ? null : new Set(args.values.only.split(',').map((id) => id.trim().toUpperCase()).filter(Boolean));
if (only) for (const id of only) if (!CASE_IDS.includes(id)) refuse(`Unknown case ${id}; use ${CASE_IDS.join(', ')}.`);
const selected = (id) => only === null || only.has(id);
const MODEL = args.values.model ?? DEFAULT_MODEL;
const settleMs = Number(args.values['settle-min'] ?? 20) * 60_000;
const timeoutMs = Number(args.values['timeout-min'] ?? 40) * 60_000;
const tmpBase = args.values.tmp ? path.resolve(args.values.tmp) : '/tmp';
if (!Number.isFinite(settleMs) || !Number.isFinite(timeoutMs)) refuse('--settle-min and --timeout-min must be numbers.');
if (args.values.home !== undefined) {
    const named = path.resolve(args.values.home);
    if (!fs.existsSync(named) || !fs.statSync(named).isDirectory()) refuse(`--home ${named} is not an existing folder; create an empty one or omit --home`);
    if (fs.readdirSync(named).length > 0) refuse(`--home ${named} is not empty; use a fresh folder`);
}
if (args.values.port !== undefined && (!/^\d+$/.test(args.values.port) || Number(args.values.port) === 18080 || Number(args.values.port) < 1024 || Number(args.values.port) > 65535)) {
    refuse(`--port ${args.values.port} is not a usable port (a number from 1024 to 65535, never 18080).`);
}

// Every identity and leftover check goes through /bin/ps, whose failure reads as "no such process": a host without a working ps would pass
// "nothing is left" without testing anything. So ps must answer for this very process before anything starts.
{
    const probe = (flags) => spawnSync('/bin/ps', [...flags, '-p', String(process.pid)], { encoding: 'utf8', timeout: 15_000, env: { PATH: '/usr/bin:/bin', LC_ALL: 'C' } });
    const answers = [['-o', 'pid='], ['-ww', '-o', 'command='], ['-o', 'lstart='], ['-o', 'rss=']].map(probe);
    if (answers.some((answer) => answer.status !== 0 || !String(answer.stdout).trim()) || String(answers[0].stdout).trim() !== String(process.pid)) {
        refuse('Refusing to run: /bin/ps does not report pid, command, lstart and rss for this process (on Linux, install procps). The suite cannot check for leftover processes without it.');
    }
}

// ----- secrets and the record ------------------------------------------------------------------------------------------------

const secrets = new Set();
if (process.env.HF_TOKEN) secrets.add(process.env.HF_TOKEN);
/** `text` without any secret value and without the value of `--api-key`. */
function scrub(text) {
    let result = String(text);
    for (const secret of secrets) if (secret) result = result.split(secret).join('***');
    return result.replace(/(--api-key[ =])\S+/g, '$1***').replace(/(Bearer )[A-Za-z0-9._~+/=-]{8,}/g, '$1***');
}
/** Scrubbed text of at most `max` characters, cut in the middle so that both ends show. */
function ev(text, max = 700) {
    const clean = scrub(text).trim();
    return clean.length > max ? `${clean.slice(0, Math.floor(max / 2))} ... ${clean.slice(-Math.floor(max / 2))}` : clean;
}
const runnerKeys = new Set(); // the per-start keys of the runners seen, for the final check that no output file holds one
function rememberKeyIn(command) {
    const match = /--api-key[ =](\S+)/.exec(command);
    if (match && match[1] !== '***') { secrets.add(match[1]); runnerKeys.add(match[1]); }
}

fs.mkdirSync(path.join(outDir, 'config'), { recursive: true });
const record = { startedAt: new Date().toISOString(), environment: null, derived: {}, steps: [], cases: [], ok: null };
const saveRecord = () => fs.writeFileSync(path.join(outDir, 'steps.json'), `${scrub(JSON.stringify(record, null, 2))}\n`);
const commandLogPath = path.join(outDir, 'commands.log');
const commandLog = fs.createWriteStream(commandLogPath);
let commandLogOpen = true;
// After the stream has ended (teardown rewrites the file when it finds a secret) a line is appended to the file itself, never written to the ended stream.
const logCommand = (line) => {
    const text = `${scrub(line)}\n`;
    if (commandLogOpen) { commandLog.write(text); return; }
    try { fs.appendFileSync(commandLogPath, text); } catch { /* the log is evidence, not a result */ }
};
/** Ends the commands.log stream and waits until the file is closed. Safe to call twice. */
async function closeCommandLog() {
    if (!commandLogOpen) return;
    commandLogOpen = false;
    await new Promise((resolve) => { commandLog.once('close', resolve); commandLog.end(); });
}

class Skip extends Error {}
class Inconclusive extends Error {}
class Failure extends Error {}
const need = (condition, message) => { if (!condition) throw new Failure(message); };

let current = null; // the case being run
/** One check of the current case: `ok` decides pass or fail; `evidence` is what was observed. */
function check(name, ok, evidence = '') {
    current.checks.push({ name, status: ok ? 'pass' : 'fail', evidence: ev(evidence, 900) });
    saveRecord();
    return Boolean(ok);
}
/** A check that did not apply. `core`: it is the point of the case, so the case is a skip, not a pass, when nothing else failed. */
function skipCheck(name, reason, { core = false } = {}) {
    current.checks.push({ name, status: 'skip', evidence: ev(reason, 900) });
    if (core) { current.coreSkipped = true; current.reason = ev(reason, 600); }
    saveRecord();
}

/** Runs one case body; a case that is not selected is not run. `hidden`: the body is setup for another case, recorded as a step. */
async function runCase(id, title, body, { hidden = false } = {}) {
    if (!hidden && !selected(id)) return;
    const entry = { id, title, status: 'running', checks: [], startedAt: new Date().toISOString() };
    (hidden ? record.steps : record.cases).push(entry);
    const previous = current;
    current = entry;
    try {
        await body();
        const statuses = entry.checks.map((item) => item.status);
        entry.status = statuses.includes('fail') ? 'fail' : (statuses.length > 0 && statuses.every((status) => status === 'skip')) || entry.coreSkipped ? 'skip' : 'pass';
        if (statuses.length === 0) { entry.status = 'fail'; entry.error = 'the case made no check'; }
    } catch (error) {
        entry.error = scrub(error?.message ?? error);
        if (error instanceof Skip) entry.status = 'skip';
        else if (error instanceof Inconclusive) entry.status = entry.checks.some((item) => item.status === 'fail') ? 'fail' : 'inconclusive';
        else {
            entry.status = 'fail';
            if (!(error instanceof Failure)) entry.stack = scrub(error?.stack ?? '');
        }
    } finally {
        // A case that failed after the engine's memory guard stopped its runner for the machine's memory saw a busy machine, not a caps result.
        // A case with a failed check keeps its failure: that check is a finding, whatever the guard did afterwards.
        if (entry.status === 'fail' && !entry.checks.some((item) => item.status === 'fail') && serveNow) {
            const settled = settleFailedCase({ status: entry.status, checks: entry.checks, guardStopped: await machineGuardStopped() });
            if (settled !== entry.status) {
                entry.status = settled;
                entry.reason = 'the engine\'s memory guard stopped the runner for the machine\'s memory (the error above is kept as evidence)';
            }
        }
        entry.finishedAt = new Date().toISOString();
        current = previous;
        saveRecord();
    }
}
async function machineGuardStopped() {
    try {
        const response = await fetch(`${base}/v1/local-llm/status`, { signal: AbortSignal.timeout(10_000) });
        return MACHINE_GUARD_STOP.test(await response.text());
    } catch { return false; }
}
/** A selected case that does not apply on this machine, recorded as skipped with the reason (no serve is started for it). */
async function skipCase(id, title, reason) {
    await runCase(id, title, async () => { throw new Skip(reason); });
}
function recordCaseFailure(id, error) {
    record.cases.push({ id, title: '(setup)', status: error instanceof Inconclusive ? 'inconclusive' : 'fail', checks: [], error: scrub(error?.message ?? error) });
    saveRecord();
}

// ----- machine, profile, paths -----------------------------------------------------------------------------------------------

function run(command, commandArgs, options = {}) {
    const result = spawnSync(command, commandArgs, { encoding: 'utf8', timeout: 30_000, ...options });
    return { status: result.status, stdout: (result.stdout ?? '').trim(), stderr: (result.stderr ?? '').trim(), error: result.error };
}
const darwin = process.platform === 'darwin';
const smi = darwin ? { error: true } : run('nvidia-smi', ['--query-gpu=name,driver_version', '--format=csv']);
const smiAnswered = !smi.error && smi.status === 0 && smi.stdout.split('\n').length > 1;
const gpu = darwin ? { gpuPresent: false, problem: null } : detectGpu({ smiAnswered, setting: 'auto' });
if (gpu.problem) refuse(`FAILED: ${gpu.problem}`, 1);
const expectedProfile = darwin ? 'metal' : gpu.gpuPresent ? 'dedicated' : 'cpu';
const RUNNER = expectedProfile === 'cpu' ? 'llama.cpp-cpu' : 'llama.cpp';

// ~/.pworker must not be created, or changed, by this run.
const realPworker = path.join(os.homedir(), '.pworker');
function signatureOf(dir) {
    const lines = [];
    const walk = (current, depth) => {
        let names = [];
        try { names = fs.readdirSync(current).sort(); } catch { return; }
        for (const name of names) {
            if (lines.length > 50_000) return;
            const full = path.join(current, name);
            let stat;
            try { stat = fs.lstatSync(full); } catch { continue; }
            lines.push(`${path.relative(dir, full)}:${stat.size}:${stat.mtimeMs}`);
            if (stat.isDirectory() && !stat.isSymbolicLink() && depth < 8) walk(full, depth + 1);
        }
    };
    walk(dir, 0);
    return crypto.createHash('sha256').update(lines.join('\n')).digest('hex');
}
const pworkerBefore = fs.existsSync(realPworker) ? { existed: true, signature: signatureOf(realPworker) } : { existed: false };

if (!fs.existsSync(tmpBase) || !fs.statSync(tmpBase).isDirectory()) refuse(`--tmp ${tmpBase} is not an existing folder.`);
const ownsHome = !args.values.home;
const home = fs.realpathSync(args.values.home ?? fs.mkdtempSync(path.join(fs.realpathSync(tmpBase), 'pwcl-')));
// A Unix socket path is limited (103 bytes on macOS), so the sockets go to a short folder of their own; it has mode 0700 (mkdtemp).
const socketDir = fs.realpathSync(fs.mkdtempSync(path.join(fs.realpathSync(tmpBase), 'pwcs-')));
const port = Number(args.values.port ?? await new Promise((resolve) => {
    const server = net.createServer();
    server.listen(0, '127.0.0.1', () => { const { port: free } = server.address(); server.close(() => resolve(free)); });
}));
/** Binds `port` on 127.0.0.1 and releases it: true when it was free, else the error code. */
const portIsFree = (candidate) => new Promise((resolve) => {
    const server = net.createServer();
    server.once('error', (error) => resolve(error?.code ?? String(error)));
    server.listen(candidate, '127.0.0.1', () => server.close(() => resolve(true)));
});
const layout = { root: path.join(home, 'local-llm'), launch: path.join(home, 'local-llm', 'state', 'launch') };
const base = `http://127.0.0.1:${port}`;

// ----- processes ---------------------------------------------------------------------------------------------------------------

const PS = '/bin/ps';
const psEnv = { PATH: '/usr/bin:/bin', LC_ALL: 'C', LANG: 'C', TZ: 'UTC' };
function ps(psArgs) {
    try { return execFileSync(PS, psArgs, { encoding: 'utf8', env: psEnv, timeout: 15_000, stdio: ['ignore', 'pipe', 'ignore'] }).trim(); } catch (error) { return typeof error?.stdout === 'string' ? error.stdout.trim() : ''; }
}
/** The full command line of `pid` (scrubbed for output; its key is remembered), or null when there is no such process. */
function psCommand(pid) {
    const text = ps(['-ww', '-o', 'command=', '-p', String(pid)]);
    if (!text) return null;
    rememberKeyIn(text);
    return scrub(text);
}
const psStart = (pid) => ps(['-o', 'lstart=', '-p', String(pid)]) || null;
const psRssBytes = (pid) => { const kib = Number(ps(['-o', 'rss=', '-p', String(pid)])); return Number.isFinite(kib) && kib > 0 ? kib * 1024 : null; };
/** Processes (not this one) started with `--host <a socket of this run's folder>`: the runners of this run. */
function runnersOfThisRun() {
    const found = [];
    for (const line of ps(['-ww', '-Ao', 'pid=,command=']).split('\n')) {
        const match = /^\s*(\d+)\s+(.*)$/.exec(line);
        if (!match || Number(match[1]) === process.pid) continue;
        if (match[2].includes(` --host ${socketDir}${path.sep}`)) { rememberKeyIn(match[2]); found.push({ pid: Number(match[1]), command: scrub(match[2]) }); }
    }
    return found;
}
const started = []; // { pid, start, what }: processes this script started
const same = (entry) => psStart(entry.pid) !== null && psStart(entry.pid) === entry.start;

/** The CLI, as a child process: resolves { code, signal, stdout, stderr } (scrubbed, whole). Needs a subcommand and a session environment. */
function pw(commandArgs, { env = activeEnv, waitMs = timeoutMs } = {}) {
    if (!Array.isArray(commandArgs) || commandArgs.length === 0) throw new Failure('refusing to run pworker without a subcommand: a bare pworker would start a proxy in ~/.pworker');
    if (!env || !env.PWORKER_HOME || !env.PWORKER_CONFIG) throw new Failure('refusing to run pworker without PWORKER_HOME and PWORKER_CONFIG');
    return new Promise((resolve) => {
        const child = spawn(process.execPath, [BIN, ...commandArgs], { env, stdio: ['pipe', 'pipe', 'pipe'] });
        let stdout = '', stderr = '';
        child.stdout.on('data', (chunk) => { stdout += chunk; });
        child.stderr.on('data', (chunk) => { stderr += chunk; });
        const timer = setTimeout(() => child.kill('SIGKILL'), waitMs);
        child.on('close', (code, signal) => {
            clearTimeout(timer);
            const result = { code, signal, stdout: scrub(stdout), stderr: scrub(stderr) };
            logCommand(`$ pworker ${commandArgs.join(' ')}  -> exit ${code ?? signal}\n${result.stdout}${result.stderr ? `\n[stderr] ${result.stderr}` : ''}`);
            resolve(result);
        });
        child.stdin.end('');
    });
}
const said = (result) => `${result.stdout}\n${result.stderr}`.trim();

async function waitFor(what, test, { waitMs = 60_000, everyMs = 250 } = {}) {
    const deadline = Date.now() + waitMs;
    for (;;) {
        const value = await test();
        if (value) return value;
        if (Date.now() > deadline) throw new Failure(`Timed out after ${Math.round(waitMs / 1000)} s waiting for ${what}`);
        await sleep(everyMs);
    }
}

async function http(method, pathname, body, { waitMs = 60_000 } = {}) {
    const res = await fetch(`${base}${pathname}`, {
        method, headers: body === undefined ? {} : { 'content-type': 'application/json', 'x-pworker-purpose': 'pworker:caps-live' },
        body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(waitMs),
    });
    const text = scrub(await res.text());
    let json = null;
    try { json = JSON.parse(text); } catch { /* not JSON */ }
    logCommand(`${method} ${pathname}${body === undefined ? '' : ` ${JSON.stringify(body)}`} -> ${res.status} ${text.slice(0, 600)}`);
    return { status: res.status, json, text };
}

// ----- sessions: one serve process per caps config ---------------------------------------------------------------------------

let activeEnv = null;
let serveNow = null; // { child, pid, start, exited, label, model }
const readLaunchRecords = () => {
    let names = [];
    try { names = fs.readdirSync(layout.launch); } catch { return []; }
    return names.filter((name) => /^[0-9a-f]{8}\.json$/.test(name)).map((name) => JSON.parse(fs.readFileSync(path.join(layout.launch, name), 'utf8')));
};
const socketsLeft = () => { try { return fs.readdirSync(socketDir).filter((name) => name.endsWith('.sock')); } catch { return []; } };
const serverPid = () => { try { return JSON.parse(fs.readFileSync(path.join(home, 'server.json'), 'utf8')).pid; } catch { return null; } };
async function health() {
    try {
        const res = await fetch(`${base}/health`, { signal: AbortSignal.timeout(2000) });
        return res.ok;
    } catch { return false; }
}

/** Writes the project configuration of a session: the port, the short socket folder and, when given, `localLlm.caps`. */
function writeConfig(label, caps) {
    const file = path.join(outDir, 'config', `${label}.json`);
    const config = { server: { host: '127.0.0.1', port }, localLlm: { socketDir, ...(caps === undefined ? {} : { caps }) } };
    fs.writeFileSync(file, `${JSON.stringify(config, null, 2)}\n`, { mode: 0o600 });
    return file;
}
function sessionEnv(configFile) {
    const env = { ...process.env, PWORKER_HOME: home, PWORKER_CONFIG: configFile, PWORKER_PORT: String(port), PWORKER_AUTOSTART: '0' };
    for (const name of ['NODE_TEST_CONTEXT', 'PWORKER_URL', 'PWORKER_TOKEN', 'PWORKER_LOCAL_LLM_CRASH_AT']) delete env[name];
    return env;
}

async function startServe(label, caps) {
    const configFile = writeConfig(label, caps);
    activeEnv = sessionEnv(configFile);
    const log = fs.createWriteStream(path.join(outDir, `serve-${label}.log`));
    const child = spawn(process.execPath, [BIN, 'serve'], { env: activeEnv, stdio: ['ignore', 'pipe', 'pipe'] });
    for (const stream of [child.stdout, child.stderr]) stream.on('data', (chunk) => log.write(scrub(chunk.toString('utf8'))));
    const exited = new Promise((resolve) => child.on('exit', (code, signal) => { log.end(); resolve({ code, signal }); }));
    const entry = { child, pid: child.pid, start: null, exited, label, model: null, caps, configFile };
    // The start time is read as soon as the process exists; a process that is gone at once has none and is not signalled later.
    for (let attempt = 0; attempt < 20 && entry.start === null; attempt += 1) { entry.start = psStart(child.pid); if (entry.start === null) await sleep(50); }
    started.push({ pid: entry.pid, start: entry.start, what: `serve ${label}` });
    serveNow = entry;
    await waitFor(`the proxy of pid ${entry.pid} to answer`, async () => (await health()) && serverPid() === entry.pid, { waitMs: 120_000, everyMs: 300 });
    return entry;
}

/** Stops the model (if any) and the serve of the session, and checks that nothing of this run's home is left from it. Returns the problems. */
async function stopServe() {
    const problems = [];
    const entry = serveNow;
    if (!entry) return problems;
    try {
        if (entry.model) {
            const result = await pw(['models', 'stop'], { waitMs: 120_000 });
            if (result.code !== 0) problems.push(`models stop exit ${result.code ?? result.signal}: ${ev(said(result), 300)}`);
            try { await waitFor('the runner to exit', () => psStart(entry.model.pid) === null || psStart(entry.model.pid) !== entry.model.start, { waitMs: 60_000 }); } catch (error) { problems.push(error.message); }
            entry.model = null;
        }
        const result = await pw(['stop'], { waitMs: 90_000 });
        if (result.code !== 0) problems.push(`pworker stop exit ${result.code ?? result.signal}: ${ev(said(result), 300)}`);
        await Promise.race([entry.exited, sleep(60_000)]);
    } catch (error) {
        problems.push(error.message);
    }
    for (const item of started.filter((candidate) => candidate.pid === entry.pid && same(candidate))) {
        problems.push(`serve ${entry.label} (pid ${item.pid}) did not exit after pworker stop; sending SIGTERM`);
        try { process.kill(item.pid, 'SIGTERM'); } catch { /* gone */ }
        await sleep(3000);
        if (same(item)) { try { process.kill(item.pid, 'SIGKILL'); } catch { /* gone */ } }
    }
    await sleep(500);
    // A `pworker models ...` verb starts a proxy for its home when none runs, unless PWORKER_AUTOSTART=0, which the suite sets
    // (bin/pworker.mjs): this stops one that appeared after the serve above ended, as a safeguard. It is recorded in this home's server.json.
    const recorded = serverPid();
    if (recorded && recorded !== entry.pid && psStart(recorded) !== null && /pworker\.mjs serve/.test(psCommand(recorded) ?? '')) {
        problems.push(`a proxy (pid ${recorded}) was started for this home by a CLI verb after serve ${entry.label}; stopping it`);
        started.push({ pid: recorded, start: psStart(recorded), what: 'proxy started by a CLI verb' });
        await pw(['stop'], { waitMs: 90_000 });
        await sleep(1000);
    }
    const left = runnersOfThisRun();
    if (left.length) problems.push(`processes of the home are left after serve ${entry.label}: ${left.map((item) => item.pid).join(', ')}`);
    if (readLaunchRecords().length) problems.push('a launch record is left');
    if (socketsLeft().length) problems.push('a socket file is left');
    serveNow = null;
    return problems;
}

/** Runs `body(session)` on a serve with `caps`; `label` names its config and log. Always stops the serve (a problem there is recorded). */
async function withServe(label, caps, body) {
    const entry = await startServe(label, caps);
    try {
        return await body(entry);
    } finally {
        const problems = await stopServe();
        if (problems.length) {
            record.steps.push({ id: `cleanup:${label}`, status: 'fail', checks: [], error: problems.join('; ') });
            saveRecord();
        }
    }
}

/** A session for cases: skipped when none of `ids` is selected; a serve that cannot start fails them all. */
async function session(label, ids, caps, body) {
    const wanted = ids.filter(selected);
    if (wanted.length === 0) return;
    try {
        await withServe(label, caps, body);
    } catch (error) {
        const have = new Set(record.cases.map((item) => item.id));
        for (const id of wanted) if (!have.has(id)) recordCaseFailure(id, error);
        if (!(error instanceof Failure || error instanceof Inconclusive)) record.steps.push({ id: `unexpected:${label}`, status: 'fail', checks: [], error: scrub(error?.stack ?? error) });
        saveRecord();
    }
}

// ----- the model ---------------------------------------------------------------------------------------------------------------

let settleLeftMs = settleMs; // the wait for the machine to admit the model, shared by the whole run

const runFlags = (params = {}) => [
    ...(params.ctx !== undefined ? ['--ctx', String(params.ctx)] : []),
    ...(params.parallel !== undefined ? ['--parallel', String(params.parallel)] : []),
    ...Object.entries(params.param ?? {}).flatMap(([key, value]) => ['--param', `${key}=${value}`]),
];
const previewCli = async (params = {}, json = true) => pw(['models', 'preview', MODEL, '--runner', RUNNER, ...(json ? ['--json'] : []), ...runFlags(params)]);
async function previewHttp(params) {
    return http('POST', '/v1/local-llm/preview', { modelId: MODEL, runnerId: RUNNER, ...(params ? { params } : {}) });
}
const admissionOf = (response) => response?.json?.preview?.admission ?? null;

/**
 * For a refusal for now (`insufficient-now`): a description when the cap, not the machine, is what limits the request (the figure the reason
 * reports is clearly below the physical one of the overview, as L7 checks), else null.
 */
async function capLimits(admission) {
    if (admission?.status !== 'insufficient-now') return null;
    const { cause, reported, physical, kind } = insufficientNowCause(admission, (await overviewJson())?.hardware);
    return cause === 'cap' ? `the reason reports ${bytes(reported)} ${kind === 'gpu' ? 'of GPU memory free' : 'available'}, below the physical ${bytes(physical)}: ${admission.reason}` : null;
}

/** Waits until a Run of `params` is admitted now (the machine can be busy after a boot); `incompatible` is a failure, a busy machine is inconclusive. */
async function waitAdmissible(params = {}) {
    const deadline = Date.now() + (settleLeftMs > 0 ? settleLeftMs : 90_000);
    const t0 = Date.now();
    for (;;) {
        const result = await previewCli(params);
        let admission = null;
        try { admission = JSON.parse(result.stdout).preview.admission; } catch { throw new Failure(`preview did not answer JSON: ${ev(said(result), 400)}`); }
        if (admission.status === 'ok') { settleLeftMs = Math.max(0, settleLeftMs - (Date.now() - t0)); return admission; }
        if (admission.status === 'incompatible') throw new Failure(`the model is incompatible under this config: ${admission.reason}`);
        // A refusal for now whose figure is clearly below what the machine physically has is the cap's room: no wait changes it.
        const limited = await capLimits(admission);
        if (limited) throw new Failure(`the cap, not the machine, limits the request (${admission.status}): ${limited}`);
        if (Date.now() > deadline) {
            settleLeftMs = 0;
            throw new Inconclusive(`the machine did not admit the model within the wait (${admission.status}: ${admission.reason}); a busy machine, not a caps result`);
        }
        await sleep(10_000);
    }
}

// What the engine's own memory guard says when it stops a runner for the machine's memory (metalGuard.mjs, deployments.mjs). A stop for
// `localLlm.caps.ram` is not in this list: the caps backstop is a product result, this is the state of the machine.
const MACHINE_GUARD_STOP = /stopped: (macOS memory pressure|host memory|memory pressure|memory telemetry|host memory cannot)/;

async function waitReady() {
    const deadline = Date.now() + timeoutMs;
    let idle = 0;
    for (;;) {
        const result = await pw(['models', 'status'], { waitMs: 60_000 });
        const text = result.stdout.trim();
        if (/^ready \S+ on \S+/m.test(text)) return text;
        if (MACHINE_GUARD_STOP.test(text)) throw new Inconclusive(`the engine's memory guard stopped the runner for the machine's memory (a busy machine, not a caps result): ${ev(text, 400)}`);
        if (/^error /m.test(text) || /^paused /m.test(text)) throw new Failure(`the model did not start: ${ev(text, 500)}`);
        idle = /^phase idle/m.test(text) ? idle + 1 : 0;
        if (idle >= 4) throw new Failure(`the accepted run vanished (status says idle): ${ev(text, 300)}`);
        if (Date.now() > deadline) throw new Failure(`the model was not ready after ${Math.round(timeoutMs / 60_000)} min: ${ev(text, 300)}`);
        await sleep(2000);
    }
}

/** The one launch record of the running model, its runner process identity and scrubbed command line. */
function runnerOf() {
    const records = readLaunchRecords().filter((item) => item.phase === 'spawned');
    need(records.length === 1, `expected one spawned launch record, found ${records.length} (${readLaunchRecords().length} records)`);
    const launch = records[0];
    need(Number.isInteger(launch.pid), `the launch record has no pid: ${JSON.stringify({ ...launch, socketPath: undefined })}`);
    const command = psCommand(launch.pid);
    need(command !== null, `no process ${launch.pid} for the launch record`);
    need(command.includes(`--host ${launch.socketPath}`), `process ${launch.pid} does not name the record's socket as --host`);
    return { pid: launch.pid, start: psStart(launch.pid), socketPath: launch.socketPath, record: { phase: launch.phase, pid: launch.pid, startId: launch.startId, startTime: launch.startTime }, argv: command, parsed: parseRunnerArgv(command) };
}

/** Runs the model with `params` under the session's caps, waits for ready, and reads the runner. Needs the machine to admit it. */
async function startModel(s, params = {}) {
    await waitAdmissible(params);
    const result = await pw(['models', 'run', MODEL, '--runner', RUNNER, '--detach', ...runFlags(params)]);
    need(result.code === 0, `models run exit ${result.code ?? result.signal}: ${ev(said(result), 600)}`);
    const statusText = await waitReady();
    const runner = runnerOf();
    started.push({ pid: runner.pid, start: runner.start, what: `runner of ${s.label}` });
    s.model = { ...runner, runText: result.stdout.trim(), statusText };
    return s.model;
}
async function ensureModel(s) { return s.model ?? startModel(s); }

async function stopModel(s) {
    const pid = s.model?.pid, start = s.model?.start;
    const result = await pw(['models', 'stop'], { waitMs: 120_000 });
    need(result.code === 0, `models stop exit ${result.code ?? result.signal}: ${ev(said(result), 300)}`);
    await waitFor('the runner to exit', () => psStart(pid) === null || psStart(pid) !== start, { waitMs: 60_000 });
    await waitFor('the launch record and the socket to be removed', () => readLaunchRecords().length === 0 && socketsLeft().length === 0, { waitMs: 30_000 });
    s.model = null;
    return { pid, output: result.stdout.trim() };
}

async function chatOnce() {
    const res = await fetch(`${base}/u/local-llm/v1/chat/completions`, {
        method: 'POST', headers: { 'content-type': 'application/json', 'x-pworker-purpose': 'pworker:caps-live' },
        body: JSON.stringify({ model: MODEL, max_tokens: 24, temperature: 0, messages: [{ role: 'user', content: 'Say hello in three words.' }] }),
        signal: AbortSignal.timeout(300_000),
    });
    const text = scrub(await res.text());
    let content = null;
    try { content = JSON.parse(text)?.choices?.[0]?.message?.content ?? null; } catch { /* not JSON */ }
    logCommand(`POST /u/local-llm/v1/chat/completions -> ${res.status} ${text.slice(0, 400)}`);
    return { status: res.status, content, text };
}

const overviewJson = async () => (await http('GET', '/v1/local-llm')).json;
const statusText = async () => (await pw(['models', 'status'])).stdout.trim();

// ----- derived values (filled by the baseline) ------------------------------------------------------------------------------------

const B = {
    profile: null, hardware: null, layers: null, ctx: null, parallel: null, threads: null, threadsFrom: null,
    needDefault: null, needHalved: null, gpuDefault: null, gpuByCtx: {}, gpuByLayers: {}, ramByLayers: {},
    rssBytes: null, // the resident size of the runner with no caps (L11), which L7's ram cap must leave room for
};
const catalogLayers = () => {
    try {
        const catalog = JSON.parse(fs.readFileSync(path.join(ROOT, 'lib', 'local-llm-host', 'catalog', 'models.json'), 'utf8'));
        const entry = catalog.models.find((model) => model.id === MODEL);
        return Number.isInteger(entry?.memory?.layers) ? entry.memory.layers : null;
    } catch { return null; }
};
/** A threads cap below the profile default that tells the two apart: 2 where the default is above 2. Null when the default is 1. */
const threadsCap = () => (Number.isInteger(B.threads) && B.threads >= 2 ? Math.min(2, B.threads - 1) : null);
const bytes = (value) => (Number.isFinite(value) ? `${value} bytes (${(value / MIB).toFixed(0)} MiB)` : String(value));
const ctxCap = () => (B.ctx > 2048 ? 2048 : B.ctx > 512 ? halvedContext(B.ctx) : null);

async function estimateWith(params) {
    const response = await previewHttp(params);
    const admission = admissionOf(response);
    need(response.status === 200 && admission?.estimate, `preview ${JSON.stringify(params ?? {})} answered ${response.status}: ${ev(response.text, 300)}`);
    return admission;
}

/** The previews of the baseline serve (no caps) that every derived cap value comes from. */
async function gatherEstimates() {
    const overview = await overviewJson();
    need(overview?.available === true, `the engine is not available: ${ev(JSON.stringify(overview?.refusal ?? overview), 300)}`);
    B.profile = overview.profile;
    B.hardware = overview.hardware;
    need(B.profile === expectedProfile || (expectedProfile === 'dedicated' && B.profile === 'unified'), `the engine runs the profile ${B.profile}, the suite expected ${expectedProfile} from the machine`);
    B.layers = catalogLayers();
    const response = await previewHttp(undefined);
    const first = admissionOf(response);
    need(response.status === 200 && first?.estimate, `preview answered ${response.status}: ${ev(response.text, 300)}`);
    const preview = response.json.preview;
    B.ctx = preview.params.ctxSize;
    B.parallel = preview.params.parallel;
    B.needDefault = ramNeedOf(first.estimate);
    B.gpuDefault = first.estimate.gpuBytes ?? null;
    const halved = halvedContext(B.ctx);
    B.needHalved = ramNeedOf((await estimateWith({ ctxSize: halved })).estimate);
    if (B.profile === 'dedicated') {
        const ctxUsed = Math.min(B.ctx, FIT_CONTEXT_FLOOR);
        for (const ctx of new Set([B.ctx, halved, ctxUsed])) B.gpuByCtx[ctx] = (await estimateWith({ ctxSize: ctx })).estimate.gpuBytes;
        if (B.layers) {
            for (let n = 1; n <= B.layers; n += 1) {
                const admission = await estimateWith({ ctxSize: ctxUsed, nGpuLayers: n });
                B.gpuByLayers[n] = admission.estimate.gpuBytes;
                B.ramByLayers[n] = admission.estimate.ramBytes;
            }
        }
    }
    return preview;
}

// ----- the cases ---------------------------------------------------------------------------------------------------------------

/** L11 and the baseline: no caps at all. It also downloads the weights (the first run) and reads the profile's default argv. */
async function baseline() {
    await withServe('baseline', undefined, async (s) => {
        let install = null;
        await runCase('setup', `install the runner ${RUNNER}`, async () => {
            install = await pw(['models', 'runner', 'install', RUNNER], { waitMs: timeoutMs });
            need(install.code === 0, `exit ${install.code ?? install.signal}: ${ev(said(install), 500)}`);
            check('runner installed', true, install.stdout.split('\n').slice(-2).join('\n'));
        }, { hidden: true });
        await runCase('setup', 'previews of the baseline: the estimates every cap value derives from', async () => {
            await gatherEstimates();
            record.derived = { profile: B.profile, runner: RUNNER, model: MODEL, layers: B.layers, defaultCtx: B.ctx, defaultParallel: B.parallel, needDefault: B.needDefault, needHalved: B.needHalved, gpuDefault: B.gpuDefault, hardware: B.hardware };
            check('estimates read', true, JSON.stringify(record.derived));
        }, { hidden: true });
        if (!B.profile) throw new Failure('the baseline could not read the estimates');
        // The first run of the model downloads the weights and gives the profile's default argv: every case but L3 and L4 needs it.
        if (CASE_IDS.some((id) => selected(id) && !['L3', 'L4'].includes(id))) {
            await runCase('L11', 'identity (G5): with no caps nothing is capped, shown or fitted', async () => {
                const admission = await waitAdmissible();
                check('preview is admitted (status ok)', admission.status === 'ok', `status ${admission.status}; need ${bytes(ramNeedOf(admission.estimate))}`);
                check('preview has no fit', admission.fit === undefined, `fit: ${JSON.stringify(admission.fit)}; warnings: ${JSON.stringify(admission.warnings)}`);
                const overview = await overviewJson();
                check('the overview has no limits.budget', overview?.limits?.budget === undefined, `limits: ${JSON.stringify(overview?.limits)}`);
                const model = await startModel(s);
                B.rssBytes = psRssBytes(model.pid);
                B.threads = model.parsed.threads;
                B.threadsFrom = 'the no-caps runner argv';
                const wantThreads = darwin ? B.hardware?.cores : null;
                check('argv ctx is the default (no capped value)', model.parsed.ctx === B.ctx, `argv: ${ev(model.argv, 1200)}; preview default ctxSize ${B.ctx}`);
                check('argv parallel is the default', model.parsed.parallel === B.parallel, `-np ${model.parsed.parallel}; default ${B.parallel}`);
                if (wantThreads) check('argv --threads is the profile default (performance cores)', model.parsed.threads === wantThreads, `--threads ${model.parsed.threads}; hardware.cores ${wantThreads}`);
                else check('argv has a --threads value (the default, read from this argv)', Number.isInteger(model.parsed.threads) && model.parsed.threads >= 1, `--threads ${model.parsed.threads}`);
                const text = await statusText();
                check('models status prints no caps line', !/caps in force/.test(text), text);
                check('models status says ready', /^ready \S+ on \S+/m.test(text), text);
                // The resident size is read again after the checks (it only grows as pages are touched): the larger reading is kept.
                const settled = psRssBytes(model.pid);
                if (Number.isFinite(settled)) B.rssBytes = Math.max(B.rssBytes ?? 0, settled);
                record.derived = { ...record.derived, runnerRssBytes: B.rssBytes };
                const stopped = await stopModel(s);
                check('stop: the runner is gone and no record or socket is left', true, `pid ${stopped.pid}; ${stopped.output}`);
            }, { hidden: !selected('L11') });
        }
        if (B.threads === null) {
            B.threads = darwin ? B.hardware?.cores ?? null : null;
            B.threadsFrom = B.threads ? 'hardware.cores (the baseline run did not give an argv)' : null;
        }
    });
}

async function caseL1() {
    const tcap = threadsCap();
    const caps = { ram: '8GiB', ...(tcap ? { threads: tcap } : {}) };
    await session('l1', ['L1'], caps, async (s) => {
        await runCase('L1', `ram + threads: ${JSON.stringify(caps)}`, async () => {
            const model = await startModel(s);
            if (tcap) {
                check(`argv --threads equals the cap (${tcap}), below the default (${B.threads})`, model.parsed.threads === tcap && tcap < B.threads, `argv: ${ev(model.argv, 1200)}`);
            } else skipCheck('threads cap', `the profile default thread count is ${B.threads}: no cap below it`, { core: true });
            const text = await statusText();
            check('models status prints the caps line', new RegExp(`caps in force \\(localLlm\\.caps\\):.*ram 8 GiB${tcap ? `.*threads ${tcap}` : ''}.*refuse mode`).test(text), text);
            const chat = await chatOnce();
            check('one chat returns 200 with content', chat.status === 200 && typeof chat.content === 'string' && chat.content.length > 0, `HTTP ${chat.status}; content: ${JSON.stringify(chat.content)}`);
            const stopped = await stopModel(s);
            check('stop: the runner is gone, no launch record or socket left', psStart(stopped.pid) === null, `runner pid ${stopped.pid}: ${stopped.output}`);
        });
    });
}

async function caseL2() {
    const cc = ctxCap();
    const tcap = threadsCap();
    if (cc === null) {
        await skipCase('L2', 'count caps', `the default context ${B.ctx} is too small to cap`);
        return;
    }
    const over = cc * 2;
    const caps = { maxContext: cc, maxParallel: 1, ...(tcap ? { threads: tcap } : {}) };
    await session('l2', ['L2'], caps, async (s) => {
        await runCase('L2', `count caps: ${JSON.stringify(caps)}`, async () => {
            // (b) an explicit value above a cap is refused before anything starts
            const ctxRun = await pw(['models', 'run', MODEL, '--runner', RUNNER, '--detach', '--ctx', String(over)]);
            check(`(b) run --ctx ${over} is refused naming localLlm.caps.maxContext`, ctxRun.code !== 0 && /localLlm\.caps\.maxContext/.test(said(ctxRun)), `exit ${ctxRun.code}: ${said(ctxRun)}`);
            const parRun = await pw(['models', 'run', MODEL, '--runner', RUNNER, '--detach', '--parallel', '2']);
            check('(b) run --parallel 2 is refused naming localLlm.caps.maxParallel', parRun.code !== 0 && /localLlm\.caps\.maxParallel/.test(said(parRun)), `exit ${parRun.code}: ${said(parRun)}`);
            if (tcap) {
                const thrRun = await pw(['models', 'run', MODEL, '--runner', RUNNER, '--detach', '--param', `threads=${tcap + 1}`]);
                check(`(c) run --param threads=${tcap + 1} is refused naming localLlm.caps.threads`, thrRun.code !== 0 && /localLlm\.caps\.threads/.test(said(thrRun)), `exit ${thrRun.code}: ${said(thrRun)}`);
                const thrPreview = await pw(['models', 'preview', MODEL, '--runner', RUNNER, '--param', `threads=${tcap + 1}`]);
                check('(c) preview with that threads value is refused naming localLlm.caps.threads', thrPreview.code !== 0 && /localLlm\.caps\.threads/.test(said(thrPreview)), `exit ${thrPreview.code}: ${said(thrPreview)}`);
            } else skipCheck('(c) explicit threads above a threads cap', `the profile default is ${B.threads} threads: no threads cap below it`);
            // (d) HTTP
            const previewBad = await previewHttp({ ctxSize: over });
            check(`(d) HTTP preview with ctxSize ${over} gives 400 invalid_params`, previewBad.status === 400 && previewBad.json?.error?.type === 'invalid_params' && /localLlm\.caps\.maxContext/.test(previewBad.text), `HTTP ${previewBad.status}: ${previewBad.text}`);
            const runBad = await http('POST', '/v1/local-llm/run', { modelId: MODEL, runnerId: RUNNER, params: { ctxSize: over } });
            check('(d) HTTP run with that ctxSize gives 400 invalid_params', runBad.status === 400 && runBad.json?.error?.type === 'invalid_params', `HTTP ${runBad.status}: ${runBad.text}`);
            check('nothing was started by the refusals', readLaunchRecords().length === 0 && runnersOfThisRun().length === 0, `launch records ${readLaunchRecords().length}; processes ${runnersOfThisRun().length}`);
            // (a) the default is clamped
            const model = await startModel(s);
            check(`(a) the default ctx (${B.ctx}) is clamped: argv ctx is ${cc}`, model.parsed.ctx === cc, `argv: ${ev(model.argv, 1200)}`);
            check('(a) -np is 1', model.parsed.parallel === 1, `-np ${model.parsed.parallel}`);
            if (tcap) check(`(a) argv --threads is ${tcap}`, model.parsed.threads === tcap, `--threads ${model.parsed.threads}`);
            // (e) a refused run with replace leaves the running runner as it was
            const before = { pid: model.pid, start: psStart(model.pid), argv: psCommand(model.pid), records: JSON.stringify(readLaunchRecords()) };
            const replaceHttp = await http('POST', '/v1/local-llm/run', { modelId: MODEL, runnerId: RUNNER, params: { ctxSize: over }, replace: true });
            check('(e) HTTP run with replace:true and an over-cap ctxSize gives 400 invalid_params', replaceHttp.status === 400 && replaceHttp.json?.error?.type === 'invalid_params', `HTTP ${replaceHttp.status}: ${replaceHttp.text}`);
            const replaceCli = await pw(['models', 'run', MODEL, '--runner', RUNNER, '--detach', '--replace', '--ctx', String(over)]);
            check('(e) CLI run --replace --ctx over the cap is refused naming the cap', replaceCli.code !== 0 && /localLlm\.caps\.maxContext/.test(said(replaceCli)), `exit ${replaceCli.code}: ${said(replaceCli)}`);
            await sleep(1000);
            const after = { pid: readLaunchRecords()[0]?.pid, start: psStart(model.pid), argv: psCommand(model.pid), records: JSON.stringify(readLaunchRecords()) };
            check('(e) the runner pid, start time and argv are unchanged', after.pid === before.pid && after.start === before.start && after.argv === before.argv, `before pid ${before.pid} ${before.start}\nafter pid ${after.pid} ${after.start}\nargv equal: ${after.argv === before.argv}`);
            check('(e) the launch record is unchanged', after.records === before.records, `records equal: ${after.records === before.records}`);
            const text = await statusText();
            check('(e) models status still says ready', /^ready \S+ on \S+/m.test(text), text);
        });
    });
}

async function caseL3() {
    if (!['metal', 'cpu', 'dedicated'].includes(B.profile)) { await skipCase('L3', 'ram too small', `the ${B.profile} profile sizes against a different need figure; not covered by this suite`); return; }
    const cap = Math.max(MIB, Math.floor(B.needDefault / 2));
    await session('l3', ['L3'], { ram: cap }, async () => {
        await runCase('L3', `ram too small: ram ${bytes(cap)} against a need of ${bytes(B.needDefault)}`, async () => {
            const preview = await previewHttp(undefined);
            const admission = admissionOf(preview);
            check('HTTP preview status is incompatible (not merely refused)', preview.status === 200 && admission?.status === 'incompatible', `HTTP ${preview.status}; status ${admission?.status}; reason: ${admission?.reason}`);
            const cli = await previewCli({}, false);
            check('CLI preview says incompatible and prints the caps line', new RegExp(`^${MODEL.replace(/[.]/g, '\\.')}: incompatible`, 'm').test(cli.stdout) && /caps in force \(localLlm\.caps\): ram/.test(cli.stdout), cli.stdout);
            const runCli = await pw(['models', 'run', MODEL, '--runner', RUNNER, '--detach']);
            check('CLI run is refused (exit 1) with the admission reason and the caps line', runCli.code !== 0 && /Needs about/.test(said(runCli)) && /caps in force \(localLlm\.caps\): ram/.test(runCli.stderr), `exit ${runCli.code}: ${said(runCli)}`);
            const runHttp = await http('POST', '/v1/local-llm/run', { modelId: MODEL, runnerId: RUNNER });
            check('HTTP run is refused as an admission refusal (422 admission_incompatible)', runHttp.status === 422 && /^admission_/.test(runHttp.json?.error?.type ?? ''), `HTTP ${runHttp.status}: ${runHttp.text}`);
            await sleep(500);
            check('no launch record was created and no runner started', readLaunchRecords().length === 0 && runnersOfThisRun().length === 0 && socketsLeft().length === 0, `launch records ${readLaunchRecords().length}; processes ${runnersOfThisRun().length}; sockets ${socketsLeft().length}`);
            const text = await statusText();
            check('models status shows no deployment', /^phase idle/m.test(text), text);
        });
    });
}

async function caseL4() {
    if (!selected('L4')) return;
    const bad = [
        { label: 'l4-ram0', caps: { ram: 0 }, key: 'ram' },
        { label: 'l4-lots', caps: { gpuMemory: 'lots' }, key: 'gpuMemory' },
        { label: 'l4-unknown', caps: { ram: '8GiB', bogusCap: 3 }, key: 'bogusCap' },
    ];
    await runCase('L4', 'invalid caps: status says invalid_caps naming the key, preview gives 503, serve stays up', async () => {
        const probe = async (label, caps, key, mustName = [`localLlm.caps.${key}`]) => {
            await withServe(label, caps, async (s) => {
                const text = await statusText();
                check(`${label} ${JSON.stringify(caps)}: models status reports invalid_caps naming ${key}`, /invalid_caps/.test(text) && mustName.every((name) => text.includes(name)), text);
                const preview = await previewHttp(undefined);
                check(`${label}: HTTP preview gives 503 invalid_caps`, preview.status === 503 && preview.json?.error?.type === 'invalid_caps', `HTTP ${preview.status}: ${preview.text}`);
                const live = await http('GET', '/v1/local-llm/status');
                check(`${label}: GET status answers 200 with available:false`, live.status === 200 && live.json?.available === false && live.json?.refusal?.code === 'invalid_caps', `HTTP ${live.status}: ${ev(live.text, 300)}`);
                check(`${label}: serve stays up`, (await health()) && same({ pid: s.pid, start: s.start }) && serverPid() === s.pid, `serve pid ${s.pid} alive, /health answers`);
                check(`${label}: nothing was started`, readLaunchRecords().length === 0 && runnersOfThisRun().length === 0, `launch records ${readLaunchRecords().length}`);
            });
        };
        for (const item of bad) await probe(item.label, item.caps, item.key);
        if (darwin) await probe('l4-mac-gpu', { gpuMemory: '4GiB' }, 'gpuMemory', ['localLlm.caps.gpuMemory', 'localLlm.caps.ram']);
        else skipCheck('gpuMemory "4GiB" is refused naming ram', 'macOS only: on Linux gpuMemory is a valid cap for an NVIDIA GPU');
    });
}

async function caseL5() {
    let plan = null;
    let caps = null;
    let reason = null;
    if (B.profile === 'metal' || B.profile === 'cpu') {
        plan = planRamFit({ ctx: B.ctx, needDefault: B.needDefault, needHalved: B.needHalved });
        if (!plan) reason = `the need at ctx ${B.ctx} (${B.needDefault}) is not above the need at ctx ${halvedContext(B.ctx)} (${B.needHalved})`;
        else caps = { ram: plan.cap, mode: 'fit' };
    } else if (B.profile === 'dedicated') {
        const found = planGpuFitContext({ ctx: B.ctx, gpuAt: (ctx) => B.gpuByCtx[ctx] ?? B.gpuByCtx[halvedContext(B.ctx)] });
        if (found.skip) reason = found.skip;
        else { plan = found; caps = { gpuMemory: found.cap, mode: 'fit' }; }
    } else reason = `the ${B.profile} profile is not covered by this suite`;
    if (!plan) { await skipCase('L5', 'fit context', reason); return; }
    await session('l5', ['L5'], caps, async (s) => {
        await runCase('L5', `fit context: ${JSON.stringify(caps)} (default ctx ${plan.from}, expected ${plan.to})`, async () => {
            const admission = admissionOf(await previewHttp(undefined));
            const lowered = admission?.fit?.lowered ?? [];
            // insufficient-now is accepted only when it is the machine's state now: when the cap's room is what is short, the cap is the finding.
            const capShort = await capLimits(admission);
            check(`preview shows fit lowered ctxSize ${plan.from} -> ${plan.to} (not incompatible; insufficient-now only when it is the machine's state now)`, (admission?.status === 'ok' || (admission?.status === 'insufficient-now' && !capShort)) && lowered.length === 1 && lowered[0].param === 'ctxSize' && lowered[0].from === plan.from && lowered[0].to === plan.to, `status ${admission?.status}; fit ${JSON.stringify(admission?.fit)}; reason: ${admission?.reason}${capShort ? `; ${capShort}` : ''}`);
            const cli = await previewCli({}, false);
            check(`CLI preview prints "fit lowered: ctxSize ${plan.from} -> ${plan.to}"`, cli.stdout.includes(`fit lowered: ctxSize ${plan.from} -> ${plan.to}`), cli.stdout);
            // an explicit ctx is never lowered
            const explicit = await previewCli({ ctx: plan.from }, false);
            check(`explicit --ctx ${plan.from} is refused by preview as incompatible`, new RegExp(`^${MODEL.replace(/[.]/g, '\\.')}: incompatible`, 'm').test(explicit.stdout) && !/fit lowered/.test(explicit.stdout), explicit.stdout);
            const explicitRun = await pw(['models', 'run', MODEL, '--runner', RUNNER, '--detach', '--ctx', String(plan.from)]);
            check(`explicit run --ctx ${plan.from} is refused (an admission refusal, not a lowered launch)`, explicitRun.code !== 0 && /Needs about/.test(said(explicitRun)), `exit ${explicitRun.code}: ${said(explicitRun)}`);
            check('the refusal started nothing', readLaunchRecords().length === 0 && runnersOfThisRun().length === 0, `launch records ${readLaunchRecords().length}`);
            const model = await startModel(s);
            check(`the run launches with the fitted ctx: argv ctx is ${plan.to}`, model.parsed.ctx === plan.to, `argv: ${ev(model.argv, 1200)}`);
            const text = await statusText();
            check(`models status shows "fit lowered: ctxSize ${plan.from} -> ${plan.to}"`, text.includes(`fit lowered: ctxSize ${plan.from} -> ${plan.to}`), text);
        });
    });
}

async function caseL6() {
    if (B.profile !== 'dedicated') { await skipCase('L6', 'fit layers', `dedicated only: the ${B.profile} profile has no nGpuLayers to move`); return; }
    const ctxUsed = Math.min(B.ctx, FIT_CONTEXT_FLOOR);
    const plan = B.layers ? planGpuFitLayers({ layers: B.layers, gpuAtLayers: (n) => B.gpuByLayers[n] }) : null;
    if (!plan) {
        const why = !B.layers ? `the catalog entry of ${MODEL} has no layer count` : `no gpuMemory cap makes the layers of ${MODEL} move (GPU need at ${B.layers} layers ${bytes(B.gpuByLayers[B.layers])}, at 1 layer ${bytes(B.gpuByLayers[1])})`;
        await skipCase('L6', 'fit layers', why);
        return;
    }
    const caps = { gpuMemory: plan.cap, mode: 'fit' };
    await session('l6', ['L6'], caps, async (s) => {
        await runCase('L6', `fit layers: ${JSON.stringify(caps)}; predicted: layers ${B.layers} -> ${plan.expectLayers}, status ${plan.expectStatus}`, async () => {
            // the 9d7a4e3 split: the RAM the estimate asks for grows as fewer layers are offloaded (explicit values: fit never touches them)
            const probes = [...new Set([B.layers, Math.ceil(B.layers / 2), 1])];
            const rams = [];
            for (const n of probes) rams.push({ n, ram: (await estimateWith({ ctxSize: ctxUsed, nGpuLayers: n })).estimate.ramBytes });
            check('estimate.ramBytes grows as nGpuLayers drops', rams.every((item, index) => index === 0 || item.ram > rams[index - 1].ram), rams.map((item) => `ngl ${item.n}: ${item.ram}`).join('; '));
            const admission = admissionOf(await previewHttp(undefined));
            // A busy GPU (or RAM) makes the fitted request refused for now and fit has nothing to lower: that is the machine, not a caps result.
            if (admission?.status === 'insufficient-now' && !(await capLimits(admission))) throw new Inconclusive(`the machine refuses the request for now (a busy GPU or memory), so fit has nothing to lower: ${admission.reason}`);
            const lowered = admission?.fit?.lowered ?? [];
            const layersStep = lowered.find((item) => item.param === 'nGpuLayers');
            check(`fit lowered nGpuLayers to ${plan.expectLayers}, below the model's ${B.layers}`, Boolean(layersStep) && layersStep.to === plan.expectLayers && layersStep.to < B.layers, `status ${admission?.status}; fit ${JSON.stringify(admission?.fit)}; reason: ${admission?.reason}`);
            check(`the fitted preview is ${plan.expectStatus}`, admission?.status === plan.expectStatus, `status ${admission?.status}; ${admission?.reason}`);
            if (plan.expectStatus === 'ok') {
                const model = await startModel(s);
                check(`argv -ngl is ${plan.expectLayers}, below ${B.layers}`, model.parsed.layers === plan.expectLayers && model.parsed.layers < B.layers, `argv: ${ev(model.argv, 1200)}`);
            } else {
                skipCheck('launch with the fitted layers (argv -ngl below the layer count)', `the fitted request is ${plan.expectStatus}, so a Run refuses it: ${admission?.reason}`, { core: true });
            }
        });
    });
}

/** L7, L9 and L10 share one run of the model under a tight ram cap. */
async function caseL7() {
    if (!['L7', 'L9', 'L10'].some(selected)) return;
    if (!['metal', 'cpu', 'dedicated'].includes(B.profile)) {
        for (const id of ['L7', 'L9', 'L10']) await skipCase(id, 'needs a runner', `the ${B.profile} profile is not covered by this suite`);
        return;
    }
    // The need plus 64 MiB, and at least 1.25 times the runner's own RSS measured with no caps (L11): a runner whose steady RSS is above its
    // estimate (a CUDA runner) would otherwise be stopped by the live ram guard inside waitReady, and L7 would fail for the runner's own size.
    const { cap, room, secondRefused } = planOurUseCap({ need: B.needDefault, rssBytes: B.rssBytes });
    if (secondRefused === false) {
        for (const id of ['L7', 'L9', 'L10']) await skipCase(id, 'needs a runner under a tight ram cap', `the cap ${bytes(cap)} leaves ${bytes(room)} beside the runner (RSS ${bytes(B.rssBytes)}), enough for a second instance of the need ${bytes(B.needDefault)}: ourUse cannot be shown`);
        return;
    }
    await session('l7', ['L7', 'L9', 'L10'], { ram: cap }, async (s) => {
        await runCase('L7', `ourUse: ram ${bytes(cap)} (the need ${bytes(B.needDefault)} plus 64 MiB, at least 1.25 x the measured runner RSS ${bytes(B.rssBytes)})`, async () => {
            const model = await ensureModel(s);
            const rss = psRssBytes(model.pid);
            need(rss !== null, 'the runner has no resident size');
            // The measured RSS above is of an earlier run; the live one decides whether a second instance can still be refused because of ourUse.
            if (B.needDefault <= cap - rss) throw new Inconclusive(`the runner holds ${bytes(rss)} now, which leaves ${bytes(cap - rss)} under the cap, enough for a second instance of the need ${bytes(B.needDefault)}: the preview cannot be refused because of ourUse`);
            const overview = await overviewJson();
            const physical = overview?.hardware?.memory?.availableBytes;
            if (Number.isFinite(physical) && cap + ramAdmitMargin(B.profile) - rss >= physical) throw new Inconclusive(`physical available memory (${bytes(physical)}) is below cap less RSS (plus the policy's margin, ${bytes(cap + ramAdmitMargin(B.profile) - rss)}): the cap is not what limits it`);
            const admission = admissionOf(await previewHttp(undefined));
            check('a preview of the same model reports insufficient-now', admission?.status === 'insufficient-now', `status ${admission?.status}; ${admission?.reason}`);
            const reported = availableNowBytes(admission?.reason);
            // The dedicated policy's refusal for now reports the room under the cap plus its own 1 GiB margin (caps.mjs, `effective`).
            const used = cap + ramAdmitMargin(B.profile) - reported;
            check('available is lower than the cap by about the runner RSS (within 128 MiB)', reported !== null && Math.abs(used - rss) <= 128 * MIB, `reported available ${bytes(reported)}; cap ${bytes(cap)}; cap - reported ${bytes(used)}; runner RSS (ps) ${bytes(rss)}; physical available ${bytes(physical)}`);
            check('the cap is what limited it, not the machine (reported available is below the physical figure)', reported !== null && reported < physical, `reported ${bytes(reported)} < physical ${bytes(physical)}`);
        });
        await runCase('L9', 'RSS reader: readRunnerMemory equals ps rss within 10 %', async () => {
            const model = await ensureModel(s);
            const reader = createRunnerMemoryReader({ launchDir: layout.launch, platformName: process.platform });
            const before = psRssBytes(model.pid);
            const value = reader();
            const after = psRssBytes(model.pid);
            check('the reader returns the runner RSS within 10 % of ps rss', within(value, before, 0.1) && within(value, after, 0.1), `reader ${bytes(value)}; ps rss before ${bytes(before)}, after ${bytes(after)}; launchDir ${layout.launch}`);
        });
        await runCase('L10', 'guard backstop: real reader, ram below the live RSS, two { stop } answers, runner not killed', async () => {
            const model = await ensureModel(s);
            const rss = psRssBytes(model.pid);
            need(rss !== null, 'the runner has no resident size');
            const below = Math.max(MIB, Math.floor(rss / 2));
            const limits = defineCapLimits(parseCaps({ ram: below }), { launchDir: layout.launch, platformName: process.platform });
            const memory = { totalBytes: B.hardware?.memory?.totalBytes ?? 16 * 1024 ** 3 };
            const samples = [limits.guardSample(), limits.guardSample()];
            check('guardSample is watched and standalone, with the real reading', samples.every((sample) => sample?.watched === true && sample.standalone === true && within(sample.usedBytes, rss, 0.1)), `${JSON.stringify(samples)}; ps rss ${bytes(rss)}`);
            const breaches = samples.map((sample) => limits.guardBreach(sample, { check: 'headroom', memory }));
            check('two consecutive guardBreach results are { stop } naming localLlm.caps.ram', breaches.every((item) => typeof item?.stop === 'string' && item.stop.includes('localLlm.caps.ram')), JSON.stringify(breaches));
            const generous = defineCapLimits(parseCaps({ ram: '64GiB' }), { launchDir: layout.launch, platformName: process.platform });
            const control = generous.guardBreach(generous.guardSample(), { check: 'headroom', memory });
            check('control: with a ram above the live RSS the guard answers null', control === null, `guardBreach: ${JSON.stringify(control)}`);
            check('the runner was not killed by the check (same pid and start time, status ready)', psStart(model.pid) === model.start && /^ready \S+ on \S+/m.test(await statusText()), `pid ${model.pid} start ${psStart(model.pid)}`);
        });
    });
}

function nvidiaRunnerMemoryMiB(pid) {
    const result = run('nvidia-smi', ['--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits']);
    if (result.error || result.status !== 0) return { error: result.stderr || String(result.error) };
    for (const line of result.stdout.split('\n')) {
        const [rowPid, used] = line.split(',').map((part) => part.trim());
        if (Number(rowPid) === pid) return { mib: Number(used), line };
    }
    return { rows: result.stdout };
}

async function caseL8() {
    if (!selected('L8')) return;
    if (B.profile !== 'dedicated') { await skipCase('L8', 'gpuMemory share', `dedicated only: the ${B.profile} profile has no GPU share`); return; }
    const need0 = B.gpuDefault;
    const low = Math.max(MIB, Math.floor(need0 * 0.9));
    const high = gpuCapAdmitting(need0, 0.1);
    await runCase('L8', `gpuMemory share: below ${bytes(low)}, above ${bytes(high)} the GPU need ${bytes(need0)}`, async () => {
        await withServe('l8-low', { gpuMemory: low }, async () => {
            const admission = admissionOf(await previewHttp(undefined));
            check('below the GPU need: preview is incompatible', admission?.status === 'incompatible', `status ${admission?.status}; ${admission?.reason}`);
            const refused = await pw(['models', 'run', MODEL, '--runner', RUNNER, '--detach']);
            check('below the GPU need: run is refused', refused.code !== 0 && /Needs about/.test(said(refused)), `exit ${refused.code}: ${said(refused)}`);
            check('nothing started', readLaunchRecords().length === 0 && runnersOfThisRun().length === 0, `launch records ${readLaunchRecords().length}`);
        });
        await withServe('l8-high', { gpuMemory: high }, async (s) => {
            const model = await startModel(s);
            check('above the GPU need: the model runs', Boolean(model.pid), `argv: ${ev(model.argv, 600)}`);
            const smiRow = nvidiaRunnerMemoryMiB(model.pid);
            if (smiRow.mib === undefined) throw new Inconclusive(`nvidia-smi does not list the runner pid ${model.pid} (${smiRow.error ?? `rows: ${smiRow.rows}`}); a container's pid namespace can hide it`);
            check('nvidia-smi: the runner uses at most the cap + 10 %', smiRow.mib * MIB <= high * 1.1, `runner pid ${model.pid} uses ${smiRow.mib} MiB; cap ${bytes(high)}; limit ${bytes(Math.floor(high * 1.1))}; row: ${smiRow.line}`);
        });
    });
}

// ----- main --------------------------------------------------------------------------------------------------------------------

function environmentFacts() {
    const sha = run('git', ['-C', ROOT, 'rev-parse', 'HEAD']);
    const dirty = run('git', ['-C', ROOT, 'status', '--porcelain']);
    return {
        platform: process.platform, arch: process.arch, node: process.version, kernel: run('uname', ['-r']).stdout,
        memoryBytes: os.totalmem(), cpus: os.cpus().length, loadAverage: os.loadavg(), uptimeSeconds: os.uptime(),
        gpu: gpu.gpuPresent ? smi.stdout : null, expectedProfile, runner: RUNNER, model: MODEL,
        git: { sha: sha.status === 0 && sha.stdout ? sha.stdout : null, dirty: dirty.status === 0 ? dirty.stdout.length > 0 : null },
        home, socketDir, port, pworkerHome: pworkerBefore, only: only ? [...only] : null,
    };
}

async function mainRun() {
    await runCase('setup', 'environment and safety', async () => {
        record.environment = environmentFacts();
        const probePath = path.join(socketDir, '0123abcd.sock');
        const limit = darwin ? 103 : 107;
        check('the socket path fits the Unix socket limit', Buffer.byteLength(probePath) <= limit, `${probePath} is ${Buffer.byteLength(probePath)} bytes; the limit is ${limit}`);
        const bindable = await portIsFree(port);
        check('the port is free (bound and released) and not 18080', port !== 18080 && bindable === true, `port ${port}; bind on 127.0.0.1: ${bindable === true ? 'succeeded' : bindable}`);
        check('~/.pworker state before the run', true, JSON.stringify(pworkerBefore));
    }, { hidden: true });
    if (record.steps.some((step) => step.status === 'fail')) throw new Failure('the environment check failed');
    await baseline();
    if (!B.profile) throw new Failure('the baseline did not read the machine; no case can derive its caps');
    await caseL1();
    await caseL2();
    await caseL3();
    await caseL4();
    await caseL5();
    await caseL6();
    await caseL7();
    await caseL8();
}

async function emergencyStop() {
    try { await stopServe(); } catch { /* best effort */ }
    for (const item of started.filter((entry) => same(entry))) {
        try { process.kill(item.pid, 'SIGTERM'); } catch { /* gone */ }
    }
    await sleep(2000);
    for (const item of started.filter((entry) => same(entry))) { try { process.kill(item.pid, 'SIGKILL'); } catch { /* gone */ } }
    // A runner that no record led to: found by this run's own socket folder, identity read again right before the signal.
    for (const item of runnersOfThisRun()) {
        const entry = { pid: item.pid, start: psStart(item.pid) };
        if (entry.start !== null && psStart(item.pid) === entry.start) { try { process.kill(item.pid, 'SIGKILL'); } catch { /* gone */ } }
    }
}

let exitCode = 0;
let finishing = false;
async function finish() {
    if (finishing) return;
    finishing = true;
    await runCase('teardown', 'cleanup: nothing of this run is left, ~/.pworker is untouched, no key in any output', async () => {
        // Every file this run wrote is read again and searched for each secret learned during the run (a runner's key is learned when its
        // command line is first read, so an earlier chunk of a serve log was scrubbed by pattern only). A hit is rewritten scrubbed and fails.
        const known = [...secrets].filter((secret) => secret.length >= 8);
        const files = [];
        const collect = (dir) => { for (const name of fs.readdirSync(dir)) { const full = path.join(dir, name); if (fs.statSync(full).isDirectory()) collect(full); else files.push(full); } };
        collect(outDir);
        await closeCommandLog();
        const leaked = [];
        for (const file of files) {
            const text = fs.readFileSync(file, 'utf8');
            if (known.some((secret) => text.includes(secret))) { leaked.push(path.relative(outDir, file)); fs.writeFileSync(file, scrub(text)); }
        }
        check('no runner key or token value appears in any output file', leaked.length === 0, `${runnerKeys.size} runner key(s) and ${known.length - runnerKeys.size} other secret(s) known; ${files.length} files scanned${leaked.length ? `; found in ${leaked.join(', ')} (rewritten scrubbed)` : '; none found'}`);
        const leftAtStart = [...started.filter((entry) => same(entry))];
        const leftRunners = runnersOfThisRun();
        check('no serve or runner process of this run is left', leftAtStart.length === 0 && leftRunners.length === 0, `started and alive: ${leftAtStart.map((entry) => `${entry.what} ${entry.pid}`).join(', ') || 'none'}; naming the home: ${leftRunners.map((item) => item.pid).join(', ') || 'none'}`);
        check('no launch record or socket is left', readLaunchRecords().length === 0 && socketsLeft().length === 0, `launch records ${readLaunchRecords().length}; sockets ${socketsLeft().length}`);
        const now = fs.existsSync(realPworker);
        if (pworkerBefore.existed) check('~/.pworker existed before and is unchanged', now && signatureOf(realPworker) === pworkerBefore.signature, `existed before: yes; exists now: ${now}`);
        else check('~/.pworker was not created by the run', !now, `existed before: no; exists now: ${now}`);
        const spent = [...started];
        record.environment = { ...record.environment, signalled: spent.map((entry) => entry.what), finishedAt: new Date().toISOString() };
    }, { hidden: true });
    // Safety net after the assertions above (so a leftover is reported before it is removed): signal survivors, remove what this run owns.
    await emergencyStop();
    if (ownsHome && !args.flags.has('keep-home') && home.startsWith(`${fs.realpathSync(tmpBase)}${path.sep}pwcl-`)) fs.rmSync(home, { recursive: true, force: true });
    if (socketDir.startsWith(`${fs.realpathSync(tmpBase)}${path.sep}pwcs-`)) fs.rmSync(socketDir, { recursive: true, force: true });
    const all = [...record.steps, ...record.cases];
    const failed = all.filter((item) => item.status === 'fail');
    const inconclusive = all.filter((item) => item.status === 'inconclusive');
    exitCode = failed.length ? 1 : inconclusive.length ? 3 : 0;
    record.ok = exitCode === 0;
    record.finishedAt = new Date().toISOString();
    saveRecord();
    const line = (item) => `${item.status.toUpperCase().padEnd(12)} ${String(item.id).padEnd(9)} ${item.title ?? ''}${item.error || item.reason ? `\n             ${item.error ?? item.reason}` : ''}${item.checks.filter((entry) => entry.status !== 'pass').map((entry) => `\n             ${entry.status} ${entry.name}: ${ev(entry.evidence, 300)}`).join('')}`;
    process.stdout.write(`${[...record.cases, ...all.filter((item) => !record.cases.includes(item) && item.status !== 'pass')].map(line).join('\n')}\n`);
    process.stdout.write(`${exitCode === 0 ? 'PASSED' : exitCode === 3 ? 'INCONCLUSIVE' : 'FAILED'}: ${path.join(outDir, 'steps.json')}\n`);
    await closeCommandLog();
    process.exit(exitCode);
}

for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => { record.steps.push({ id: 'interrupted', status: 'fail', checks: [], error: `received ${signal}` }); finish(); });

try {
    await mainRun();
} catch (error) {
    if (!(error instanceof Failure || error instanceof Inconclusive)) record.steps.push({ id: 'unexpected', status: 'fail', checks: [], error: scrub(error?.stack ?? error) });
    else record.steps.push({ id: 'stopped', status: error instanceof Inconclusive ? 'inconclusive' : 'fail', checks: [], error: scrub(error.message) });
    for (const id of CASE_IDS.filter(selected)) if (!record.cases.some((item) => item.id === id)) recordCaseFailure(id, new Inconclusive(`not run: ${scrub(error?.message ?? error)}`));
} finally {
    await finish();
}
