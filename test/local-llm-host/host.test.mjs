// The local-llm host of one home: ownership, reaping and the lazy controller, with a real Unix socket and a stand-in runner
// process (fixtures/fake-runner.mjs) where a test needs a runner, and a fake /proc where it needs a process table. The host is
// given `platformName: 'linux'` so these run on any machine; the real /proc is used only by linux-orphan.test.mjs.
import '../home.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { httpFetch } from '../../lib/http-fetch.mjs';
import { createLocalLlmHost, hostLockFile, prepareSocketDir } from '../../lib/local-llm-host/host.mjs';
import { LAUNCH_SCHEMA, readRecords } from '../../lib/local-llm-host/launch.mjs';
import { ownerPortFor } from '../../lib/local-llm-host/ownership.mjs';
import { CRASH_ENV, createRunnerStarter, envCrashHook } from '../../lib/local-llm-host/starter.mjs';
import { createLaunchRecorder } from '../../lib/local-llm-host/launch.mjs';
import { MODEL, SNAPSHOT, fakeAdapter, freePort, runModel, scratch, until } from './fixtures/engine.mjs';
import { recordingKill, removeProc, runnerArgv, writeProc } from './fixtures/proc.mjs';

/** A host on a fresh home with a fake /proc, a free owner port and the stand-in runner. */
async function makeHost(t, { home = scratch(t), config = {}, extra = {}, withRunner = true } = {}) {
    const procDir = path.join(home, 'proc');
    fs.mkdirSync(procDir, { recursive: true });
    const logs = [];
    const ownerPort = extra.ownerPort ?? await freePort();
    const adapter = fakeAdapter();
    const controllerOptions = withRunner ? {
        runners: { 'fake-gguf': adapter },
        snapshot: async () => structuredClone(SNAPSHOT),
        download: async ({ artifact }) => ({ status: 'complete', path: `/models/${artifact.file}`, bytesTransferred: 0 }),
        inspect: async () => ({ state: 'complete', bytes: 1 }),
        remove: async () => 1,
        detectRunner: () => ({ installed: true, version: 'v1', reason: null }),
        pollMs: 5,
        stopGraceMs: 500,
        profile: 'dedicated',
        ...extra.controllerOptions,
    } : (extra.controllerOptions ?? {});
    const host = createLocalLlmHost({
        config, home, env: { PATH: '/usr/bin:/bin', HOME: '/home/test', LANG: 'C.UTF-8' }, platformName: 'linux', procDir, ownerPort,
        log: (line) => logs.push(line), controllerOptions, ...extra, ...(extra.controllerOptions ? { controllerOptions } : {}),
    });
    t.after(() => host.drain().catch(() => {}));
    return { host, home, procDir, logs, ownerPort };
}

test('creating a host has no side effects: no directory, no listener, no controller', async (t) => {
    const home = scratch(t);
    const { host } = await makeHost(t, { home });
    assert.deepEqual(fs.readdirSync(home), ['proc'], 'only the fake /proc of the test');
    assert.equal(host.current(), null);
    const description = host.describe();
    assert.equal(description.ownership.state, 'idle');
    assert.deepEqual(fs.readdirSync(home), ['proc']);
});

test('acquire lays out the 0700 home layout, takes the owner port and writes the informational owner record', async (t) => {
    const { host, home, ownerPort } = await makeHost(t);
    const result = await host.acquire();
    assert.equal(result.owned, true);
    assert.equal(result.port, ownerPort);
    for (const name of ['models', 'runners', 'run', 'state', 'logs', 'home']) {
        const stat = fs.statSync(path.join(home, 'local-llm', name));
        assert.equal(stat.isDirectory(), true, name);
        assert.equal(stat.mode & 0o777, 0o700, `${name} is 0700`);
    }
    const socketDir = path.join(home, 'r');
    assert.equal(fs.statSync(socketDir).mode & 0o777, 0o700);
    assert.equal(host.layout.socketDir, socketDir);
    const record = JSON.parse(fs.readFileSync(path.join(home, 'local-llm', 'state', 'owner.json'), 'utf8'));
    assert.equal(record.pid, process.pid);
    assert.equal(record.port, ownerPort);
    await host.drain();
    // Released: another host takes the same port at once.
    const { host: next } = await makeHost(t, { home, extra: { ownerPort } });
    assert.equal((await next.acquire()).owned, true);
});

test('the real owner port of a home comes from its realpath', async (t) => {
    const home = scratch(t);
    const procDir = path.join(home, 'proc');
    fs.mkdirSync(procDir);
    const link = `${home}-link`;
    fs.symlinkSync(home, link);
    t.after(() => fs.rmSync(link, { force: true }));
    const host = createLocalLlmHost({ config: {}, home: link, env: {}, platformName: 'linux', procDir });
    t.after(() => host.drain());
    const { port } = await host.acquire();
    assert.equal(port, ownerPortFor(fs.realpathSync(home)));
    assert.equal(ownerPortFor(fs.realpathSync(link)), port);
});

test('a loser never reaps: a process that does not get the port reads no launch record and signals nothing', async (t) => {
    const { host: owner, home, procDir, ownerPort } = await makeHost(t);
    await owner.acquire();
    // The live owner's runner, recorded and running, in the home both of them use.
    const sock = path.join(home, 'r', 'abcdef01.sock');
    const launchDir = path.join(home, 'local-llm', 'state', 'launch');
    writeProc(procDir, { pid: 4100, startTime: '5000', argv: runnerArgv(sock) });
    fs.writeFileSync(path.join(launchDir, 'abcdef01.json'), `${JSON.stringify({ schema: LAUNCH_SCHEMA, startId: 'abcdef01', socketPath: sock, phase: 'spawned', pid: 4100, startTime: '5000' })}\n`);
    const kill = recordingKill();
    const { host: loser } = await makeHost(t, { home, extra: { ownerPort, killImpl: kill } });
    await assert.rejects(loser.acquire(), (error) => error.code === 'owner_held' && error.details.pid === process.pid);
    assert.deepEqual(kill.calls, [], 'the refused process signalled nothing');
    assert.equal(fs.existsSync(path.join(launchDir, 'abcdef01.json')), true, 'and left the live owner\'s record');
    await assert.rejects(loser.controller(), { code: 'owner_held' });
    assert.equal(loser.current(), null, 'no controller is created without the ownership');
});

test('the owner reaps at acquire: the orphan of an earlier process is ended before anything else starts', async (t) => {
    const home = scratch(t);
    const launchDir = path.join(home, 'local-llm', 'state', 'launch');
    fs.mkdirSync(launchDir, { recursive: true });
    const sock = path.join(home, 'r', 'abcdef02.sock');
    const procDir = path.join(home, 'proc');
    fs.mkdirSync(procDir);
    writeProc(procDir, { pid: 4200, startTime: '5100', argv: runnerArgv(sock) });
    fs.writeFileSync(path.join(launchDir, 'abcdef02.json'), `${JSON.stringify({ schema: LAUNCH_SCHEMA, startId: 'abcdef02', socketPath: sock, phase: 'spawned', pid: 4200, startTime: '5100' })}\n`);
    const kill = recordingKill({ dies: { SIGTERM: (pid) => removeProc(procDir, -pid) } });
    const { host, logs } = await makeHost(t, { home, extra: { killImpl: kill } });
    const result = await host.acquire();
    assert.deepEqual(kill.calls, [[-4200, 'SIGTERM']]);
    assert.deepEqual(result.reaped.reaped, ['abcdef02']);
    assert.ok(logs.some((line) => /ended the orphaned runner abcdef02/.test(line)));
});

test('a predecessor that is still draining is waited for; one that never leaves is a final owner_held for that wait, and a later attempt can succeed', async (t) => {
    const { host: first, home, ownerPort } = await makeHost(t);
    await first.acquire();
    const { host: second } = await makeHost(t, { home, extra: { ownerPort } });
    const waiting = second.acquire({ waitMs: 5000 });
    setTimeout(() => first.drain(), 400);
    assert.equal((await waiting).owned, true, 'acquired once the predecessor released');

    const { host: third } = await makeHost(t, { home, extra: { ownerPort } });
    await assert.rejects(third.acquire({ waitMs: 300 }), { code: 'owner_held' });
    await second.drain();
    assert.equal((await third.acquire()).owned, true, 'owner_held is not remembered for good');
});

test('a port held by a program that is not pworker is remembered as the refusal of every engine use, and the other parts keep working', async (t) => {
    const port = await freePort();
    const holder = net.createServer((socket) => socket.on('error', () => {}));
    await new Promise((resolve) => holder.listen(port, '127.0.0.1', resolve));
    t.after(() => new Promise((resolve) => { holder.close(resolve); holder.closeAllConnections?.(); }));
    const { host, logs } = await makeHost(t, { extra: { ownerPort: port } });
    await assert.rejects(host.acquire(), { code: 'owner_port_taken' });
    await assert.rejects(host.controller(), (error) => error.code === 'owner_port_taken' && error.message.includes(String(port)));
    assert.equal(host.refusal.code, 'owner_port_taken');
    assert.equal(host.describe().ownership.refusal.code, 'owner_port_taken');
    assert.ok(logs.some((line) => /not pworker/.test(line)));
    assert.equal(host.current(), null);
});

test('a disabled engine and an unsupported platform create nothing and say why', async (t) => {
    const disabledHome = scratch(t);
    const disabled = createLocalLlmHost({ config: { localLlm: { enabled: false } }, home: disabledHome, env: {}, platformName: 'linux' });
    assert.equal((await disabled.acquire()).owned, false);
    await assert.rejects(disabled.controller(), { code: 'local_llm_disabled' });
    assert.deepEqual(fs.readdirSync(disabledHome), []);

    const macHome = scratch(t);
    const mac = createLocalLlmHost({ config: {}, home: macHome, env: {}, platformName: 'darwin' });
    const result = await mac.acquire();
    assert.equal(result.owned, false);
    assert.match(result.reason, /not supported on darwin/);
    await assert.rejects(mac.controller(), { code: 'local_llm_unsupported' });
    assert.equal(mac.describe().supported.ok, false);
    assert.deepEqual(fs.readdirSync(macHome), [], 'no directory and no listener');
});

test('socketDir: the default is made 0700 whatever the umask, an existing wrong one is refused, and so are a symlink, another owner and a relative override', (t) => {
    const root = scratch(t);
    const previous = process.umask(0o000);
    try {
        const made = prepareSocketDir(path.join(root, 'made'));
        assert.equal(fs.statSync(made).mode & 0o777, 0o700, 'created 0700 under umask 000');
    } finally { process.umask(previous); }

    const loose = path.join(root, 'loose');
    fs.mkdirSync(loose, { mode: 0o755 });
    fs.chmodSync(loose, 0o755);
    assert.throws(() => prepareSocketDir(loose), (error) => error.code === 'invalid_socket_dir' && /mode 0755/.test(error.message));
    const group = path.join(root, 'group');
    fs.mkdirSync(group);
    fs.chmodSync(group, 0o770);
    assert.throws(() => prepareSocketDir(group), { code: 'invalid_socket_dir' }, 'group-writable');
    fs.chmodSync(group, 0o700);
    assert.equal(prepareSocketDir(group), group);

    const target = path.join(root, 'target');
    fs.mkdirSync(target, { mode: 0o700 });
    const link = path.join(root, 'link');
    fs.symlinkSync(target, link);
    assert.throws(() => prepareSocketDir(link), (error) => error.code === 'invalid_socket_dir' && /symbolic link/.test(error.message));
    const file = path.join(root, 'file');
    fs.writeFileSync(file, 'x');
    assert.throws(() => prepareSocketDir(file), /not a directory/);
    assert.throws(() => prepareSocketDir(target, { uid: process.getuid() + 1 }), (error) => error.code === 'invalid_socket_dir' && /owned by user/.test(error.message));
    for (const bad of ['relative/dir', '/a/../b', '/a:b', '/a,b']) assert.throws(() => prepareSocketDir(bad), { code: 'invalid_socket_dir' }, bad);
});

test('the localLlm.socketDir override is checked at acquire and refused before an owner port is taken', async (t) => {
    const root = scratch(t);
    const loose = path.join(root, 'loose');
    fs.mkdirSync(loose);
    fs.chmodSync(loose, 0o770);
    const { host } = await makeHost(t, { config: { localLlm: { socketDir: loose } } });
    await assert.rejects(host.acquire(), (error) => error.code === 'invalid_socket_dir' && error.message.includes(loose));
    assert.equal(host.describe().ownership.port, null);
    const link = path.join(root, 'link');
    fs.symlinkSync(root, link);
    const { host: linked } = await makeHost(t, { config: { localLlm: { socketDir: link } } });
    await assert.rejects(linked.acquire(), { code: 'invalid_socket_dir' });
    const good = path.join(root, 'good');
    const { host: fine } = await makeHost(t, { config: { localLlm: { socketDir: good } } });
    assert.equal((await fine.acquire()).owned, true);
    assert.equal(fine.layout.socketDir, good);
    assert.equal(fs.statSync(good).mode & 0o777, 0o700);
});

test('the socket path fits under the test home on both platforms, and an over-long home fails the run before launch, naming the limit and both fixes', async (t) => {
    const { host, home } = await makeHost(t);
    await host.acquire();
    const longest = path.join(host.layout.socketDir, '00000000.sock');
    assert.ok(Buffer.byteLength(longest) <= 103, `${longest} is ${Buffer.byteLength(longest)} bytes; macOS allows 103`);

    // A home so deep that <home>/r/<8 hex>.sock cannot fit sun_path.
    const deep = path.join(scratch(t), 'd'.repeat(60), 'd'.repeat(40));
    fs.mkdirSync(deep, { recursive: true });
    const { host: tooDeep } = await makeHost(t, { home: deep });
    const controller = await tooDeep.controller();
    await assert.rejects(controller.run({ modelId: MODEL, runnerId: 'fake-gguf', requestId: 'request-deep-01' }), (error) => {
        assert.equal(error.code, 'socket_path_too_long');
        assert.match(error.message, /107 bytes/);
        assert.match(error.message, /PWORKER_HOME/);
        assert.match(error.message, /localLlm\.socketDir/);
        return true;
    });
    assert.equal(fs.readdirSync(path.join(deep, 'local-llm', 'state', 'launch')).length, 0, 'no launch record: nothing was started');
    assert.ok(home);
});

test('a model runs on a real Unix socket: records before the process, a socket target that answers with the per-start key, and a clean stop', async (t) => {
    const { host, home } = await makeHost(t);
    const controller = await host.controller();
    const launchDir = path.join(home, 'local-llm', 'state', 'launch');
    await runModel(controller);
    const target = controller.chatTarget();
    assert.equal(target.baseUrl, 'http://localhost');
    assert.match(path.basename(target.socketPath), /^[0-9a-f]{8}\.sock$/);
    assert.equal(path.dirname(target.socketPath), path.join(home, 'r'));
    // The record names the running process and the socket.
    const [{ record }] = readRecords(launchDir);
    assert.equal(record.phase, 'spawned');
    assert.equal(record.socketPath, target.socketPath);
    assert.equal(record.pid, controller.state.deployment.runner.pid);
    assert.equal(controller.state.deployment.runner.port, null);
    // A real request over the socket, with the key the engine generated for this start.
    const wrong = await httpFetch('http://localhost/v1/models', { socketPath: target.socketPath, headers: { authorization: 'Bearer nope' } });
    assert.equal(wrong.status, 401);
    const answer = await httpFetch('http://localhost/v1/chat/completions', {
        method: 'POST', socketPath: target.socketPath, headers: { authorization: `Bearer ${target.apiKey}`, 'content-type': 'application/json' },
        body: JSON.stringify({ model: target.model, messages: [] }),
    });
    assert.equal((await answer.json()).choices[0].message.content, `echo ${target.model}`);
    // The runner is not reachable on a TCP port: the stand-in listens on the socket only.
    const stopped = controller.stop();
    await stopped;
    await until(() => !fs.existsSync(target.socketPath) && readRecords(launchDir).length === 0);
});

test('the runner starter writes the intent before the spawn and the pid after it, and a failure of either leaves nothing running unrecorded', async (t) => {
    const home = scratch(t);
    const launchDir = path.join(home, 'launch');
    const procDir = path.join(home, 'proc');
    fs.mkdirSync(procDir);
    const order = [];
    const recorder = createLaunchRecorder({ dir: launchDir, procDir, platformName: 'linux' });
    const spy = {
        ...recorder,
        intent: (listen) => { order.push('intent'); recorder.intent(listen); },
        spawned: (info) => { order.push('spawned'); recorder.spawned(info); },
    };
    let killed = 0;
    const child = (pid) => ({ pid, exited: new Promise(() => {}), kill: () => { killed += 1; }, running: true });
    const listen = { startId: '11111111', socketPath: path.join(home, 'r', '11111111.sock') };
    const startProcess = () => { order.push('spawn'); return child(7000); };
    const starter = createRunnerStarter({ recorder: spy, socketDir: path.join(home, 'r'), platform: {}, startProcess });
    starter({ listen, command: 'x', args: [] });
    assert.deepEqual(order, ['intent', 'spawn', 'spawned']);
    assert.equal(readRecords(launchDir)[0].record.pid, 7000);

    // No intent, no spawn.
    order.length = 0;
    const failing = { ...recorder, intent: () => { throw new Error('disk full'); } };
    assert.throws(() => createRunnerStarter({ recorder: failing, socketDir: '/r', platform: {}, startProcess })({ listen, command: 'x', args: [] }), /disk full/);
    assert.deepEqual(order, []);

    // A pid that cannot be recorded stops the runner instead of leaving it unfindable.
    const lateFailure = { ...recorder, spawned: () => { throw Object.assign(new Error('disk full'), { code: 'ENOSPC' }); } };
    assert.throws(() => createRunnerStarter({ recorder: lateFailure, socketDir: '/r', platform: {}, startProcess })({ listen, command: 'x', args: [] }), /disk full/);
    assert.equal(killed, 1);
    assert.throws(() => starter({ command: 'x', args: [] }), /listen address/);
});

test('a crash between the spawn and the record leaves a child the reaper can identify, found by its socket path', async (t) => {
    const home = scratch(t);
    const launchDir = path.join(home, 'local-llm', 'state', 'launch');
    const procDir = path.join(home, 'proc');
    fs.mkdirSync(procDir);
    const socketDir = path.join(home, 'r');
    fs.mkdirSync(launchDir, { recursive: true });
    fs.mkdirSync(socketDir, { mode: 0o700 });
    const listen = { startId: '22222222', socketPath: path.join(socketDir, '22222222.sock') };
    const recorder = createLaunchRecorder({ dir: launchDir, procDir, platformName: 'linux' });
    // The "process" exists in the fake process table as soon as it is spawned, like a real child.
    const startProcess = () => { writeProc(procDir, { pid: 7100, startTime: '600', argv: runnerArgv(listen.socketPath) }); return { pid: 7100, exited: new Promise(() => {}), kill() {}, running: true }; };
    const crash = new Error('crash after spawn');
    const starter = createRunnerStarter({
        recorder, socketDir, platform: {}, startProcess,
        crashHook: (point) => { if (point === 'after-spawn') throw crash; },
    });
    assert.throws(() => starter({ listen, command: 'x', args: [] }), (error) => error === crash);
    // On disk: the intent record only. In the process table: the child.
    assert.deepEqual(readRecords(launchDir).map(({ record }) => [record.phase, record.pid]), [['intent', null]]);
    assert.equal(fs.existsSync(path.join(procDir, '7100')), true);
    // The next owner: acquire reaps it.
    const kill = recordingKill({ dies: { SIGTERM: (pid) => removeProc(procDir, -pid) } });
    const host = createLocalLlmHost({ config: {}, home, env: {}, platformName: 'linux', procDir, ownerPort: await freePort(), killImpl: kill });
    t.after(() => host.drain());
    const result = await host.acquire();
    assert.deepEqual(kill.calls, [[-7100, 'SIGTERM']]);
    assert.deepEqual(result.reaped.reaped, ['22222222']);
    assert.deepEqual(readRecords(launchDir), []);
});

test('the crash hook is deterministic: it fires only at the named point, and only in the process whose environment names it', () => {
    const killed = [];
    const kill = (pid, signal) => killed.push([pid, signal]);
    const hook = envCrashHook({ [CRASH_ENV]: 'after-spawn' }, { kill });
    hook('after-intent');
    assert.deepEqual(killed, []);
    hook('after-spawn');
    assert.deepEqual(killed, [[process.pid, 'SIGKILL']]);
    envCrashHook({}, { kill })('after-spawn');
    assert.equal(killed.length, 1);
});

test('the host\'s runner lock files exist per architecture and are complete locks of the two Linux runners', async () => {
    const { validateRunnerLock } = await import('../../lib/local-llm/controller/runnerLock.mjs');
    for (const arch of ['x64', 'arm64']) {
        const file = hostLockFile(arch);
        assert.ok(file && fs.existsSync(file), arch);
        const lock = validateRunnerLock(JSON.parse(fs.readFileSync(file, 'utf8')), { requireExecutable: true });
        assert.deepEqual(Object.keys(lock.runners).sort(), ['llama.cpp', 'llama.cpp-cpu']);
        for (const entry of Object.values(lock.runners)) {
            assert.equal(entry.version, 'b11295');
            assert.equal(entry.check.executable, 'llama-server');
        }
        // The CUDA build ships its runtime in a second archive, unpacked next to the binaries (their rpath is $ORIGIN).
        const cuda = lock.runners['llama.cpp'];
        assert.equal(cuda.files.length, 2);
        assert.ok(cuda.files.some((f) => f.name.startsWith('cudart-')));
        assert.ok(cuda.files.every((f) => f.extract === undefined && f.strip === undefined), 'both archives unpack into the runnable copy\'s root, one level stripped');
        assert.ok('libcuda.so.1' in cuda.check.optionalLibraries, 'the driver library may be absent on a machine without a GPU');
    }
    assert.equal(hostLockFile('ia32'), null);
});

test('only the CUDA and CPU llama.cpp builds are offered, the CUDA build only with nvidia-smi on PATH, and every other runner is listed with a reason', async (t) => {
    const withGpu = (await makeHost(t, { withRunner: false, extra: { which: (name) => (name === 'nvidia-smi' ? '/usr/bin/nvidia-smi' : null) } })).host.describe();
    const without = (await makeHost(t, { withRunner: false, extra: { which: () => null } })).host.describe();
    const by = (description) => Object.fromEntries(description.runners.map((runner) => [runner.id, runner]));
    assert.equal(by(withGpu)['llama.cpp'].supported, true);
    assert.equal(by(without)['llama.cpp'].supported, false);
    assert.match(by(without)['llama.cpp'].reason, /NVIDIA GPU/);
    assert.equal(by(withGpu)['llama.cpp-cpu'].supported, true);
    assert.equal(by(without)['llama.cpp-cpu'].supported, true);
    for (const id of ['ik_llama.cpp', 'ollama', 'vllm', 'tabbyapi', 'lmstudio', 'llama.cpp-vulkan']) {
        assert.equal(by(without)[id].supported, false, id);
        assert.ok(by(without)[id].reason, id);
    }
});

test('the engine\'s environment is filtered: PATH with the directory of nvidia-smi, HOME, LANG and the Hugging Face token, nothing else', async (t) => {
    const home = scratch(t);
    fs.mkdirSync(path.join(home, 'keys'));
    fs.writeFileSync(path.join(home, 'keys', 'huggingface.env'), 'HF_TOKEN=hf_synthetic_file_token\n');
    const env = { PATH: '/opt/gpu/bin:/usr/bin', HOME: '/home/u', LANG: 'de_DE.UTF-8', CUDA_MPS_PIPE_DIRECTORY: '/tmp/mps', OPENAI_API_KEY: 'sk-secret', LOCAL_LLM_AMD_APU: '1' };
    const which = (name) => (name === 'nvidia-smi' ? '/opt/gpu/bin/nvidia-smi' : null);
    const host = createLocalLlmHost({ config: {}, home, env, platformName: 'linux', which });
    assert.deepEqual(host.engineEnvironment(), {
        PATH: '/usr/local/bin:/usr/bin:/bin:/opt/gpu/bin', HOME: '/home/u', LANG: 'de_DE.UTF-8', HF_TOKEN: 'hf_synthetic_file_token',
    });
    assert.equal(host.redact('token hf_synthetic_file_token here'), 'token *** here');
    // The process environment wins over the key file; with neither there is no HF_TOKEN key at all.
    const withEnvToken = createLocalLlmHost({ config: {}, home, env: { ...env, HF_TOKEN: 'hf_synthetic_env_token' }, platformName: 'linux', which });
    assert.equal(withEnvToken.engineEnvironment().HF_TOKEN, 'hf_synthetic_env_token');
    assert.equal(withEnvToken.redact('a hf_synthetic_env_token b'), 'a *** b');
    const bare = createLocalLlmHost({ config: {}, home: scratch(t), env: { PATH: '/usr/bin' }, platformName: 'linux', which: () => null, homeDir: '/home/fallback' });
    assert.deepEqual(bare.engineEnvironment(), { PATH: '/usr/local/bin:/usr/bin:/bin', HOME: '/home/fallback', LANG: 'C.UTF-8' });
    assert.equal(bare.redact('nothing to hide'), 'nothing to hide');
});
