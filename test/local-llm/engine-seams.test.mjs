// The engine's inputs (slice B of the engine move): the host, the message wording, the platform strategies, the
// runner table built from a host, the per-instance hardware-query reaper, the signal seam and the refusal of hidden
// default paths. The moved suites run against an agent-like host (fixtures/host.mjs); this file runs the engine against
// hosts that are not, and proves the controller hands each input to the code that uses it.
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog } from '../../lib/local-llm/controller/catalog.mjs';
import { createController } from '../../lib/local-llm/controller/deployments.mjs';
import { createHardwareQueryReaper, readGpu, readSnapshot } from '../../lib/local-llm/controller/hardware.mjs';
import * as runnerLockModule from '../../lib/local-llm/controller/runnerLock.mjs';
import { loadRunnerLock, loadRunnerLocks } from '../../lib/local-llm/controller/runnerLock.mjs';
import * as catalogModule from '../../lib/local-llm/controller/catalog.mjs';
import { setOomScoreAdj, startRunnerProcess } from '../../lib/local-llm/controller/runnerProcess.mjs';
import { createRunnerInstaller } from '../../lib/local-llm/controller/runnerInstaller.mjs';
import { vulkanContractOf } from '../../lib/local-llm/controller/vulkan.mjs';
import { LocalLlmError } from '../../lib/local-llm/errors.mjs';
import { DEFAULT_TOOL_PATH, defineHost, linuxPlatform } from '../../lib/local-llm/host.mjs';
import { DEFAULT_MESSAGES, MESSAGE_KEYS, defineMessages } from '../../lib/local-llm/messages.mjs';
import { createRunners, defaultPorts } from '../../lib/local-llm/runners/index.mjs';
import { runInstallCheck } from '../../lib/local-llm/tools/runnerInstallCheck.mjs';
import { HARDWARE_QUERY_REAP_MS } from '../../lib/local-llm/drainBudget.mjs';
import { SEED_CATALOG_FILE } from './fixtures/seedCatalog.mjs';
import { AGENT_HOST_SPEC, PLOINKY_MESSAGES, agentLikeHost } from './fixtures/host.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const KEY = 'k'.repeat(43);
const BARE = Object.freeze({ dataDir: '/x/data', runRoot: '/x/run' });
const FORBIDDEN = /ploinky|soul_gateway|Settings > Agents|local-llm/i;

const code = (expected, field) => (error) => error instanceof LocalLlmError && error.code === expected
    && (field === undefined || error.details?.field === field);
const messageCode = (key) => (error) => error instanceof LocalLlmError && error.code === 'invalid_messages' && (key === undefined || error.details?.key === key);

function tempDir(t, prefix) {
    const dir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), prefix));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

// ------------------------------------------------------------------ host

test('defineHost freezes the host and what it holds, fills the documented defaults and reads no file', (t) => {
    const touched = [];
    for (const name of ['readFileSync', 'statSync', 'lstatSync', 'existsSync', 'readdirSync', 'accessSync', 'realpathSync', 'openSync']) {
        const original = fs[name];
        t.mock.method(fs, name, (...args) => { touched.push(name); return original(...args); });
    }
    const host = defineHost({ ...BARE, executables: { ollama: '/nowhere/ollama' }, vulkan: { serverPath: '/v/s', vulkaninfo: '/v/i', icdDir: '/v/icd' } });
    assert.deepEqual(touched, [], 'defineHost touched the file system');
    for (const value of [host, host.executables, host.vulkan, host.platform]) assert.equal(Object.isFrozen(value), true);
    assert.equal(host.dataDir, '/x/data');
    assert.equal(host.runRoot, '/x/run');
    for (const field of ['sharedModelsRoot', 'imageContractFile', 'imageRunnerLockFile', 'seedCatalogFile', 'nvidiaSmi', 'nvidiaLibDir',
        'lmStudioSdkDir', 'uv', 'python', 'gpuAccess']) assert.equal(defineHost(BARE)[field], null, field);
    assert.equal(defineHost(BARE).vulkan, null);
    assert.deepEqual(defineHost(BARE).executables, {});
    assert.equal(defineHost(BARE).hfBaseUrl, 'https://huggingface.co');
    assert.equal(defineHost(BARE).shmDir, '/dev/shm');
    assert.equal(defineHost(BARE).toolPath, DEFAULT_TOOL_PATH);
    assert.equal(defineHost(BARE).agentRunnerLockFile('x64'), null);
    assert.equal(defineHost(BARE).platform.name, 'linux');
    assert.throws(() => { host.dataDir = '/y'; }, TypeError);
    // The specification is copied: changing it afterwards changes nothing.
    const spec = { ...BARE, executables: { ollama: '/a/ollama' } };
    const copied = defineHost(spec);
    spec.executables.ollama = '/b/ollama';
    assert.equal(copied.executables.ollama, '/a/ollama');
});

test('defineHost refuses a missing, relative, unnormalized or colon path, an unknown key and a malformed part, naming the field', () => {
    const vulkan = { serverPath: '/v/s', vulkaninfo: '/v/i', icdDir: '/v/icd' };
    const refused = [
        [undefined, '(host)'], [null, '(host)'], [[], '(host)'],
        [{ runRoot: '/x/run' }, 'dataDir'], [{ dataDir: '/x/data' }, 'runRoot'],
        [{ ...BARE, dataDir: 'data' }, 'dataDir'], [{ ...BARE, runRoot: '/x/../run' }, 'runRoot'], [{ ...BARE, runRoot: '/x/./run' }, 'runRoot'],
        [{ ...BARE, nvidiaLibDir: '/a:/b' }, 'nvidiaLibDir'], [{ ...BARE, sharedModelsRoot: '' }, 'sharedModelsRoot'], [{ ...BARE, uv: 7 }, 'uv'],
        [{ ...BARE, nvidiaSmi: 'nvidia-smi' }, 'nvidiaSmi'],
        [{ ...BARE, surprise: 1 }, 'surprise'],
        [{ ...BARE, executables: { 'no-such-runner': '/x/y' } }, 'executables.no-such-runner'],
        [{ ...BARE, executables: { ollama: 'ollama' } }, 'executables.ollama'],
        [{ ...BARE, executables: [] }, 'executables'],
        [{ ...BARE, vulkan: { serverPath: '/v/s' } }, 'vulkan.vulkaninfo'],
        [{ ...BARE, vulkan: { ...vulkan, extra: '/v/e' } }, 'vulkan.extra'],
        [{ ...BARE, vulkan: { ...vulkan, icdDir: '/v/a:b' } }, 'vulkan.icdDir'],
        [{ ...BARE, vulkan: '/v' }, 'vulkan'],
        [{ ...BARE, executables: { 'llama.cpp-vulkan': '/v/s' } }, 'executables.llama.cpp-vulkan'],
        [{ ...BARE, executables: { 'llama.cpp-vulkan': '/other/s' }, vulkan }, 'executables.llama.cpp-vulkan'],
        [{ ...BARE, gpuAccess: { source: 'x' } }, 'gpuAccess'], [{ ...BARE, gpuAccess: { report: () => ({}) } }, 'gpuAccess'],
        [{ ...BARE, agentRunnerLockFile: '/x/lock.json' }, 'agentRunnerLockFile'],
        [{ ...BARE, hfBaseUrl: 'ftp://huggingface.example' }, 'hfBaseUrl'], [{ ...BARE, hfBaseUrl: 'not a url' }, 'hfBaseUrl'],
        [{ ...BARE, toolPath: '' }, 'toolPath'],
        [{ ...BARE, platform: { name: 'linux' } }, 'platform'],
    ];
    for (const [spec, field] of refused) assert.throws(() => defineHost(spec), code('invalid_host', field), JSON.stringify(spec));
    // A PATH list is the one value that may hold ':'; an executable named by both keys must agree.
    assert.equal(defineHost({ ...BARE, toolPath: '/a:/b' }).toolPath, '/a:/b');
    assert.equal(defineHost({ ...BARE, executables: { 'llama.cpp-vulkan': '/v/s' }, vulkan }).executables['llama.cpp-vulkan'], '/v/s');
});

// -------------------------------------------------------------- platform

test('linuxPlatform produces exactly the paths, the OOM write and the tar argv the engine always used', async () => {
    const platform = linuxPlatform();
    assert.equal(platform.name, 'linux');
    assert.equal(platform.procDir, '/proc');
    assert.equal(platform.devDriDir, '/dev/dri');
    assert.equal(platform.openatPath(7, 'weights.gguf'), '/proc/self/fd/7/weights.gguf');
    assert.equal(platform.unreadableReason, '/proc/self/fd is not readable');
    const links = [];
    const fsApi = { promises: { readlink: async (link) => { links.push(link); return '/data/models/x.gguf'; } } };
    assert.equal(await platform.openedPathOf(fsApi, 12), '/data/models/x.gguf');
    assert.deepEqual(links, ['/proc/self/fd/12']);
    const writes = [];
    platform.setOomScoreAdj(4242, 1000, { fsApi: { writeFileSync: (file, text) => writes.push([file, text]) } });
    assert.deepEqual(writes, [['/proc/4242/oom_score_adj', '1000']]);
    assert.deepEqual(platform.tarFlags('/opt/runners/x/1', 1), ['-C', '/opt/runners/x/1', '--strip-components=1', '--no-same-owner', '--touch']);
    assert.deepEqual(platform.tarFlags('/r', 0), ['-C', '/r', '--strip-components=0', '--no-same-owner', '--touch']);
    // The module-level helper writes through the same strategy.
    const more = [];
    setOomScoreAdj(77, undefined, { fsApi: { writeFileSync: (file, text) => more.push([file, text]) } });
    assert.deepEqual(more, [['/proc/77/oom_score_adj', '1000']]);
    // A host may place /proc and /dev/dri elsewhere; the strategy follows.
    const moved = linuxPlatform({ procDir: '/mnt/proc', devDriDir: '/mnt/dri' });
    assert.equal(moved.openatPath(3, 'a'), '/mnt/proc/self/fd/3/a');
    assert.equal(moved.devDriDir, '/mnt/dri');
    assert.throws(() => linuxPlatform({ procDir: 'proc' }), code('invalid_host', 'platform.procDir'));
});

test('a platform given to the download binder decides which descriptor path is trusted', async () => {
    const { anchorRoot, openBound } = await import('../../lib/local-llm/controller/workspaceReuse.mjs');
    const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-platform-')));
    try {
        const file = path.join(root, 'a.bin');
        fs.writeFileSync(file, 'x');
        const seen = [];
        const platform = { ...linuxPlatform(), openedPathOf: async (fsApi, fd) => { seen.push(fd); return file; } };
        const anchor = await anchorRoot(fs, root, { platform });
        assert.equal(anchor.platform, platform);
        const opened = await openBound(fs, anchor, file);
        assert.ok(opened.handle ?? opened, 'the file opened');
        assert.equal(seen.length, 1, 'the binder asked the injected platform where the descriptor points');
        await (opened.handle ?? opened).close();
        // A platform that cannot say fails closed with the platform's own reason.
        const blind = { ...linuxPlatform(), openedPathOf: async () => { throw new Error('no'); }, unreadableReason: 'the descriptor table is hidden' };
        await assert.rejects(openBound(fs, await anchorRoot(fs, root, { platform: blind }), file),
            { code: 'UNSAFE_PATH', message: 'Cannot confirm which file was opened (the descriptor table is hidden)' });
    } finally {
        fs.rmSync(root, { recursive: true, force: true });
    }
});

test('startRunnerProcess marks the runner through the platform and signals its group only through killImpl', async () => {
    const child = Object.assign(new EventEmitter(), { pid: 4321, stdout: new EventEmitter(), stderr: new EventEmitter(), kill() {} });
    const marks = [];
    const kills = [];
    const platform = { ...linuxPlatform(), setOomScoreAdj: (pid, value) => marks.push([pid, value]) };
    const handle = startRunnerProcess({
        command: '/nowhere', args: [], env: {}, log: { append() {} }, spawnImpl: () => child, platform,
        // Records every signal; only the final SIGKILL of the group also ends this test's own child, so the stop can finish.
        killImpl: (pid, signal) => { kills.push([pid, signal]); if (signal === 'SIGKILL' && pid === -record.pid) process.kill(record.pid, 'SIGKILL'); },
    });
    assert.deepEqual(marks, [[4321, 1000]]);
    const stopping = handle.stop({ graceMs: 10 });
    setTimeout(() => child.emit('exit', null, 'SIGKILL'), 40);
    await stopping;
    assert.deepEqual(kills.slice(0, 2), [[-4321, 'SIGTERM'], [-4321, 'SIGKILL']]);
    assert.ok(kills.every(([pid]) => pid === -4321), 'no signal went anywhere else');
    // oomScoreAdj null marks nothing.
    const quiet = Object.assign(new EventEmitter(), { pid: 4322, stdout: new EventEmitter(), stderr: new EventEmitter(), kill() {} });
    marks.length = 0;
    startRunnerProcess({ command: '/nowhere', args: [], env: {}, log: { append() {} }, spawnImpl: () => quiet, platform, oomScoreAdj: null, killImpl: () => {} });
    assert.deepEqual(marks, []);
});

// -------------------------------------------------------------- messages

test('the default messages are neutral, every key is a function, and the Ploinky wording fixture covers exactly the keys', () => {
    assert.ok(MESSAGE_KEYS.length >= 14);
    assert.equal(Object.isFrozen(DEFAULT_MESSAGES), true);
    const samples = {
        notReady: [], gpuAccessReported: ['driver not granted'], gpuAbsent: [], hfTokenHint: [], vulkanRuntimeMissing: [],
        apuSwitchOff: ['Radeon', 'SWITCH', 'on'], vllmUnifiedSwitchHint: ['SWITCH', 'on'], lmStudioEnableHint: ['SWITCH', 'on'],
        lmStudioStagedUpdate: [], profileRestartGpuRecovered: [], profileRestartChanged: [], restartToUseDevice: ['vulkan'],
        restartToRunOnCpu: ['dedicated'], restartForMemoryModel: ['unified', 'dedicated'],
    };
    assert.deepEqual([...MESSAGE_KEYS].sort(), Object.keys(samples).sort(), 'a key without a sample is untested wording');
    for (const key of MESSAGE_KEYS) {
        assert.equal(typeof DEFAULT_MESSAGES[key], 'function', key);
        const text = DEFAULT_MESSAGES[key](...samples[key]);
        assert.equal(typeof text, 'string', key);
        assert.ok(text.length > 10, key);
        assert.doesNotMatch(text, FORBIDDEN, `${key}: ${text}`);
        assert.equal(typeof PLOINKY_MESSAGES[key](...samples[key]), 'string', `the Ploinky wording has ${key}`);
    }
    assert.equal(DEFAULT_MESSAGES.gpuAccessReported(null).includes('the cause is unknown'), true);
    assert.deepEqual(Object.keys(PLOINKY_MESSAGES).sort(), [...MESSAGE_KEYS].sort());
    // The Ploinky wording is Ploinky's: the control for the pattern above.
    assert.match(PLOINKY_MESSAGES.notReady(), /Settings > Agents/);
});

test('defineMessages overrides a key, refuses an unknown key or a non-function, and the result is frozen', () => {
    const custom = defineMessages({ notReady: () => 'nothing yet' });
    assert.equal(custom.notReady(), 'nothing yet');
    assert.equal(custom.gpuAbsent(), DEFAULT_MESSAGES.gpuAbsent());
    assert.equal(Object.isFrozen(custom), true);
    assert.throws(() => defineMessages({ notReadi: () => 'x' }), messageCode('notReadi'));
    assert.throws(() => defineMessages({ notReady: 'text' }), messageCode('notReady'));
    assert.throws(() => defineMessages(null), messageCode());
    assert.throws(() => defineMessages([]), messageCode());
    assert.equal(defineMessages().notReady(), DEFAULT_MESSAGES.notReady());
});

// ------------------------------------------------------------------- R2

test('the seed catalog and the runner locks are never read from a default location', async (t) => {
    const dir = tempDir(t, 'local-llm-r2-');
    for (const missing of [undefined, null, '', 7]) {
        assert.throws(() => catalogModule.loadSeedCatalog(missing), code('missing_file', 'file'), String(missing));
        assert.throws(() => loadRunnerLock(missing), code('missing_file', 'file'), String(missing));
    }
    assert.throws(() => catalogModule.loadSeedCatalog(), code('missing_file'));
    assert.throws(() => loadRunnerLocks(), code('missing_file'));
    assert.throws(() => loadRunnerLocks({}), code('missing_file'));
    assert.throws(() => loadRunnerLocks({ image: null }), code('missing_file'), 'the agent lock must be named too, even to say it is absent');
    assert.throws(() => loadRunnerLocks({ agent: null }), code('missing_file'), 'the image lock must be named too, even to say it is absent');
    // A named file is read; null says there is none; a named file that does not exist is an empty lock (an older image).
    assert.equal(catalogModule.loadSeedCatalog(SEED_CATALOG_FILE).length > 0, true);
    const none = loadRunnerLocks({ image: null, agent: null });
    assert.deepEqual([none.runners, none.origin, none.clashes, none.ignored], [{}, {}, [], []]);
    const absent = loadRunnerLocks({ image: path.join(dir, 'absent.json'), agent: path.join(dir, 'absent-agent.json') });
    assert.deepEqual(absent.runners, {});
    assert.deepEqual(loadRunnerLock(path.join(dir, 'absent.json')).runners, {});
    // The engine has no default location and no agent-lock helper to name one.
    assert.equal('agentRunnerLockFile' in runnerLockModule, false);
    assert.equal('DEFAULT_RUNNER_LOCK' in runnerLockModule, false);
    // The controller refuses a host that gives it no catalog file and no catalog.
    const host = defineHost({ dataDir: tempDir(t, 'local-llm-r2-data-'), runRoot: path.join(dir, 'run') });
    assert.throws(() => createController({ host }), code('missing_file', 'file'));
    assert.throws(() => createController({}), code('invalid_host', 'host'));
});

// -------------------------------------------------------------- runners

const API_KEY = 'a'.repeat(32);
const MODEL = Object.freeze({ id: 'm1' });

test('a bare host gives runners without executables or driver directories; they run from a lock\'s copy or are unsupported', () => {
    const runners = createRunners({ host: defineHost(BARE) });
    assert.deepEqual(Object.keys(runners), ['llama.cpp', 'ik_llama.cpp', 'ollama', 'vllm', 'tabbyapi', 'lmstudio', 'llama.cpp-cpu', 'llama.cpp-vulkan']);
    assert.equal(Object.isFrozen(runners), true);
    for (const id of ['llama.cpp', 'ik_llama.cpp', 'ollama', 'llama.cpp-vulkan', 'llama.cpp-cpu']) assert.equal(runners[id].executable, null, id);
    const launch = runners['llama.cpp'].buildLaunch({ artifactPath: '/m/q.gguf', params: {}, port: 18080, apiKey: API_KEY, model: MODEL, runnerDir: '/x/run/llama.cpp/b1' });
    assert.equal(launch.command, '/x/run/llama.cpp/b1/llama-server');
    assert.deepEqual(launch.env, {});
    const ik = runners['ik_llama.cpp'].buildLaunch({ artifactPath: '/m/q.gguf', params: {}, port: 18081, apiKey: API_KEY, model: MODEL, runnerDir: '/x/run/ik/1' });
    assert.equal(ik.command, '/x/run/ik/1/llama-server');
    const cpu = runners['llama.cpp'].buildLaunch({ artifactPath: '/m/q.gguf', params: {}, port: 18080, apiKey: API_KEY, model: MODEL, runnerDir: '/x/run/llama.cpp/b1', profile: 'cpu' });
    assert.deepEqual(cpu.env, { CUDA_VISIBLE_DEVICES: '' });
    // Ollama needs its own binary or a runnable copy; vLLM and TabbyAPI set no driver path.
    assert.throws(() => runners.ollama.buildLaunch({ params: {}, port: 11434, dataDir: '/x/data', model: MODEL }), { code: 'invalid_launch' });
    const ollama = runners.ollama.buildLaunch({ params: {}, port: 11434, dataDir: '/x/data', model: MODEL, runnerDir: '/x/run/ollama/1' });
    assert.equal(ollama.command, '/x/run/ollama/1/bin/ollama');
    assert.equal('LD_LIBRARY_PATH' in ollama.env, false);
    const vllm = runners.vllm.buildLaunch({ runnerDir: '/x/run/vllm/1', artifactPath: '/m/w', params: {}, port: 18082, apiKey: API_KEY, model: MODEL,
        cacheDir: '/x/cache', rpcDir: '/x/rpc', gpuMemoryUtilization: 0.5 });
    assert.equal('LD_LIBRARY_PATH' in vllm.env || 'TRITON_LIBCUDA_PATH' in vllm.env, false);
    const tabby = runners.tabbyapi.buildLaunch({ runnerDir: '/x/run/tabby/1', artifactPath: '/m/w', params: {}, port: 18083, apiKey: API_KEY, model: MODEL,
        cacheDir: '/x/cache', tmpDir: '/x/tmp' });
    assert.equal('LD_LIBRARY_PATH' in tabby.env, false);
    // No LM Studio SDK directory: unsupported, with a reason, and out of the port table.
    assert.equal(runners.lmstudio.supported, false);
    assert.match(runners.lmstudio.unsupportedReason, /LM Studio/);
    assert.equal('lmstudio' in defaultPorts(runners), false);
    assert.equal(Object.keys(defaultPorts(runners)).length, 7);
    // A Vulkan launch needs the host's Vulkan layout.
    assert.throws(() => runners['llama.cpp-vulkan'].buildLaunch({ artifactPath: '/m/q.gguf', params: {}, port: 18086, apiKey: API_KEY, model: MODEL, runnerDir: '/x/run/v/1',
        profile: 'vulkan', icdFile: '/opt/local-llm/vulkan/icd.d/radeon_icd.json' }), { code: 'invalid_launch' });
    assert.equal(vulkanContractOf({ vulkan_contract: '1' }, null), null);
});

test('a host\'s executables, driver directory, SDK directory and Vulkan layout are the ones the runners launch', () => {
    const vulkan = { serverPath: '/srv/vk/llama-server', vulkaninfo: '/srv/vk/vulkaninfo', icdDir: '/srv/vk/icd' };
    const host = defineHost({ ...BARE, nvidiaLibDir: '/srv/nv/lib', lmStudioSdkDir: '/srv/lms/sdk', vulkan, toolPath: '/srv/bin:/usr/bin',
        executables: { 'llama.cpp': '/srv/llama/llama-server', 'ik_llama.cpp': '/srv/ik/llama-server', ollama: '/srv/ollama/ollama' } });
    const runners = createRunners({ host });
    assert.equal(runners['llama.cpp'].executable, '/srv/llama/llama-server');
    assert.equal(runners['ik_llama.cpp'].executable, '/srv/ik/llama-server');
    assert.equal(runners.ollama.executable, '/srv/ollama/ollama');
    assert.equal(runners['llama.cpp-vulkan'].executable, '/srv/vk/llama-server', 'the Vulkan build is launched from the Vulkan layout');
    assert.equal(runners['llama.cpp-cpu'].executable, null);
    assert.equal(runners.lmstudio.supported, true);
    const launch = (runner, extra = {}) => runner.buildLaunch({ artifactPath: '/m/q.gguf', params: {}, port: 18080, apiKey: API_KEY, model: MODEL, ...extra });
    assert.deepEqual(launch(runners['llama.cpp']), { ...launch(runners['llama.cpp']), command: '/srv/llama/llama-server', env: { LD_LIBRARY_PATH: '/srv/nv/lib' } });
    assert.equal(launch(runners['llama.cpp'], { runnerDir: '/x/run/llama.cpp/b2' }).command, '/srv/llama/llama-server', 'the host\'s executable wins over a runnable copy');
    assert.equal(runners.ollama.buildLaunch({ params: {}, port: 11434, dataDir: '/x/data', model: MODEL }).env.LD_LIBRARY_PATH, '/srv/nv/lib');
    assert.equal(runners.lmstudio.buildLaunch({ runnerDir: '/x/run/lms/1' }).env.LD_LIBRARY_PATH, '/srv/nv/lib');
    const vllm = runners.vllm.buildLaunch({ runnerDir: '/x/run/vllm/1', artifactPath: '/m/w', params: {}, port: 18082, apiKey: API_KEY, model: MODEL,
        cacheDir: '/x/cache', rpcDir: '/x/rpc', gpuMemoryUtilization: 0.5 });
    assert.equal(vllm.env.LD_LIBRARY_PATH, '/srv/nv/lib');
    assert.equal(vllm.env.TRITON_LIBCUDA_PATH, '/srv/nv/lib');
    // The ICD files are the host's, and only they are accepted for a launch.
    const vk = (profile, icdFile) => launch(runners['llama.cpp-vulkan'], { profile, icdFile });
    assert.deepEqual(vk('vulkan', '/srv/vk/icd/radeon_icd.json').env, { VK_DRIVER_FILES: '/srv/vk/icd/radeon_icd.json', VK_LOADER_LAYERS_DISABLE: '~all~' });
    assert.equal(vk('venus', '/srv/vk/icd/virtio_icd.json').env.VK_DRIVER_FILES, '/srv/vk/icd/virtio_icd.json');
    assert.throws(() => vk('venus', '/srv/vk/icd/radeon_icd.json'), { code: 'invalid_launch' });
    assert.throws(() => vk('vulkan', '/opt/local-llm/vulkan/icd.d/radeon_icd.json'), { code: 'invalid_launch' });
    // The probe of the Vulkan build names both of the host's ICD files; the other runners' probes use the host's PATH and driver directory.
    const probes = [];
    const spawnSync = (command, args, options) => { probes.push({ command, env: options.env }); return { status: 0, stdout: 'build 11159,', stderr: '' }; };
    runners['llama.cpp-vulkan'].detect({ spawnSync });
    assert.equal(probes[0].command, '/srv/vk/llama-server');
    assert.equal(probes[0].env.VK_DRIVER_FILES, '/srv/vk/icd/radeon_icd.json:/srv/vk/icd/virtio_icd.json');
    const savedPath = process.env.PATH;
    try {
        delete process.env.PATH;
        runners['llama.cpp'].detect({ spawnSync });
        assert.equal(probes[1].env.PATH, '/srv/bin:/usr/bin');
        assert.equal(probes[1].env.LD_LIBRARY_PATH, '/srv/nv/lib');
    } finally {
        process.env.PATH = savedPath;
    }
    // vulkanContractOf compares against the host's layout, not a built-in one.
    const keys = { vulkan_contract: '1', vulkan_llama_cpp: 'b11159', vulkan_llama_cpp_server: '/srv/vk/llama-server', vulkan_mesa: '25.3.6',
        vulkan_icd_radv: '/srv/vk/icd/radeon_icd.json', vulkan_icd_venus: '/srv/vk/icd/virtio_icd.json', vulkaninfo: '/srv/vk/vulkaninfo' };
    assert.notEqual(vulkanContractOf(keys, vulkan), null);
    assert.equal(vulkanContractOf(keys, { ...vulkan, icdDir: '/opt/local-llm/vulkan/icd.d' }), null);
    assert.equal(vulkanContractOf({ ...keys, vulkaninfo: '/usr/bin/vulkaninfo' }, vulkan), null);
});

test('the runners word their operator hints through the messages they were built with', () => {
    const host = agentLikeHost();
    const wording = defineMessages({
        vllmUnifiedSwitchHint: (name, value) => `SET ${name}=${value} FOR VLLM`,
        lmStudioEnableHint: (name, value) => `SET ${name}=${value} FOR LMSTUDIO`,
    });
    const custom = createRunners({ host, messages: wording });
    assert.match(custom.vllm.enabled({}, 'unified').reason, /SET LOCAL_LLM_VLLM_UNIFIED=experimental FOR VLLM$/);
    assert.match(custom.lmstudio.enabled({}).reason, /SET LOCAL_LLM_LMSTUDIO=internal-use FOR LMSTUDIO$/);
    const neutral = createRunners({ host });
    for (const reason of [neutral.vllm.enabled({}, 'unified').reason, neutral.vllm.enabled({}, null).reason, neutral.lmstudio.enabled({}).reason]) {
        assert.doesNotMatch(reason, FORBIDDEN);
        assert.match(reason, /LOCAL_LLM_/);
    }
    assert.deepEqual(neutral.lmstudio.enabled({ LOCAL_LLM_LMSTUDIO: 'internal-use' }), { enabled: true, reason: null });
});

// ----------------------------------------------------------- the reaper

test('each reaper keeps its own stopped queries, and a query with no reaper is still stopped by its signal', async () => {
    const stuck = (pid) => Object.assign(new EventEmitter(), { pid, exitCode: null, signalCode: null, kill() { return true; } });
    const one = createHardwareQueryReaper({ reapMs: 30 });
    const two = createHardwareQueryReaper({ reapMs: 30 });
    const stop = new AbortController();
    const child = stuck(31001);
    const reading = one.runHardwareQuery(() => child, '/q', [], { signal: stop.signal });
    stop.abort();
    const result = await reading;
    assert.equal(result.aborted, true);
    assert.deepEqual(one.unreapedQueries(), [31001]);
    assert.deepEqual(two.unreapedQueries(), [], 'another instance does not see it');
    await two.stoppedQueriesSettled();
    child.emit('exit', null, 'SIGKILL');
    assert.deepEqual(one.unreapedQueries(), []);
    // No reaper given to a module-level reader: it still honors its signal and shares nothing.
    const free = new AbortController();
    const waiting = readGpu({ nvidiaSmi: '/q', execFileImpl: () => stuck(31002), env: {}, signal: free.signal });
    free.abort();
    await assert.rejects(waiting, { name: 'AbortError' });
    assert.deepEqual(one.unreapedQueries(), []);
});

// ------------------------------------------------- the controller's wiring

const DEDICATED = Object.freeze({
    gpu: { available: true, name: 'RTX', memoryModel: 'dedicated', totalBytes: 6 * GIB, freeBytes: 6 * GIB, usedBytes: 0, processes: [] },
    memory: { totalBytes: 32 * GIB, availableBytes: 28 * GIB },
    disk: { freeBytes: 400 * GIB },
});

function neutralController(t, { host, ...options } = {}) {
    const dataDir = host ? host.dataDir : tempDir(t, 'local-llm-seams-');
    const useHost = host ?? defineHost({ dataDir, runRoot: path.join(dataDir, 'run'), seedCatalogFile: SEED_CATALOG_FILE });
    const controller = createController({
        host: useHost, dataDir, env: { PATH: '/usr/bin' }, pollMs: 2, stopGraceMs: 50, ...options,
    });
    t.after(() => controller.drain());
    return { controller, dataDir, host: useHost };
}

test('the overview names the gateway model the caller gave and none by default; the not-ready text is the messages\'', async (t) => {
    const { controller } = neutralController(t, { snapshot: async () => structuredClone(DEDICATED) });
    assert.equal((await controller.overview()).gatewayModel, null);
    assert.throws(() => controller.chatTarget(), (error) => error.code === 'not_ready' && error.message === DEFAULT_MESSAGES.notReady());
    assert.doesNotMatch(DEFAULT_MESSAGES.notReady(), FORBIDDEN);
    const named = neutralController(t, { snapshot: async () => structuredClone(DEDICATED), gatewayModelName: 'gw/model/default', messages: defineMessages({ notReady: () => 'custom not ready' }) });
    assert.equal((await named.controller.overview()).gatewayModel, 'gw/model/default');
    assert.throws(() => named.controller.chatTarget(), { code: 'not_ready', message: 'custom not ready' });
    // The key keeps its place in the overview whatever its value.
    assert.deepEqual(Object.keys(await controller.overview()).slice(-3), ['models', 'deployment', 'gatewayModel']);
});

test('the controller hands its host, wording, reaper and platform to the hardware reader, and its reaper is the one the drain waits on', async (t) => {
    const gpuAccess = { report: () => ({ unavailable: false, reason: null }), source: 'test-source' };
    const platform = linuxPlatform();
    const dataDir = tempDir(t, 'local-llm-wire-');
    const host = defineHost({ dataDir, runRoot: path.join(dataDir, 'run'), seedCatalogFile: SEED_CATALOG_FILE, nvidiaSmi: '/fixture/nvidia-smi', gpuAccess, platform });
    const messages = defineMessages({ gpuAbsent: () => 'custom absent' });
    let options = null;
    const child = Object.assign(new EventEmitter(), { pid: 424242, exitCode: null, signalCode: null, kill() { return true; } });
    const { controller } = neutralController(t, { host, messages, readHardwareSnapshot: async (given) => {
        options = given;
        // A query that outlives its stop, started through the reaper the controller handed over.
        void given.reaper.runHardwareQuery(() => child, '/fixture/nvidia-smi', [], { signal: given.signal });
        return structuredClone(DEDICATED);
    } });
    await controller.overview();
    assert.equal(options.nvidiaSmi, '/fixture/nvidia-smi');
    assert.equal(options.gpuAccess, host.gpuAccess, 'the host\'s own frozen reader');
    assert.equal(options.gpuAccess.source, 'test-source');
    assert.equal(options.messages, messages);
    assert.equal(options.platform, platform);
    assert.equal(options.dataDir, dataDir);
    assert.equal(typeof options.reaper.runHardwareQuery, 'function');
    const began = Date.now();
    await controller.drain();
    assert.ok(Date.now() - began >= HARDWARE_QUERY_REAP_MS - 50, 'the drain waited for the query the reaper held');
    const log = (await controller.status()).logs.map((entry) => entry.line).join('\n');
    assert.match(log, /pid 424242\) was killed but has not exited/);
    child.emit('exit', null, 'SIGKILL');
});

function fixtureSmi(t, { name = 'RTX' } = {}) {
    const dir = tempDir(t, 'local-llm-smi-');
    const file = path.join(dir, 'nvidia-smi');
    fs.writeFileSync(file, `#!/bin/sh
case "$1" in
  --query-gpu=name*) echo "${name}, 6144, 0, 6144, 580.1" ;;
  --query-gpu=pci*) echo "0x123410DE, 8.6, none, 0, 5, 30" ;;
  *) ;;
esac
`, { mode: 0o755 });
    return file;
}

test('the default hardware reader uses the host\'s nvidia-smi, /proc and /dev/dri, and its GPU-access reader and wording when the GPU is missing', async (t) => {
    const proc = tempDir(t, 'local-llm-proc-');
    fs.writeFileSync(path.join(proc, 'meminfo'), 'MemTotal: 8388608 kB\nMemAvailable: 4194304 kB\nSwapFree: 0 kB\nMemFree: 2097152 kB\nCached: 1048576 kB\n');
    const dri = tempDir(t, 'local-llm-dri-');
    const platform = linuxPlatform({ procDir: proc, devDriDir: dri });
    const base = { dataDir: tempDir(t, 'local-llm-reader-'), platform };
    const hostWith = (overrides) => defineHost({ ...base, runRoot: path.join(base.dataDir, 'run'), seedCatalogFile: SEED_CATALOG_FILE, ...overrides });

    const present = neutralController(t, { host: hostWith({ nvidiaSmi: fixtureSmi(t) }) });
    const view = (await present.controller.overview()).hardware;
    assert.equal(view.gpu.available, true);
    assert.equal(view.gpu.name, 'RTX');
    assert.equal(view.memory.totalBytes, 8 * GIB, 'the memory came from the host\'s proc directory');

    const missing = neutralController(t, { host: hostWith({ nvidiaSmi: null }) });
    const gone = (await missing.controller.overview()).hardware.gpu;
    assert.equal(gone.state, 'absent');
    assert.equal(gone.reason, DEFAULT_MESSAGES.gpuAbsent());

    const reported = neutralController(t, {
        host: hostWith({ nvidiaSmi: null, gpuAccess: { report: (env) => ({ unavailable: env.GRANT_STATE === 'refused', reason: env.GRANT_WHY }), source: 'grant-env' } }),
        env: { PATH: '/usr/bin', GRANT_STATE: 'refused', GRANT_WHY: 'the grant\nwas refused' },
        messages: defineMessages({ gpuAccessReported: (reason) => `REPORTED(${reason})` }),
    });
    const refused = (await reported.controller.overview()).hardware.gpu;
    assert.equal(refused.reason, 'REPORTED(the grant was refused)');
    assert.equal(refused.diagnostic.source, 'grant-env');
    assert.equal(refused.diagnostic.code, 'gpu_access_unavailable');
});

test('readGpu and readSnapshot take their nvidia-smi, GPU-access reader and wording as input and have no defaults for them', async () => {
    const enoent = (command, args, options, callback) => callback(Object.assign(new Error('missing'), { code: 'ENOENT' }), '', '');
    const bare = await readGpu({ execFileImpl: enoent, env: { ANY: 'x' } });
    assert.equal(bare.state, 'absent');
    assert.equal(bare.reason, DEFAULT_MESSAGES.gpuAbsent());
    assert.equal(bare.diagnostic.source, 'agent-query');
    const ran = [];
    const asked = await readGpu({ execFileImpl: (command, args, options, callback) => { ran.push(command); callback(Object.assign(new Error('missing'), { code: 'ENOENT' }), '', ''); }, nvidiaSmi: '/given/nvidia-smi', env: {} });
    assert.deepEqual(ran, ['/given/nvidia-smi']);
    assert.equal(asked.state, 'absent');
    const fsApi = { readFileSync: () => 'MemTotal: 1024 kB\nMemAvailable: 512 kB\n', readdirSync: () => [], lstatSync: () => ({}) };
    const snapshot = await readSnapshot({ dataDir: '/nowhere', fsApi, execFileImpl: enoent, statfs: async () => ({ bsize: 1, blocks: 10, bavail: 5 }), env: {} });
    assert.equal(snapshot.gpu.state, 'absent');
    assert.equal(snapshot.memory.totalBytes, 1024 * 1024);
});

function fakeProcessRunner(record) {
    return {
        id: 'fake-process',
        displayName: 'Fake process runner',
        weightFormat: 'gguf',
        pinnedVersion: 'v1',
        supported: true,
        executable: process.execPath,
        port: 18998,
        apiKey: true,
        paramSchema: { type: 'object', additionalProperties: false, properties: {} },
        basicParams: [],
        normalizeParams: () => ({}),
        describeContext: () => ({ totalContext: 4096, perRequestContext: 4096, parallel: 1, kvUnified: false }),
        detect: () => ({ installed: true, version: 'v1', reason: null }),
        async start(ctx) {
            const launched = ctx.launch({ command: process.execPath, args: ['-e', 'setInterval(() => {}, 1000)'], env: {} });
            record.pid = launched.pid;
            await ctx.waitForHttp(`http://127.0.0.1:${ctx.port}/ready`, { process: launched });
            return {};
        },
        chatModel: (deployment) => deployment.modelId,
        admit: ({ model }) => ({ status: 'ok', reason: null, estimate: { gpuBytes: GIB, basis: 'fake', isEstimate: true }, warnings: [] }),
        parseReport: () => ({ device: null, totalMiB: 1 }),
    };
}

test('the default runner start marks the process through the host\'s platform and signals its group through the injected killImpl', async (t) => {
    const dataDir = tempDir(t, 'local-llm-process-');
    const marks = [];
    const kills = [];
    const platform = { ...linuxPlatform(), setOomScoreAdj: (pid, value) => marks.push([pid, value]) };
    const host = defineHost({ dataDir, runRoot: path.join(dataDir, 'run'), seedCatalogFile: SEED_CATALOG_FILE, platform });
    const record = {};
    const fake = fakeProcessRunner(record);
    const weights = path.join(dataDir, 'weights.gguf');
    fs.writeFileSync(weights, 'x');
    t.after(() => { try { process.kill(record.pid, 'SIGKILL'); } catch {} });
    const { controller } = neutralController(t, {
        host, dataDir,
        runners: { ...createRunners({ host }), [fake.id]: fake },
        snapshot: async () => structuredClone(DEDICATED),
        download: async ({ artifact }) => ({ status: 'complete', path: weights, bytesTransferred: 0, artifact }),
        inspect: async () => ({ state: 'complete', bytes: 1 }),
        verify: async () => ({ notes: [] }),
        detectRunner: () => ({ installed: true, version: 'v1', reason: null }),
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        apiKeyFactory: () => KEY,
        // Records every signal; only the final SIGKILL of the group also ends this test's own child, so the stop can finish.
        killImpl: (pid, signal) => { kills.push([pid, signal]); if (signal === 'SIGKILL' && pid === -record.pid) process.kill(record.pid, 'SIGKILL'); },
    });
    await controller.run({ modelId: 'gpt-oss-20b', runnerId: fake.id, requestId: 'request-seams-01' });
    for (let i = 0; i < 400 && controller.state.deployment?.phase !== 'ready'; i += 1) await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(controller.state.deployment?.phase, 'ready');
    assert.deepEqual(marks, [[record.pid, 1000]], 'the platform marked the new process');
    await controller.stop();
    assert.ok(kills.length >= 1 && kills.every(([pid]) => pid === -record.pid), JSON.stringify(kills));
    assert.equal(kills[0][1], 'SIGTERM');
});

// --------------------------------------------------- the runner installer

test('the installer takes its tools, run root and platform from its caller and refuses what it was not given', async (t) => {
    const lock = loadRunnerLocks({ image: null, agent: null });
    assert.throws(() => createRunnerInstaller({ lock }), code('invalid_host', 'cacheRoot'));
    const dir = tempDir(t, 'local-llm-installer-');
    const python = { schema: 'local-llm.runners-lock/v1', runners: {
        py: { version: '1', kind: 'python', licence: { name: 'MIT', url: 'https://example.org' },
            files: [{ name: 'x-1-py3-none-any.whl', url: 'https://files.pythonhosted.org/x-1-py3-none-any.whl', size: 1, sha256: 'a'.repeat(64) }] },
        ar: { version: '1', kind: 'archive', licence: { name: 'MIT', url: 'https://example.org' },
            files: [{ name: 'r.tar.gz', url: 'https://github.com/o/r/releases/download/v1/r.tar.gz', size: 10, sha256: 'b'.repeat(64) }] },
    } };
    const validated = runnerLockModule.validateRunnerLock(python);
    const cannot = createRunnerInstaller({ lock: validated, cacheRoot: path.join(dir, 'cache') });
    assert.equal(cannot.installable('py'), false, 'no uv or python: a Python runner is not installable');
    assert.equal(cannot.installable('ar'), true);
    assert.equal(createRunnerInstaller({ lock: validated, cacheRoot: path.join(dir, 'cache'), uv: '/u/uv', python: '/u/python3' }).installable('py'), true);
    const entry = cannot.entryFor('ar');
    assert.equal(cannot.pathsFor(entry).filesDir, path.join(dir, 'cache', 'ar', '1', 'files'));
    assert.throws(() => cannot.pathsFor(entry).runDir, code('invalid_host', 'runRoot'));
    const withRoot = createRunnerInstaller({ lock: validated, cacheRoot: path.join(dir, 'cache'), runRoot: path.join(dir, 'run') });
    assert.equal(withRoot.pathsFor(entry).runDir, path.join(dir, 'run', 'ar', '1'));
    assert.deepEqual(Object.keys(withRoot.pathsFor(entry)), ['idCache', 'idRun', 'cacheDir', 'filesDir', 'record', 'runDir', 'marker', 'tmpDir', 'stageDir']);
});

test('the install check takes its lock from its arguments and never from a default location, and reports instead of exiting', async (t) => {
    const dir = tempDir(t, 'local-llm-check-');
    const refused = await runInstallCheck(['llama.cpp']);
    assert.equal(refused.code, 1);
    assert.equal(refused.report.error, 'missing_file');
    const lockFile = path.join(dir, 'lock.json');
    fs.writeFileSync(lockFile, JSON.stringify({ schema: 'local-llm.runners-lock/v1', runners: {
        open: { version: '1', kind: 'archive', licence: { name: 'MIT', url: 'https://example.org' },
            files: [{ name: 'r.tar.gz', url: 'https://github.com/o/r/releases/download/v1/r.tar.gz', size: 10, sha256: 'a'.repeat(64) }] },
    } }));
    const checked = await runInstallCheck(['open', '--lock', lockFile, '--validate-only']);
    assert.equal(checked.code, 0);
    assert.equal(checked.report.ok, true);
    assert.match(checked.report.skipped, /validate-only/);
    const unknown = await runInstallCheck(['nothing', '--lock', lockFile]);
    assert.equal(unknown.code, 1);
    assert.equal(unknown.report.error, 'not_installable');
});

test('the seed catalog the controller loads is the host\'s file', async (t) => {
    const { controller } = neutralController(t, { snapshot: async () => structuredClone(DEDICATED) });
    const ids = (await controller.overview()).models.map((model) => model.id);
    const seedIds = loadSeedCatalog(SEED_CATALOG_FILE).map((model) => model.id);
    assert.ok(ids.length > 0 && ids.every((id) => seedIds.includes(id)), 'the models offered are the host\'s seed catalog entries');
    assert.ok(ids.includes('gpt-oss-20b'));
    // A host whose catalog file is not a catalog is refused where it is read.
    const dir = tempDir(t, 'local-llm-badcat-');
    const bad = path.join(dir, 'models.json');
    fs.writeFileSync(bad, JSON.stringify({ schema: 'other', models: [] }));
    assert.throws(() => createController({ host: defineHost({ dataDir: dir, runRoot: path.join(dir, 'run'), seedCatalogFile: bad }) }), /Unsupported catalog/);
    assert.equal(AGENT_HOST_SPEC.dataDir, '/data', 'the suite\'s agent-like host is the one in the table of the handoff');
});
