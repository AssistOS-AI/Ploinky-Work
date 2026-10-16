// The host on macOS (lib/local-llm-host/host.mjs with `platformName: 'darwin'`), on any machine: the support table, the host lock and
// runner registry per architecture, the readers it gives the engine, the permission chain of the home, the reaper on the `ps` table,
// the metal launch over a Unix socket, and the seed catalog. Every system tool is a fake (`darwinExec`, `execFileImpl`, `procs`), so
// nothing runs `sysctl`, `vm_stat`, `ps` or a runner here; the real-Mac probes are scripts outside the test globs.
import '../home.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog } from '../../lib/local-llm/controller/catalog.mjs';
import { METAL } from '../../lib/local-llm/controller/profiles.mjs';
import { catalogProfileOf } from '../../lib/local-llm/controller/vulkan.mjs';
import { LocalLlmError } from '../../lib/local-llm/errors.mjs';
import { validateRunnerLock } from '../../lib/local-llm/controller/runnerLock.mjs';
import { SEED_CATALOG_FILE, checkHomeLinks, checkPrivateDirs, createLocalLlmHost, hostLockFile, hostLockVersion } from '../../lib/local-llm-host/host.mjs';
import { LAUNCH_SCHEMA, readRecords } from '../../lib/local-llm-host/launch.mjs';
import { darwinPlatform } from '../../lib/local-llm/darwinPlatform.mjs';
import { defineHost } from '../../lib/local-llm/host.mjs';
import { createRunners } from '../../lib/local-llm/runners/index.mjs';
import { freePort, scratch } from './fixtures/engine.mjs';

const fixture = (name) => fs.readFileSync(new URL(`./fixtures/darwin/${name}`, import.meta.url), 'utf8');
const T0_EPOCH = '1792122134';

// The captured sysctl and vm_stat output of a 16 GiB M1, as the tools print it.
function capturedExec() {
    const known = new Map(`${fixture('sysctl.txt')}${fixture('swapusage.txt')}`.split('\n').filter(Boolean).map((line) => [line.slice(0, line.indexOf(':')), line]));
    return (file, args) => (file === '/usr/bin/vm_stat' ? fixture('vm_stat.txt') : `${args.filter((key) => known.has(key)).map((key) => known.get(key)).join('\n')}\n`);
}

const NO_PROCS = Object.freeze({ listPids: () => [], readStat: () => null, readCmdline: () => null });

// macOS lets a bind of 0.0.0.0:P succeed next to a listener on 127.0.0.1:P; Linux refuses it. The wildcard probe of the owner bind
// is therefore tested against a fake `net` in ownership-darwin.test.mjs and against the real kernel by the probe script
// ownership-probe.mjs on a Mac. Here the host runs its darwin code on any machine, so a bind of the wildcard address is answered as
// free and everything else is the real `net`.
const wildcardFreeNet = {
    ...net,
    createServer(...args) {
        const server = net.createServer(...args);
        const listen = server.listen.bind(server);
        server.listen = (options, callback) => (options?.host === '0.0.0.0' ? (setImmediate(callback), server) : listen(options, callback));
        return server;
    },
};
const bits = (file) => fs.statSync(file).mode & 0o777;

/** A Mac host on a fresh home with the system tools faked. `extra` overrides any option of createLocalLlmHost. */
async function macHost(t, { arch = 'arm64', home = scratch(t), extra = {} } = {}) {
    const logs = [];
    const ownerPort = extra.ownerPort ?? await freePort();
    const host = createLocalLlmHost({
        config: {}, home, env: { PATH: '/usr/bin:/bin', HOME: '/Users/test', LANG: 'en_US.UTF-8' }, platformName: 'darwin', arch, ownerPort,
        darwinExec: capturedExec(), procs: NO_PROCS, netImpl: wildcardFreeNet, log: (line) => logs.push(line), ...extra,
    });
    t.after(() => host.drain().catch(() => {}));
    return { host, home, logs, ownerPort };
}

// ------------------------------------------------------------------ support, locks, registry

test('macOS is supported on arm64 and x64, and nothing else is: another architecture or platform creates nothing', () => {
    for (const arch of ['arm64', 'x64']) assert.deepEqual(createLocalLlmHost({ home: '/h', platformName: 'darwin', arch }).supported, { ok: true, reason: null }, arch);
    assert.match(createLocalLlmHost({ home: '/h', platformName: 'darwin', arch: 'ppc64' }).supported.reason, /not supported on darwin-ppc64 in this release/);
    assert.match(createLocalLlmHost({ home: '/h', platformName: 'win32', arch: 'x64' }).supported.reason, /not supported on win32 in this release/);
    assert.deepEqual(createLocalLlmHost({ home: '/h', platformName: 'linux', arch: 'x64' }).supported, { ok: true, reason: null });
});

test('the host lock is chosen by platform and architecture; Linux stays the default', () => {
    const dir = path.dirname(hostLockFile('x64'));
    assert.equal(hostLockFile('x64'), path.join(dir, 'runners.lock.host-linux-amd64.json'));
    assert.equal(hostLockFile('arm64', 'linux'), path.join(dir, 'runners.lock.host-linux-arm64.json'));
    assert.equal(hostLockFile('arm64', 'darwin'), path.join(dir, 'runners.lock.host-darwin-arm64.json'));
    assert.equal(hostLockFile('x64', 'darwin'), path.join(dir, 'runners.lock.host-darwin-amd64.json'));
    assert.equal(hostLockFile('ia32', 'darwin'), null);
    assert.equal(hostLockFile('arm64', 'win32'), null);
    // Each lock holds exactly its one runner at the pinned build.
    for (const [arch, id, other] of [['arm64', 'llama.cpp', 'llama.cpp-cpu'], ['x64', 'llama.cpp-cpu', 'llama.cpp']]) {
        const lock = validateRunnerLock(JSON.parse(fs.readFileSync(hostLockFile(arch, 'darwin'), 'utf8')), { requireExecutable: true });
        assert.deepEqual(Object.keys(lock.runners), [id], arch);
        assert.equal(hostLockVersion(id, arch, { platformName: 'darwin' }), 'b11295');
        assert.equal(hostLockVersion(other, arch, { platformName: 'darwin' }), null, 'the other build is not in this lock');
        assert.equal(lock.runners[id].check.executable, 'llama-server');
    }
    assert.equal(hostLockVersion('llama.cpp', 'arm64'), 'b11295', 'no platform: the Linux lock');
});

test('arm64 offers the Metal build of llama.cpp, x64 its CPU build, and every other runner says it is not offered on macOS; no CUDA wording', async (t) => {
    const reasons = (description) => Object.fromEntries(description.runners.map((runner) => [runner.id, runner]));
    const arm = (await macHost(t, { arch: 'arm64' })).host.describe();
    const intel = (await macHost(t, { arch: 'x64' })).host.describe();
    assert.equal(reasons(arm)['llama.cpp'].supported, true);
    assert.equal(reasons(arm)['llama.cpp-cpu'].supported, false);
    assert.equal(reasons(intel)['llama.cpp-cpu'].supported, true);
    assert.equal(reasons(intel)['llama.cpp'].supported, false);
    assert.equal(arm.runners.length, 8);
    for (const description of [arm, intel]) {
        for (const runner of description.runners.filter((entry) => !entry.supported)) {
            assert.match(runner.reason, /is not offered by the standalone pworker on macOS in this release\.$/, runner.id);
            assert.doesNotMatch(runner.reason, /NVIDIA|CUDA|nvidia-smi|Linux/, runner.id);
        }
    }
    assert.equal(arm.runnerLockFile, hostLockFile('arm64', 'darwin'));
    assert.equal(intel.runnerLockFile, hostLockFile('x64', 'darwin'));
    assert.equal(arm.supported.ok, true);
    // The Linux wording stays Linux-only.
    const linux = createLocalLlmHost({ home: scratch(t), env: {}, platformName: 'linux', arch: 'x64', which: () => null }).describe();
    assert.match(linux.runners.find((runner) => runner.id === 'llama.cpp').reason, /NVIDIA GPU/);
    assert.match(linux.runners.find((runner) => runner.id === 'ollama').reason, /on Linux in this release/);
});

test('a Mac has no nvidia-smi: the engine environment does not look for it, even when one is on the PATH', async (t) => {
    const { host } = await macHost(t, { extra: { which: () => '/opt/gpu/bin/nvidia-smi' } });
    assert.deepEqual(host.engineEnvironment(), { PATH: '/usr/local/bin:/usr/bin:/bin', HOME: '/Users/test', LANG: 'en_US.UTF-8' });
    assert.equal(host.runnerVersion('llama.cpp'), 'b11295');
    assert.equal(host.runnerVersion('llama.cpp-cpu'), null, 'not in the arm64 lock');
});

// ------------------------------------------------------------------ the controller on a Mac

test('arm64: the controller commits to the metal profile and reads the memory of the Mac with the macOS readers; the runner is installable from the darwin lock', async (t) => {
    const { host } = await macHost(t);
    const controller = await host.controller();
    const overview = await controller.overview();
    assert.equal(overview.profile, 'metal');
    assert.equal(overview.hardware.memory.totalBytes, 17_179_869_184);
    assert.equal(overview.hardware.memory.availableBytes, 2_889_121_792);
    assert.equal(overview.hardware.memory.pressureLevel, 2);
    assert.equal(overview.hardware.gpu.backend, 'metal');
    assert.equal(overview.hardware.gpu.deviceCapBytes, null, 'the runner is not installed, so there is no device limit and admission refuses');
    assert.equal(overview.hardware.cores, 4, 'the performance cores');
    const byId = Object.fromEntries(overview.runners.map((runner) => [runner.id, runner]));
    assert.equal(byId['llama.cpp'].pinnedVersion, 'b11295', 'the host lock, not the agent\'s build');
    assert.equal(byId['llama.cpp'].install.version, 'b11295');
    assert.equal(byId['llama.cpp'].install.totalBytes, 11_817_829);
    assert.equal(byId['llama.cpp'].installed, false);
    assert.equal(Object.hasOwn(byId, 'llama.cpp-cpu'), false);
    assert.deepEqual(overview.models.map((model) => model.id), ['qwen2.5-0.5b-instruct-q4_k_m', 'qwen3-0.6b'], 'both seed models are offered on the metal profile');
});

test('x64: the controller commits to the cpu profile with no GPU, the Intel Mac\'s memory and the CPU build', async (t) => {
    const { host } = await macHost(t, { arch: 'x64' });
    const overview = await (await host.controller()).overview();
    assert.equal(overview.profile, 'cpu');
    assert.equal(overview.hardware.gpu.available, false);
    assert.equal(overview.hardware.memory.totalBytes, 17_179_869_184, 'admission reads the macOS figures, not /proc/meminfo');
    const byId = Object.fromEntries(overview.runners.map((runner) => [runner.id, runner]));
    assert.equal(byId['llama.cpp-cpu'].install.version, 'b11295');
    assert.equal(byId['llama.cpp-cpu'].install.totalBytes, 11_372_934);
    assert.deepEqual(overview.models.map((model) => model.id), ['qwen2.5-0.5b-instruct-q4_k_m', 'qwen3-0.6b']);
    // A Run of the Metal build is refused with the host's own reason, and nothing is downloaded.
    await assert.rejects((await host.controller()).run({ modelId: 'qwen3-0.6b', runnerId: 'llama.cpp', requestId: 'mac-x64-run-0001' }), { code: 'runner_unsupported', message: /not offered by the standalone pworker on macOS/ });
});

test('controllerOptions still override the macOS seams', async (t) => {
    const { host } = await macHost(t, { extra: { controllerOptions: {
        profile: 'cpu', readMemory: () => ({ totalBytes: 1, availableBytes: 1 }),
        snapshot: async () => ({ gpu: { available: false }, memory: { totalBytes: 8 * 1024 ** 3, availableBytes: 4 * 1024 ** 3 }, disk: { freeBytes: 1024 ** 4, totalBytes: 1024 ** 4 }, cpus: 4 }),
    } } });
    const overview = await (await host.controller()).overview();
    assert.equal(overview.profile, 'cpu');
    assert.equal(overview.hardware.memory.totalBytes, 8 * 1024 ** 3);
});

test('the Metal device comes from the installed runner\'s --list-devices, run with the filtered environment; a runner that is not installed is not run', async (t) => {
    const calls = [];
    const execFileImpl = (file, args, options, callback) => {
        calls.push({ file, args, options });
        setImmediate(() => callback(null, fixture('list-devices.txt'), ''));
    };
    const { host } = await macHost(t, { extra: { execFileImpl, env: { PATH: '/usr/bin:/bin', HOME: '/Users/test', LANG: 'en_US.UTF-8', HF_TOKEN: 'hf_synthetic_probe_token', OPENAI_API_KEY: 'sk-secret' } } });
    const controller = await host.controller();
    assert.equal((await controller.overview()).hardware.gpu.deviceCapBytes, null);
    assert.deepEqual(calls, [], 'nothing is installed, so nothing is run');
    // The installer's copy: the program and the marker it writes last.
    const copy = path.join(host.layout.run, 'llama.cpp', 'b11295');
    fs.mkdirSync(copy, { recursive: true });
    fs.writeFileSync(path.join(copy, 'llama-server'), '#!/bin/sh\n', { mode: 0o755 });
    assert.equal((await controller.overview()).hardware.gpu.deviceCapBytes, null, 'a copy without the marker is not installed');
    assert.deepEqual(calls, []);
    fs.writeFileSync(path.join(copy, '.ready.json'), '{}\n');
    const gpu = (await controller.overview()).hardware.gpu;
    assert.equal(gpu.deviceCapBytes, 12_712_935_424, '12124 MiB');
    assert.equal(gpu.name, 'Apple M1');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].file, path.join(copy, 'llama-server'));
    assert.deepEqual(calls[0].args, ['--list-devices']);
    assert.equal(calls[0].options.cwd, host.layout.engineHome);
    assert.ok(Number.isInteger(calls[0].options.timeout) && calls[0].options.timeout > 0 && calls[0].options.timeout <= 60_000);
    assert.deepEqual(Object.keys(calls[0].options.env).sort(), ['HOME', 'LANG', 'PATH'], 'the engine\'s environment without the token: no provider key either');
    assert.equal(calls[0].options.env.HF_TOKEN, undefined);
    assert.equal(calls[0].options.env.OPENAI_API_KEY, undefined);
    // The answer is remembered until iogpu.wired_limit_mb changes.
    await controller.overview();
    assert.equal(calls.length, 1);
});

test('a runner that cannot be run for the device leaves no device limit, and is asked again at the next snapshot', async (t) => {
    let attempts = 0;
    const execFileImpl = (file, args, options, callback) => { attempts += 1; setImmediate(() => callback(Object.assign(new Error('Command failed'), { code: 1 }), '', '')); };
    const { host } = await macHost(t, { extra: { execFileImpl } });
    const controller = await host.controller();
    const copy = path.join(host.layout.run, 'llama.cpp', 'b11295');
    fs.mkdirSync(copy, { recursive: true });
    fs.writeFileSync(path.join(copy, 'llama-server'), '#!/bin/sh\n', { mode: 0o755 });
    fs.writeFileSync(path.join(copy, '.ready.json'), '{}\n');
    assert.equal((await controller.overview()).hardware.gpu.deviceCapBytes, null);
    assert.equal((await controller.overview()).hardware.gpu.deviceCapBytes, null);
    assert.equal(attempts, 2);
});

// ------------------------------------------------------------------ the permission chain

const refusal = (host) => host.acquire().then(() => assert.fail('the home was accepted'), (error) => error);
// `dir` at `mode`, every directory made on the way to it 0700, whatever the umask is.
function looseDir(dir, mode = 0o777) {
    const made = [];
    for (let current = dir; !fs.existsSync(current); current = path.dirname(current)) made.unshift(current);
    for (const level of made) fs.mkdirSync(level, { mode: 0o700 });
    for (const level of made) fs.chmodSync(level, level === dir ? mode : 0o700);
}

test('a home that others can write is refused before anything is created under it, and the owner port is not taken', async (t) => {
    const holder = scratch(t);
    fs.chmodSync(holder, 0o777);
    const { host, ownerPort } = await macHost(t, { home: holder });
    const error = await refusal(host);
    assert.equal(error.code, 'invalid_host');
    assert.ok(error.message.includes(holder), error.message);
    assert.match(error.message, /can be written by its group or by others/);
    assert.match(error.message, new RegExp(`chmod go-w ${holder}`));
    assert.deepEqual(fs.readdirSync(holder), [], 'nothing was created in the loose directory');
    assert.equal(host.describe().ownership.state, 'refused');
    // The refusal is remembered, for acquire and for the engine.
    await assert.rejects(host.acquire(), { code: 'invalid_host' });
    await assert.rejects(host.controller(), { code: 'invalid_host' });
    // And the port was never bound: this process takes it.
    const probe = net.createServer();
    await new Promise((resolve, reject) => { probe.once('error', reject); probe.listen(ownerPort, '127.0.0.1', resolve); });
    await new Promise((resolve) => probe.close(resolve));
});

test('an engine directory that others can write is refused and nothing is created inside it', async (t) => {
    const home = scratch(t);
    const engine = path.join(home, 'local-llm');
    looseDir(engine);
    const { host } = await macHost(t, { home });
    const error = await refusal(host);
    assert.equal(error.code, 'invalid_host');
    assert.ok(error.message.includes(engine), error.message);
    assert.deepEqual(fs.readdirSync(engine), []);
    assert.deepEqual(fs.readdirSync(home), ['local-llm']);
});

test('a layout directory that is a link, writable by its group, or not the user\'s own is refused (R-2: the run root too)', async (t) => {
    for (const name of ['models', 'runners', 'run', 'state', 'logs', 'home']) {
        const group = scratch(t);
        looseDir(path.join(group, 'local-llm', name), 0o770);
        const refusedGroup = await refusal((await macHost(t, { home: group })).host);
        assert.equal(refusedGroup.code, 'invalid_host', name);
        assert.ok(refusedGroup.message.includes(path.join(group, 'local-llm', name)), name);
        assert.match(refusedGroup.message, /group or by others/, name);

        const linked = scratch(t);
        const elsewhere = path.join(scratch(t), 'elsewhere');
        looseDir(elsewhere, 0o700);
        fs.mkdirSync(path.join(linked, 'local-llm'), { mode: 0o700 });
        fs.symlinkSync(elsewhere, path.join(linked, 'local-llm', name));
        const refusedLink = await refusal((await macHost(t, { home: linked })).host);
        assert.equal(refusedLink.code, 'invalid_host', name);
        assert.match(refusedLink.message, /is a symbolic link/, name);
        assert.deepEqual(fs.readdirSync(elsewhere), [], `${name}: nothing is created through the link`);
    }
    // The launch directory is part of the state directory and is checked as well.
    const launch = scratch(t);
    looseDir(path.join(launch, 'local-llm', 'state', 'launch'));
    assert.match((await refusal((await macHost(t, { home: launch })).host)).message, /launch/);
    // The link in the middle of the engine directory itself.
    const middle = scratch(t);
    const target = scratch(t);
    fs.symlinkSync(target, path.join(middle, 'local-llm'));
    assert.match((await refusal((await macHost(t, { home: middle })).host)).message, /is a symbolic link/);
    assert.deepEqual(fs.readdirSync(target), []);
});

test('a directory of another user is refused: the owner is compared with the user id of the process', async (t) => {
    const { host } = await macHost(t, { extra: { uid: process.getuid() + 1 } });
    const error = await refusal(host);
    assert.equal(error.code, 'invalid_host');
    assert.match(error.message, /is owned by user \d+, not by this user/);
    // Without a user id nothing can be said about the owner, and nothing is accepted.
    const { host: unknown } = await macHost(t, { extra: { uid: null } });
    assert.match((await refusal(unknown)).message, /user id is unknown/);
});

test('a home that is a link to a real directory is resolved first and accepted; the usual modes 0700, 0750 and 0755 are accepted', async (t) => {
    const real = scratch(t);
    const link = path.join(scratch(t), 'home-link');
    fs.symlinkSync(real, link);
    const { host } = await macHost(t, { home: link });
    assert.equal((await host.acquire()).owned, true);
    for (const dir of ['models', 'runners', 'run', 'state', 'logs', 'home']) assert.equal(bits(path.join(real, 'local-llm', dir)), 0o700, dir);
    await host.drain();
    for (const mode of [0o700, 0o750, 0o755]) {
        const home = scratch(t);
        fs.chmodSync(home, mode);
        const { host: other } = await macHost(t, { home });
        assert.equal((await other.acquire()).owned, true, mode.toString(8));
        await other.drain();
    }
});

// A holder directory at `mode` (whatever the umask is) with a link in it to a fresh real 0700 home; returns { holder, link, real }.
function linkedHome(t, mode, { name = 'store' } = {}) {
    const holder = scratch(t);
    fs.chmodSync(holder, mode);
    const real = scratch(t);
    const link = path.join(holder, name);
    fs.symlinkSync(real, link);
    return { holder, link, real };
}

test('a home that is a link inside a directory others can write and that is not sticky is refused, naming the link, its holder and the real path; nothing is created', async (t) => {
    for (const mode of [0o777, 0o775, 0o772]) {
        const { holder, link, real } = linkedHome(t, mode);
        const { host } = await macHost(t, { home: link });
        const error = await refusal(host);
        assert.equal(error.code, 'invalid_host', mode.toString(8));
        assert.ok(error.message.includes(link) && error.message.includes(holder), error.message);
        assert.match(error.message, /can be written by its group or by others and is not sticky/);
        assert.ok(error.message.includes(`Set PWORKER_HOME to the real path ${real}`), error.message);
        assert.deepEqual(fs.readdirSync(real), [], `${mode.toString(8)}: nothing was created through the link`);
        await assert.rejects(host.acquire(), { code: 'invalid_host' }, 'the refusal is remembered');
    }
    // A link in the middle of the path counts as well: the home is below it.
    const middle = linkedHome(t, 0o777);
    fs.mkdirSync(path.join(middle.real, 'sub'), { mode: 0o700 });
    const { host: below } = await macHost(t, { home: path.join(middle.link, 'sub') });
    assert.equal((await refusal(below)).code, 'invalid_host');
    assert.deepEqual(fs.readdirSync(path.join(middle.real, 'sub')), []);
});

test('a home link in a sticky holder (mode 1777) or in a directory only the user can write is accepted; a holder of another user is not', async (t) => {
    for (const mode of [0o1777, 0o1775, 0o755, 0o750, 0o700]) {
        const { link, real } = linkedHome(t, mode);
        const { host } = await macHost(t, { home: link });
        assert.equal((await host.acquire()).owned, true, mode.toString(8));
        assert.equal(bits(path.join(real, 'local-llm')), 0o700, mode.toString(8));
        await host.drain();
    }
    // The holder is the user's own for the file system, but this process says it is another user: the link could be replaced.
    const { link } = linkedHome(t, 0o755);
    const { host: other } = await macHost(t, { home: link, extra: { uid: process.getuid() + 1 } });
    const error = await refusal(other);
    assert.equal(error.code, 'invalid_host');
    assert.match(error.message, /is owned by user \d+, not by this user \(\d+\) or root/);
});

test('only links are checked: a real home inside a directory others can write is accepted as it was, and Linux accepts a home link in such a directory', async (t) => {
    const holder = scratch(t);
    fs.chmodSync(holder, 0o777);
    const home = path.join(holder, 'home');
    fs.mkdirSync(home, { mode: 0o700 });
    const { host } = await macHost(t, { home });
    assert.equal((await host.acquire()).owned, true);
    await host.drain();
    const { link, real } = linkedHome(t, 0o777);
    const linux = createLocalLlmHost({ config: {}, home: link, env: {}, platformName: 'linux', arch: 'x64', ownerPort: await freePort(), procDir: path.join(real, 'no-proc') });
    t.after(() => linux.drain().catch(() => {}));
    assert.equal((await linux.acquire()).owned, true);
});

test('checkHomeLinks: the holder may be the user or root, a dangling link is left to the walk of the resolved home, and a missing path is not a link', (t) => {
    const uid = process.getuid();
    const { link, holder } = linkedHome(t, 0o755);
    assert.doesNotThrow(() => checkHomeLinks(link, { uid }));
    assert.doesNotThrow(() => checkHomeLinks(path.join(holder, 'missing', 'deeper'), { uid }));
    // A holder owned by root is accepted whoever the user is (/ holds /var and /tmp).
    assert.doesNotThrow(() => checkHomeLinks('/var/tmp', { uid }));
    assert.throws(() => checkHomeLinks(link, { uid: null }), /user id is unknown/);
    const dangling = path.join(holder, 'dangling');
    fs.symlinkSync(path.join(holder, 'nowhere'), dangling);
    assert.doesNotThrow(() => checkHomeLinks(dangling, { uid }));
    fs.chmodSync(holder, 0o777);
    assert.throws(() => checkHomeLinks(dangling, { uid }), (error) => error.code === 'invalid_host' && /Set PWORKER_HOME to the real path\./.test(error.message));
});

test('the check runs on macOS only: on Linux a loose home and a linked layout directory are accepted as they always were', async (t) => {
    const home = scratch(t);
    fs.chmodSync(home, 0o777);
    const target = scratch(t);
    fs.mkdirSync(path.join(home, 'local-llm'));
    fs.symlinkSync(target, path.join(home, 'local-llm', 'models'));
    const ownerPort = await freePort();
    const linux = createLocalLlmHost({ config: {}, home, env: {}, platformName: 'linux', arch: 'x64', ownerPort, procDir: path.join(home, 'no-proc') });
    t.after(() => linux.drain().catch(() => {}));
    assert.equal((await linux.acquire()).owned, true);
});

test('checkPrivateDirs: a missing directory is skipped or refused as asked, a file is not a directory, and a good directory passes', (t) => {
    const root = scratch(t);
    const good = path.join(root, 'good');
    fs.mkdirSync(good, { mode: 0o700 });
    const file = path.join(root, 'file');
    fs.writeFileSync(file, 'x');
    const uid = process.getuid();
    assert.doesNotThrow(() => checkPrivateDirs([root, good], { uid }));
    assert.doesNotThrow(() => checkPrivateDirs([good, path.join(root, 'missing')], { uid, requireAll: false }));
    assert.throws(() => checkPrivateDirs([path.join(root, 'missing')], { uid }), (error) => error instanceof LocalLlmError && error.code === 'invalid_host' && /cannot be read \(ENOENT\)/.test(error.message));
    assert.throws(() => checkPrivateDirs([file], { uid }), /is not a directory/);
    assert.throws(() => checkPrivateDirs([good], { uid: uid + 1 }), /owned by user/);
});

// ------------------------------------------------------------------ the reaper on a Mac

test('acquire on a Mac reaps an orphan found through the ps table', async (t) => {
    const home = scratch(t);
    const launchDir = path.join(home, 'local-llm', 'state', 'launch');
    const socketPath = path.join(home, 'r', '16161616.sock');
    fs.mkdirSync(launchDir, { recursive: true, mode: 0o700 });
    fs.writeFileSync(path.join(launchDir, '16161616.json'), `${JSON.stringify({ schema: LAUNCH_SCHEMA, startId: '16161616', socketPath, phase: 'spawned', pid: 7100, startTime: T0_EPOCH })}\n`);
    const alive = new Map([[7100, { state: 'S', pgrp: 7100, startTime: T0_EPOCH, command: `/run/llama.cpp/b11295/llama-server -m /m/q.gguf --host ${socketPath} --api-key k` }]]);
    const procs = {
        listPids: () => [...alive.keys()],
        readStat: (pid) => (alive.has(pid) ? { state: alive.get(pid).state, pgrp: alive.get(pid).pgrp, startTime: alive.get(pid).startTime } : null),
        readCmdline: (pid) => alive.get(pid)?.command ?? null,
    };
    const signals = [];
    const killImpl = (pid, signal) => { signals.push([pid, signal]); alive.delete(-pid); };
    const { host } = await macHost(t, { home, extra: { procs, killImpl } });
    const result = await host.acquire();
    assert.deepEqual(signals, [[-7100, 'SIGTERM']]);
    assert.deepEqual(result.reaped.reaped, ['16161616']);
    assert.deepEqual(readRecords(launchDir), []);
});

// ------------------------------------------------------------------ the metal launch over a socket

test('the metal launch with a socket has --host <socket> and no --port, and the metal tuning arguments are the ones they were', () => {
    const socketPath = '/Users/u/pw/r/abcd1234.sock';
    const catalog = loadSeedCatalog(SEED_CATALOG_FILE);
    const model = catalog.find((entry) => entry.id === 'qwen3-0.6b');
    // The runner of a Mac host: no executable of its own, the runnable copy of the lock's install.
    const engineHost = defineHost({ dataDir: '/x/data', runRoot: '/x/run', platform: darwinPlatform({ exec: () => '4\n' }), listen: { socketDir: '/Users/u/pw/r', pathHint: 'x' } });
    const runner = createRunners({ host: engineHost })['llama.cpp'];
    const launch = runner.buildLaunch({
        artifactPath: '/models/m.gguf', params: { ctxSize: 2048 }, port: 18080, apiKey: 'k'.repeat(40), model, profile: 'metal', runnerDir: '/x/run/llama.cpp/b11295', listen: { socketPath },
    });
    assert.equal(launch.command, '/x/run/llama.cpp/b11295/llama-server');
    const host = launch.args.indexOf('--host');
    assert.equal(launch.args[host + 1], socketPath);
    assert.equal(launch.args.includes('--port'), false);
    assert.equal(launch.args.includes('127.0.0.1'), false);
    const tail = launch.args.slice(launch.args.indexOf('-c'));
    assert.deepEqual([tail.slice(0, 10), tail[10], tail.slice(12)], [
        ['-c', '2048', '-ngl', '999', '-fa', 'auto', '-ctk', 'f16', '-ctv', 'f16'], '--threads', ['-np', '1', '--cache-ram', String(METAL.cacheRamMiB)],
    ]);
    assert.deepEqual(launch.env, {});
    // Without a socket the same launch is the loopback one it was.
    const tcp = runner.buildLaunch({ artifactPath: '/models/m.gguf', params: { ctxSize: 2048 }, port: 18080, apiKey: 'k'.repeat(40), model, profile: 'metal', runnerDir: '/r' });
    assert.deepEqual(tcp.args.slice(tcp.args.indexOf('--host'), tcp.args.indexOf('--host') + 4), ['--host', '127.0.0.1', '--port', '18080']);
});

// ------------------------------------------------------------------ the seed catalog

test('the seed catalog has qwen3-0.6b, pinned to its commit, sizes and sha256, and both models are offered on the metal and cpu profiles', () => {
    const catalog = loadSeedCatalog(SEED_CATALOG_FILE);
    assert.deepEqual(catalog.map((entry) => entry.id), ['qwen2.5-0.5b-instruct-q4_k_m', 'qwen3-0.6b']);
    const qwen3 = catalog.find((entry) => entry.id === 'qwen3-0.6b');
    assert.equal(qwen3.license, 'Apache-2.0');
    assert.equal(qwen3.contextLength, 40960);
    assert.deepEqual(qwen3.sources.gguf, {
        type: 'huggingface', repo: 'Qwen/Qwen3-0.6B-GGUF', file: 'Qwen3-0.6B-Q8_0.gguf', revision: '23749fefcc72300e3a2ad315e1317431b06b590a',
        commit: '23749fefcc72300e3a2ad315e1317431b06b590a', size: 639_446_688, sha256: '9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031', quantization: 'Q8_0',
    });
    // 2 (K and V) x 28 layers x 8 KV heads x 128 head dimensions x 2 bytes (f16), from the model's config.json.
    assert.equal(qwen3.memory.layers, 28);
    assert.equal(qwen3.memory.kvBytesPerToken, 2 * 28 * 8 * 128 * 2);
    assert.equal(qwen3.memory.kvBytesPerToken, 114_688);
    assert.equal(qwen3.recommended.metal['llama.cpp'].ctxSize, 2048);
    // `metal` is not a catalog profile (the catalog maps it to cpu), so the models are listed on it through `cpu`.
    assert.equal(catalogProfileOf('metal'), 'cpu');
    for (const entry of catalog) assert.ok(entry.profiles.includes(catalogProfileOf('metal')), entry.id);
    assert.ok(qwen3.profiles.includes('cpu'));
});

// ------------------------------------------------------------------ the socket path limit

test('the socket path limit on a Mac is 103 bytes: a socket path of 104 is refused before anything is downloaded, one of 103 is not', async (t) => {
    const base = scratch(t);
    // `<dir>/00000000.sock` is the longest socket path of a start; the directory is padded so that the whole path is `length` bytes.
    const socketDirFor = (length) => path.join(base, 'x'.repeat(length - '/00000000.sock'.length - base.length - 1));
    for (const [length, refused] of [[104, true], [103, false]]) {
        const socketDir = socketDirFor(length);
        assert.equal(Buffer.byteLength(path.join(socketDir, '00000000.sock')), length);
        const { host } = await macHost(t, { home: scratch(t), extra: { config: { localLlm: { socketDir } } } });
        const attempt = (await host.controller()).run({ modelId: 'qwen3-0.6b', runnerId: 'llama.cpp', requestId: `mac-socket-${length}` });
        if (refused) {
            await assert.rejects(attempt, (error) => error.code === 'socket_path_too_long' && error.details.limit === 103 && error.details.length === 104 && /shorter PWORKER_HOME/.test(error.message));
        } else {
            // It fits; the run stops at the next check (the runner is not installed), not at the path.
            await assert.rejects(attempt, (error) => error.code !== 'socket_path_too_long');
        }
    }
});
