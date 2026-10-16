// The host on macOS (lib/local-llm-host/host.mjs with `platformName: 'darwin'`), on any machine: the support table, the host lock and
// runner registry per architecture, the readers it gives the engine, the permission chain of the home, the reaper on the `ps` table,
// the metal launch over a Unix socket, and the seed catalog. Every system tool is a fake (`darwinExec`, `execFileImpl`, `procs`), so
// nothing runs `sysctl`, `vm_stat`, `ps` or a runner here; the real-Mac probes are scripts outside the test globs.
import '../home.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog } from '../../lib/local-llm/controller/catalog.mjs';
import { METAL } from '../../lib/local-llm/controller/profiles.mjs';
import { catalogProfileOf } from '../../lib/local-llm/controller/vulkan.mjs';
import { LocalLlmError } from '../../lib/local-llm/errors.mjs';
import { validateRunnerLock } from '../../lib/local-llm/controller/runnerLock.mjs';
import { SEED_CATALOG_FILE, checkHomePath, checkPrivateDirs, createLocalLlmHost, hostLockFile, hostLockVersion } from '../../lib/local-llm-host/host.mjs';
import { LAUNCH_SCHEMA, readRecords } from '../../lib/local-llm-host/launch.mjs';
import { darwinPlatform } from '../../lib/local-llm/darwinPlatform.mjs';
import { defineHost } from '../../lib/local-llm/host.mjs';
import { createRunners } from '../../lib/local-llm/runners/index.mjs';
import { freePort } from './fixtures/engine.mjs';

// The product rule holds every directory on the way to the home to its holder, `/tmp` and the ancestors of the machine's TMPDIR
// included, and those differ from machine to machine (a TMPDIR under a group-writable, non-sticky directory is common on Linux). So
// each test builds its homes below a directory of its own, made here with mode 0700, and hands it to the host as `homeCheckRoot`:
// the walk starts there and trusts it, and everything below it is checked as in production. What depends on the real `/` is tested
// on a fake file system (memFs) or by the real-layout probe script.
const roots = new WeakMap();
function rootOf(t) {
    if (!roots.has(t)) {
        const dir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'dh-'));
        fs.chmodSync(dir, 0o700);
        t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
        roots.set(t, dir);
    }
    return roots.get(t);
}
/** A fresh 0700 directory below the test's root. */
function scratch(t) {
    const dir = fs.mkdtempSync(path.join(rootOf(t), 'h-'));
    fs.chmodSync(dir, 0o700);
    return dir;
}

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
        darwinExec: capturedExec(), procs: NO_PROCS, netImpl: wildcardFreeNet, homeCheckRoot: rootOf(t), log: (line) => logs.push(line), ...extra,
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
    assert.match(error.message, /is owned by user \d+ .*not by this user/);
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

// ------------------------------------------------------------------ the safe-path walk of the home (checkHomePath)

// A real directory under `parent` named `name` at `mode`, whatever the umask is.
function dirAt(parent, name, mode = 0o700) {
    const dir = path.join(parent, name);
    fs.mkdirSync(dir, { mode: 0o700 });
    fs.chmodSync(dir, mode);
    return dir;
}

// The real file system, except that the owner of the given paths (a Map of path -> uid) is as told: other users' files without another user.
function fsWithOwners(owners) {
    return {
        ...fs,
        lstatSync(file, options) {
            const stat = fs.lstatSync(file, options);
            return owners.has(file) ? Object.assign(Object.create(Object.getPrototypeOf(stat)), stat, { uid: owners.get(file) }) : stat;
        },
    };
}

const UID = process.getuid();
const walk = (t, home, extra = {}) => checkHomePath(home, { uid: UID, root: rootOf(t), ...extra });
const refused = (pattern) => (error) => error.code === 'invalid_host' && pattern.test(error.message);

test('checkHomePath: two links in a row through a holder others can write are refused (the second link\'s holder is the first link\'s target, not the link)', async (t) => {
    const top = scratch(t);
    const loose = dirAt(top, 'w', 0o777);
    const real = dirAt(top, 'real');
    fs.symlinkSync(real, path.join(loose, 'L2'));
    fs.symlinkSync(loose, path.join(top, 'L1'));
    const home = path.join(top, 'L1', 'L2');
    assert.throws(() => walk(t, home), refused(/cannot be used: .*w, which holds .*L2, can be written by its group or by others and is not sticky \(mode 0777\)/));
    const { host } = await macHost(t, { home });
    assert.equal((await refusal(host)).code, 'invalid_host');
    assert.deepEqual(fs.readdirSync(real), [], 'nothing was created through the links');
    // The same two links through holders only the user can write are accepted.
    const tidy = scratch(t);
    const target = dirAt(tidy, 'w', 0o755);
    const home2 = dirAt(tidy, 'real');
    fs.symlinkSync(home2, path.join(target, 'L2'));
    fs.symlinkSync(target, path.join(tidy, 'L1'));
    assert.doesNotThrow(() => walk(t, path.join(tidy, 'L1', 'L2')));
});

test('checkHomePath: a link in a holder that others can write is refused, naming the link, its owner and mode, and the holder; a sticky holder or the user\'s own directory is accepted', async (t) => {
    for (const mode of [0o777, 0o775, 0o772]) {
        const top = scratch(t);
        const holder = dirAt(top, 'holder', mode);
        const real = dirAt(top, 'real');
        const link = path.join(holder, 'store');
        fs.symlinkSync(real, link);
        assert.throws(() => walk(t, link), refused(new RegExp(`${holder.replace(/[/.]/g, '\\$&')}, which holds ${link.replace(/[/.]/g, '\\$&')}, can be written by its group or by others and is not sticky`)), mode.toString(8));
        const { host } = await macHost(t, { home: link });
        const error = await refusal(host);
        assert.equal(error.code, 'invalid_host');
        assert.match(error.message, /Set PWORKER_HOME to a directory under one that only you can write/);
        assert.deepEqual(fs.readdirSync(real), [], `${mode.toString(8)}: nothing was created through the link`);
        await assert.rejects(host.acquire(), { code: 'invalid_host' }, 'the refusal is remembered');
    }
    for (const mode of [0o1777, 0o1775, 0o755, 0o750, 0o700]) {
        const top = scratch(t);
        const holder = dirAt(top, 'holder', mode);
        const real = dirAt(top, 'real');
        fs.symlinkSync(real, path.join(holder, 'store'));
        const { host } = await macHost(t, { home: path.join(holder, 'store') });
        assert.equal((await host.acquire()).owned, true, mode.toString(8));
        assert.equal(bits(path.join(real, 'local-llm')), 0o700);
        await host.drain();
    }
});

test('checkHomePath: a link or a directory that another user owns is refused even in a sticky holder, because its owner can retarget or rename it', (t) => {
    const top = scratch(t);
    const sticky = dirAt(top, 'sticky', 0o1777);
    const real = dirAt(top, 'real');
    const link = path.join(sticky, 'L');
    fs.symlinkSync(real, link);
    const own = dirAt(sticky, 'dir');
    assert.doesNotThrow(() => walk(t, link));
    assert.doesNotThrow(() => walk(t, own));
    assert.throws(() => walk(t, link, { fsApi: fsWithOwners(new Map([[link, UID + 4242]])) }), refused(/L, a symbolic link in .*sticky, is owned by user \d+ \(mode 0\d{3}\), not by this user \(\d+\) or root/));
    assert.throws(() => walk(t, own, { fsApi: fsWithOwners(new Map([[own, UID + 4242]])) }), refused(/dir, a directory in .*sticky, is owned by user \d+/));
    // A directory of another user on the way is refused, root's is fine.
    assert.throws(() => walk(t, own, { fsApi: fsWithOwners(new Map([[sticky, UID + 4242]])) }), refused(/sticky, a directory in .*, is owned by user \d+/));
    const everyoneIsRoot = { ...fs, lstatSync: (file, options) => Object.assign(Object.create(Object.getPrototypeOf(fs.lstatSync(file, options))), fs.lstatSync(file, options), { uid: 0 }) };
    assert.doesNotThrow(() => walk(t, own, { uid: UID + 1, fsApi: everyoneIsRoot }), 'root-owned files and holders are accepted whoever the user is');
    assert.throws(() => walk(t, own, { uid: UID + 1 }), refused(/is owned by user/), 'and the user\'s own are not when the user is someone else');
});

test('checkHomePath: a real directory is held to the same rule: in a holder others can write and that is not sticky it is refused, in a sticky one it is accepted', async (t) => {
    const top = scratch(t);
    const loose = dirAt(top, 'loose', 0o777);
    const home = dirAt(loose, 'home');
    assert.throws(() => walk(t, home), refused(/loose, which holds .*home, can be written by its group or by others and is not sticky/));
    const { host } = await macHost(t, { home });
    assert.equal((await refusal(host)).code, 'invalid_host');
    assert.deepEqual(fs.readdirSync(home), [], 'nothing was created in the home');
    const sticky = dirAt(top, 'sticky', 0o1777);
    const stickyHome = dirAt(sticky, 'home');
    assert.doesNotThrow(() => walk(t, stickyHome));
    const { host: accepted } = await macHost(t, { home: stickyHome });
    assert.equal((await accepted.acquire()).owned, true);
    await accepted.drain();
});

test('checkHomePath: a relative link and `..` are resolved the way the kernel does, after the link: the real directory is what is held to the rule', async (t) => {
    const top = scratch(t);
    const a = dirAt(top, 'a');
    const real = dirAt(top, 'b');
    fs.symlinkSync('../b', path.join(a, 'L'));
    assert.doesNotThrow(() => walk(t, path.join(a, 'L')));
    const { host } = await macHost(t, { home: path.join(a, 'L') });
    assert.equal((await host.acquire()).owned, true);
    assert.equal(bits(path.join(real, 'local-llm')), 0o700);
    await host.drain();
    // A `..` in a link's target is applied to the real directory the link led to: `M` is `N/../h` and `N` is `safe/deep`, so `M` is
    // `top/safe/h`, not `top/h`. Only one of the two `h` is a directory, which tells the two readings apart.
    const safe = dirAt(top, 'safe');
    dirAt(safe, 'deep');
    fs.symlinkSync('safe/deep', path.join(top, 'N'));
    fs.symlinkSync('N/../h', path.join(top, 'M'));
    fs.writeFileSync(path.join(top, 'h'), 'a file, in the lexical reading');
    const kernelReading = dirAt(safe, 'h');
    assert.equal(fs.statSync(path.join(top, 'M')).ino, fs.statSync(kernelReading).ino, 'the kernel reads M as safe/h');
    assert.doesNotThrow(() => walk(t, path.join(top, 'M')));
    fs.rmSync(path.join(top, 'h'));
    fs.rmSync(kernelReading, { recursive: true });
    dirAt(top, 'h');
    fs.writeFileSync(path.join(safe, 'h'), 'a file, in the kernel reading');
    assert.throws(() => walk(t, path.join(top, 'M')), refused(/neither a directory nor a symbolic link/));
    // A home that itself has a `..` is not walked, because the host's paths and the kernel would read it differently.
    assert.throws(() => walk(t, `${top}/N/../h`), refused(/'\.\.' component/));
    // `.` and repeated separators are nothing.
    assert.doesNotThrow(() => walk(t, `${top}//./b/.`));
});

test('checkHomePath: a loop of links is refused after 40 expansions, a dangling link is not a directory, and a missing home is allowed only when it is not made yet', async (t) => {
    const top = scratch(t);
    fs.symlinkSync(path.join(top, 'b'), path.join(top, 'a'));
    fs.symlinkSync(path.join(top, 'a'), path.join(top, 'b'));
    assert.throws(() => walk(t, path.join(top, 'a')), refused(/more than 40 symbolic links/));
    fs.symlinkSync('self', path.join(top, 'self'));
    assert.throws(() => walk(t, path.join(top, 'self', 'x')), refused(/more than 40 symbolic links/));
    // 40 links in a chain are fine, 41 are not.
    const chain = dirAt(top, 'chain');
    const end = dirAt(chain, 'end');
    let previous = end;
    for (let i = 0; i < 40; i += 1) {
        const link = path.join(chain, `l${i}`);
        fs.symlinkSync(previous, link);
        previous = link;
    }
    assert.doesNotThrow(() => walk(t, previous));
    const over = path.join(chain, 'l40');
    fs.symlinkSync(previous, over);
    assert.throws(() => walk(t, over), refused(/more than 40 symbolic links/));
    // Missing parts.
    const missing = path.join(top, 'not', 'yet');
    assert.throws(() => walk(t, missing), refused(/cannot be read \(ENOENT\)/));
    assert.doesNotThrow(() => walk(t, missing, { allowMissing: true }));
    const loose = dirAt(top, 'loose2', 0o777);
    assert.throws(() => walk(t, path.join(loose, 'new', 'home'), { allowMissing: true }), refused(/loose2, which holds .*new, can be written by its group or by others/), 'what is missing would be made in a loose holder');
    const file = path.join(top, 'file');
    fs.writeFileSync(file, 'x');
    assert.throws(() => walk(t, file), refused(/neither a directory nor a symbolic link/));
    assert.throws(() => walk(t, path.join(top, 'dangling-to-nowhere')), refused(/ENOENT/));
    fs.symlinkSync(path.join(top, 'nowhere'), path.join(top, 'dangling'));
    assert.throws(() => walk(t, path.join(top, 'dangling')), refused(/nowhere. cannot be read \(ENOENT\)|nowhere cannot be read \(ENOENT\)/));
    assert.throws(() => walk(t, path.join(top, 'chain'), { uid: null }), refused(/user id is unknown/));
    // A home that is not made yet under a loose holder is refused by the host before anything is made.
    const { host } = await macHost(t, { home: path.join(loose, 'newhome') });
    assert.equal((await refusal(host)).code, 'invalid_host');
    assert.deepEqual(fs.readdirSync(loose), [], 'the home was not made in the loose holder');
});

// A file system in memory: { path: { type: 'dir' | 'link', uid, mode, target } }, with only what the walk reads.
function memFs(entries) {
    const stat = (spec) => ({ uid: spec.uid ?? 0, mode: spec.mode ?? 0o755, isSymbolicLink: () => spec.type === 'link', isDirectory: () => spec.type === 'dir' });
    return {
        lstatSync(file) {
            if (!entries[file]) throw Object.assign(new Error(`ENOENT: ${file}`), { code: 'ENOENT' });
            return stat(entries[file]);
        },
        readlinkSync: (file) => entries[file].target,
    };
}
const dir = (uid = 0, mode = 0o755) => ({ type: 'dir', uid, mode });
const link = (target, uid = 0) => ({ type: 'link', uid, mode: 0o755, target });

test('checkHomePath from the real root `/`, on a fake file system laid out like a Mac: /var and /tmp are root-owned links, and the usual homes pass', () => {
    const me = 501;
    const entries = {
        '/': dir(), '/Users': dir(), '/Users/me': dir(me, 0o750), '/Users/me/pw': dir(me, 0o700),
        '/var': link('private/var'), '/tmp': link('private/tmp'), '/private': dir(), '/private/var': dir(), '/private/var/folders': dir(),
        '/private/var/folders/zz': dir(me, 0o755), '/private/var/folders/zz/T': dir(me, 0o700), '/private/var/folders/zz/T/home': dir(me, 0o700),
        '/private/tmp': dir(0, 0o1777), '/private/tmp/x': dir(me, 0o700),
    };
    const fsApi = memFs(entries);
    const check = (home, extra = {}) => checkHomePath(home, { fsApi, uid: me, ...extra });
    for (const home of ['/Users/me/pw', '/var/folders/zz/T/home', '/tmp/x', '/private/tmp/x']) assert.doesNotThrow(() => check(home), home);
    // The same layout, one rule broken at a time.
    const broken = (change) => memFs({ ...entries, ...change });
    assert.throws(() => check('/Users/me/pw', { fsApi: broken({ '/': dir(0, 0o777) }) }), refused(/\/, which holds \/Users, can be written by its group or by others and is not sticky \(mode 0777\)/));
    assert.doesNotThrow(() => check('/Users/me/pw', { fsApi: broken({ '/': dir(0, 0o1777) }) }));
    assert.throws(() => check('/Users/me/pw', { fsApi: broken({ '/Users': dir(0, 0o775) }) }), refused(/\/Users, which holds \/Users\/me, can be written/));
    assert.throws(() => check('/Users/me/pw', { fsApi: broken({ '/Users/me': dir(me, 0o770) }) }), refused(/\/Users\/me, which holds \/Users\/me\/pw, can be written/));
    assert.throws(() => check('/Users/me/pw', { fsApi: broken({ '/Users': dir(777) }) }), refused(/\/Users, a directory in \/, is owned by user 777/));
    assert.throws(() => check('/Users/me/pw', { fsApi: broken({ '/': dir(777) }) }), refused(/\/, which holds \/Users, is owned by user 777, not by this user/));
    assert.throws(() => check('/tmp/x', { fsApi: broken({ '/private/tmp': dir(0, 0o777) }) }), refused(/\/private\/tmp, which holds \/private\/tmp\/x, can be written/));
    assert.throws(() => check('/tmp/x', { fsApi: broken({ '/tmp': link('private/tmp', 777) }) }), refused(/\/tmp, a symbolic link in \/, is owned by user 777/));
    // A link that points out of a loose directory does not make the loose directory safe.
    assert.throws(() => check('/Users/me/via/pw', { fsApi: memFs({ ...entries, '/Users/me/via': link('/loose/dir'), '/loose': dir(0, 0o777), '/loose/dir': dir(me), '/loose/dir/pw': dir(me) }) }), refused(/\/loose, which holds \/loose\/dir/));
});

test('checkHomePath `root`: the walk starts at the directory it is given and trusts it, a home outside it is refused, and an absolute link target outside it is walked from `/`', () => {
    const me = 501;
    const entries = { '/': dir(), '/a': dir(me, 0o777), '/a/b': dir(me, 0o700), '/a/b/home': dir(me, 0o700), '/a/b/out': link('/elsewhere/real'), '/a/b/in': link('/a/b/home'),
        '/elsewhere': dir(0, 0o777), '/elsewhere/real': dir(me, 0o700) };
    const fsApi = memFs(entries);
    const check = (home, extra = {}) => checkHomePath(home, { fsApi, uid: me, ...extra });
    // The loose /a is trusted as the root; it is a refused holder when the walk starts at `/`.
    assert.doesNotThrow(() => check('/a/b/home', { root: '/a' }));
    assert.throws(() => check('/a/b/home'), refused(/\/a, which holds \/a\/b, can be written/));
    assert.throws(() => check('/a/b/home', { root: '/other' }), refused(/not below \/other/));
    assert.throws(() => check('/another/home', { root: '/a' }), refused(/not below \/a, where the check of its path starts/));
    assert.throws(() => check('/ab/home', { root: '/a' }), refused(/not below/), 'a sibling that shares a prefix is not below');
    for (const root of ['relative', '/a/', '/a/../a', '']) assert.throws(() => check('/a/b/home', { root }), refused(/not an absolute normalized path/), JSON.stringify(root));
    // An absolute link target below the root continues from the root; one outside it is walked from `/`, every ancestor checked.
    assert.doesNotThrow(() => check('/a/b/in', { root: '/a' }));
    assert.throws(() => check('/a/b/out', { root: '/a' }), refused(/\/elsewhere, which holds \/elsewhere\/real, can be written/));
    // The home itself being the root leaves nothing to check below it.
    assert.doesNotThrow(() => check('/a', { root: '/a' }));
});

test('the check runs on macOS only: on Linux a loose home, a home in a loose holder and a linked layout directory are accepted as they always were', async (t) => {
    const home = scratch(t);
    fs.chmodSync(home, 0o777);
    const target = scratch(t);
    fs.mkdirSync(path.join(home, 'local-llm'));
    fs.symlinkSync(target, path.join(home, 'local-llm', 'models'));
    const ownerPort = await freePort();
    const linux = createLocalLlmHost({ config: {}, home, env: {}, platformName: 'linux', arch: 'x64', ownerPort, procDir: path.join(home, 'no-proc') });
    t.after(() => linux.drain().catch(() => {}));
    assert.equal((await linux.acquire()).owned, true);
    // A home that is a link in a holder others can write, and a real home in one, are as accepted on Linux as they were.
    const top = scratch(t);
    const holder = dirAt(top, 'holder', 0o777);
    fs.symlinkSync(dirAt(top, 'real'), path.join(holder, 'store'));
    for (const loose of [path.join(holder, 'store'), dirAt(holder, 'home')]) {
        const other = createLocalLlmHost({ config: {}, home: loose, env: {}, platformName: 'linux', arch: 'x64', ownerPort: await freePort(), procDir: path.join(top, 'no-proc') });
        t.after(() => other.drain().catch(() => {}));
        assert.equal((await other.acquire()).owned, true, loose);
    }
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
