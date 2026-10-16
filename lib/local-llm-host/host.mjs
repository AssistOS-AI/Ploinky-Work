// The local-llm host of one pworker home: what `pworker serve` gives the engine (lib/local-llm) so that it runs as a standalone
// server. It owns the home's layout, the ownership of the home, the recovery of runners an earlier process left behind, and one
// lazily created engine controller. It runs on Linux and on macOS (Apple Silicon: the metal profile; Intel: the cpu profile,
// with no hardware to prove it on yet); on another platform it reports itself unsupported and creates nothing.
//
//   const host = createLocalLlmHost({ config, home, env });      // no file system access, no listener
//
//   host.enabled                     false when config.localLlm.enabled === false
//   host.supported                   { ok, reason }: false on a platform without a host here (the engine is not created)
//   host.layout                      { home, root, models, runners, run, state, logs, engineHome, launchDir, ownerFile, keysFile, socketDir }
//   await host.acquire({ waitMs })   take ownership of the home and reap orphaned runners (idempotent). Resolves
//                                    { owned, port, reaped }; owned false when disabled or unsupported. Rejects with a
//                                    LocalLlmError and remembers it: owner_held (another pworker of this home runs the engine;
//                                    retried for waitMs, for a predecessor that is still draining), owner_port_collision,
//                                    owner_port_taken, invalid_socket_dir, and on macOS invalid_host for a home or engine directory
//                                    that is a link, not the user's own, or writable by others. `host.refusal` holds the remembered error.
//   await host.controller()          the engine controller (createLocalLlmController), created on first use after acquire().
//                                    Throws the remembered refusal, local_llm_disabled, local_llm_unsupported or shutting_down.
//   host.current()                   the controller if it was created, else null (never creates one)
//   await host.drain()               drain the controller if there is one, then release the ownership. Idempotent; later
//                                    controller() calls throw shutting_down.
//   host.describe()                  { enabled, supported, ownership: { state, port, refusal }, socketDir, socketDirProblem, layout,
//                                    runners: [{ id, supported, reason }], seedCatalogFile, runnerLockFile }; no side effects, never throws
//   host.engineEnvironment()         the filtered environment the engine and its runners get: PATH (system directories and, on
//                                    Linux, the directory of nvidia-smi when it is on PATH), HOME, LANG and HF_TOKEN (the process
//                                    environment, then keys/huggingface.env) when there is one; nothing else
//   host.runnerVersion(runnerId)     the version this platform's host lock pins for a runner, or null
//   host.redact(text)                `text` with the Hugging Face token replaced by ***
//   host.secretValues()              [the Hugging Face token] when there is one
//
// How the gate and the routes (slice 2) plug in: the gate calls `await host.controller()` and then `controller.chatTarget()`,
// `controller.run(...)` and `controller.stop()`; the routes call the controller's overview, status, run, stop, cancelDownload,
// deleteWeights, addModel, lookupModel, updateModel, removeModel, installRunner and uninstallRunner. A chat target is
// { runnerId, modelId, baseUrl: 'http://localhost', socketPath, apiKey, model, requestOptions }: send it with
// httpFetch(url, { socketPath }) and the per-start apiKey as the bearer token.
//
// On macOS the host gives the engine its platform (darwinPlatform), the fixed profile (metal on arm64, cpu on x64), the macOS
// readers of darwin.mjs, the `ps` process table of darwinProcs.mjs and the host lock of darwin; it checks the directories of
// the home before anything is created in them (see checkPrivateDirs). Linux is as it was.
//
// Seams for tests (all optional): platformName, arch, procDir, ownerPort, which, killImpl, fsApi, netImpl, log, crashHook,
// reapOptions, uid, procs and darwinExec and execFileImpl (macOS: the process table, the exec of sysctl and vm_stat, and the exec
// of the runner's --list-devices), controllerOptions (passed to createLocalLlmController, so a test can inject snapshot, download, ...).

import { execFile, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createLocalLlmController } from '../local-llm/index.mjs';
import { LocalLlmError } from '../local-llm/errors.mjs';
import { defineMessages } from '../local-llm/messages.mjs';
import { RUNNER_IDS } from '../local-llm/runners/index.mjs';
import { createRunners } from '../local-llm/runners/index.mjs';
import { defineHost, linuxPlatform } from '../local-llm/host.mjs';
import { platformFor } from '../local-llm/darwinPlatform.mjs';
import { parseEnvFile } from '../settings.mjs';
import { createDarwinSnapshotReader, readDarwinMemory, readDarwinPressure } from './darwin.mjs';
import { createDarwinProcs } from './darwinProcs.mjs';
import { createLaunchRecorder } from './launch.mjs';
import { acquireOwnership, ownerPortFor } from './ownership.mjs';
import { reapOrphans } from './reaper.mjs';
import { createRunnerStarter, envCrashHook } from './starter.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const SEED_CATALOG_FILE = path.join(HERE, 'catalog', 'models.json');
const LOCK_ARCH = Object.freeze({ x64: 'amd64', arm64: 'arm64' });
const LOCK_PLATFORM = Object.freeze({ linux: 'linux', darwin: 'darwin' });
// What each Mac architecture runs: the profile the host commits to, and the one runner its lock holds.
const DARWIN_PROFILE = Object.freeze({ arm64: 'metal', x64: 'cpu' });
const DARWIN_RUNNER = Object.freeze({ metal: 'llama.cpp', cpu: 'llama.cpp-cpu' });
/** Where the host's runner locks are, by CPU architecture name of this process (`process.arch`) and platform (Linux unless told). */
export const hostLockFile = (arch, platformName = 'linux') => (LOCK_ARCH[arch] && LOCK_PLATFORM[platformName]
    ? path.join(HERE, 'locks', `runners.lock.host-${LOCK_PLATFORM[platformName]}-${LOCK_ARCH[arch]}.json`) : null);
/** The entry a host lock has for a runner, or null when the lock or the entry is unreadable. */
function hostLockEntry(runnerId, arch, platformName, fsApi) {
    const file = hostLockFile(arch, platformName);
    if (!file) return null;
    try {
        const entry = JSON.parse(fsApi.readFileSync(file, 'utf8'))?.runners?.[runnerId];
        return entry && typeof entry === 'object' ? entry : null;
    } catch {
        return null;
    }
}
/** The version a host lock pins for a runner (`b11295` for llama.cpp), or null when the lock or the entry is unreadable. */
export function hostLockVersion(runnerId, arch = process.arch, { fsApi = fs, platformName = 'linux' } = {}) {
    const version = hostLockEntry(runnerId, arch, platformName, fsApi)?.version;
    return typeof version === 'string' && version ? version : null;
}
const SYSTEM_PATH = Object.freeze(['/usr/local/bin', '/usr/bin', '/bin']);
const OWNER_WAIT_STEP_MS = 250;
export const OWNER_WAIT_MS = 45_000;
const NVIDIA_PROFILES = Object.freeze(['dedicated', 'unified']);
const METAL_PROFILES = Object.freeze(['metal']);
const PROBE_TIMEOUT_MS = 15_000;
const PATH_HINT = 'Use a shorter PWORKER_HOME, or set localLlm.socketDir to a short absolute directory that you own with mode 0700.';

/** The wording of the engine's messages for pworker: its own commands instead of a container's. */
export const PWORKER_MESSAGES = defineMessages({
    notReady: () => 'No local model is ready; run one with `pworker models run <id>`.',
    hfTokenHint: () => 'store a token with `pworker models hf-token` and restart `pworker serve`',
    gpuAbsent: () => 'No NVIDIA GPU is available: nvidia-smi was not found on PATH or did not answer.',
    profileRestartGpuRecovered: () => 'GPU access is available now; restart `pworker serve` to select it. Stopping and starting the model does not change the compute profile.',
    profileRestartChanged: () => 'After changing the GPU, the driver or the machine, restart `pworker serve` to select a new compute profile.',
    restartToUseDevice: (profile) => `pworker started with the ${profile} profile; restart \`pworker serve\` to use the current device or run on the CPU.`,
    restartToRunOnCpu: (profile) => `pworker started with the ${profile} profile; restart \`pworker serve\` to run on the CPU.`,
    restartForMemoryModel: (current, selected) => `The GPU now reports ${current} memory, but pworker started with the ${selected} profile; restart \`pworker serve\` to use it.`,
});

function whichIn(name, pathList, fsApi = fs) {
    for (const dir of String(pathList || '').split(':')) {
        if (!dir || !path.isAbsolute(dir)) continue;
        const file = path.join(dir, name);
        try {
            if (fsApi.statSync(file).isFile()) {
                fsApi.accessSync(file, fs.constants.X_OK);
                return file;
            }
        } catch { /* not here */ }
    }
    return null;
}

/**
 * The directory runners' sockets live in, created 0700 when it does not exist and checked either way: a real directory (not a
 * link) owned by this user with no group or other permission. Anything else is refused with `invalid_socket_dir`.
 */
export function prepareSocketDir(dir, { fsApi = fs, uid = typeof process.getuid === 'function' ? process.getuid() : null } = {}) {
    const refuse = (why) => new LocalLlmError('invalid_socket_dir', `The socket directory ${dir} ${why}.`, { socketDir: dir });
    if (typeof dir !== 'string' || !path.isAbsolute(dir) || path.normalize(dir) !== dir || dir.includes(':') || dir.includes(',')) {
        throw refuse('must be an absolute normalized path without ":" or ","');
    }
    fsApi.mkdirSync(path.dirname(dir), { recursive: true, mode: 0o700 });
    let created = false;
    try {
        fsApi.mkdirSync(dir, { mode: 0o700 });
        created = true;
    } catch (error) {
        if (error?.code !== 'EEXIST') throw error;
    }
    // A directory made now is made 0700 whatever the umask said; one that was there is only checked.
    if (created) fsApi.chmodSync(dir, 0o700);
    const stat = fsApi.lstatSync(dir);
    if (stat.isSymbolicLink()) throw refuse('is a symbolic link');
    if (!stat.isDirectory()) throw refuse('is not a directory');
    if (uid !== null && stat.uid !== uid) throw refuse(`is owned by user ${stat.uid}, not by this user (${uid})`);
    if ((stat.mode & 0o777) !== 0o700) throw refuse(`has mode ${(stat.mode & 0o777).toString(8).padStart(4, '0')}; it must be 0700 (no access for the group or others)`);
    return dir;
}

/**
 * The macOS check of the directories the engine keeps its files in (N1, R-2): `dirs`, in order from the real home down, must each be
 * a real directory (not a link), owned by `uid`, with no group or other write bit. A directory that is not there yet is skipped
 * unless `requireAll`. Anything else is refused with `invalid_host`, naming the directory and the fix. Linux does not run it.
 */
export function checkPrivateDirs(dirs, { fsApi = fs, uid = typeof process.getuid === 'function' ? process.getuid() : null, requireAll = true } = {}) {
    for (const dir of dirs) {
        const refuse = (why, fix) => new LocalLlmError('invalid_host', `The directory ${dir} ${why}; pworker keeps its data in directories that only you can change. ${fix}`, { field: 'layout', directory: dir });
        let stat;
        try {
            stat = fsApi.lstatSync(dir);
        } catch (error) {
            if (error?.code === 'ENOENT' && !requireAll) continue;
            throw refuse(`cannot be read (${error?.code ?? error?.message})`, 'Make it a directory that you own and that nobody else can write.');
        }
        if (stat.isSymbolicLink()) throw refuse('is a symbolic link', 'Replace it with a real directory, or point PWORKER_HOME at the directory that holds the data.');
        if (!stat.isDirectory()) throw refuse('is not a directory', 'Move it away.');
        if (uid === null) throw refuse('cannot be checked: the user id is unknown', 'Run pworker as a regular user.');
        if (stat.uid !== uid) throw refuse(`is owned by user ${stat.uid}, not by this user (${uid})`, 'Use a PWORKER_HOME that you own.');
        if ((stat.mode & 0o022) !== 0) throw refuse(`can be written by its group or by others (mode ${(stat.mode & 0o777).toString(8).padStart(4, '0')})`, `Run \`chmod go-w ${dir}\`.`);
    }
}

export function createLocalLlmHost({
    config = {}, home, env = process.env, platformName = process.platform, arch = process.arch, procDir = '/proc',
    ownerPort = null, which = null, killImpl = null, fsApi = fs, netImpl = undefined, log = () => {}, crashHook = null,
    reapOptions = {}, controllerOptions = {}, homeDir = os.homedir(),
    uid = typeof process.getuid === 'function' ? process.getuid() : null, procs = null, darwinExec = execFileSync, execFileImpl = execFile,
} = {}) {
    if (typeof home !== 'string' || !path.isAbsolute(home)) throw new LocalLlmError('invalid_host', 'The pworker home must be an absolute path.', { field: 'home' });
    const section = config.localLlm && typeof config.localLlm === 'object' ? config.localLlm : {};
    const enabled = section.enabled !== false;
    const darwin = platformName === 'darwin';
    // On a Mac the profile is not decided from a probe: Apple Silicon runs the Metal build, an Intel Mac the CPU build.
    const darwinProfile = darwin ? DARWIN_PROFILE[arch] ?? null : null;
    const supported = platformName === 'linux' || darwinProfile
        ? { ok: true, reason: null }
        : { ok: false, reason: `The local-llm engine is not supported on ${darwin ? `darwin-${arch}` : platformName} in this release.` };
    const engineRoot = path.join(home, 'local-llm');
    const layout = Object.freeze({
        home,
        root: engineRoot,
        models: path.join(engineRoot, 'models'),
        runners: path.join(engineRoot, 'runners'),
        run: path.join(engineRoot, 'run'),
        state: path.join(engineRoot, 'state'),
        logs: path.join(engineRoot, 'logs'),
        engineHome: path.join(engineRoot, 'home'),
        launchDir: path.join(engineRoot, 'state', 'launch'),
        ownerFile: path.join(engineRoot, 'state', 'owner.json'),
        keysFile: path.join(home, 'keys', 'huggingface.env'),
        socketDir: section.socketDir ?? path.join(home, 'r'),
    });
    const findOnPath = which ?? ((name, pathList) => whichIn(name, pathList, fsApi));

    let phase = 'idle';
    let acquiring = null;
    let ownership = null;
    let refusal = null;
    let controller = null;
    let draining = null;
    let closing = false; // set when a drain is asked for: a pending acquire gives up instead of acquiring and reaping after the shutdown request
    let token = null;

    // A refusal is remembered. Only `owner_held` is tried again by the next acquire(): the other holder may have been draining.
    function fail(error) {
        refusal = error;
        phase = 'refused';
        hostLog(`local-llm: ${error.message}`);
        return error;
    }

    // Every line the host writes passes the same filter as an error text: the Hugging Face token never reaches a log sink.
    const hostLog = (line) => { try { log(redactToken(String(line))); } catch { /* a log sink must not break the host */ } };
    const redactToken = (text) => {
        const secret = readToken();
        return secret ? text.split(secret).join('***') : text;
    };

    function readToken() {
        if (token !== null) return token;
        token = (env.HF_TOKEN || parseEnvFile(layout.keysFile).HF_TOKEN || '').trim();
        return token;
    }

    // Everything the engine's runners and downloads get from the environment: nothing else crosses (no CUDA_MPS_*, no keys).
    function engineEnvironment(nvidiaSmi) {
        const pathDirs = [...SYSTEM_PATH];
        if (nvidiaSmi && !pathDirs.includes(path.dirname(nvidiaSmi))) pathDirs.push(path.dirname(nvidiaSmi));
        const filtered = { PATH: pathDirs.join(':'), HOME: env.HOME || homeDir, LANG: env.LANG || 'C.UTF-8' };
        if (readToken()) filtered.HF_TOKEN = readToken();
        return filtered;
    }

    // The runners of v1. On Linux: llama.cpp (the CUDA build, only with an NVIDIA GPU) and its CPU build. On a Mac: llama.cpp (the
    // Metal build) on Apple Silicon, its CPU build on an Intel Mac, because the host lock of each holds that one entry. The rest are
    // listed as unsupported with a reason.
    function hostRunners(engineHost) {
        const all = createRunners({ host: engineHost, messages: PWORKER_MESSAGES });
        const table = {};
        for (const id of RUNNER_IDS) {
            const runner = all[id];
            let reason = null;
            if (darwin) {
                if (id !== DARWIN_RUNNER[darwinProfile]) reason = `${runner.displayName} is not offered by the standalone pworker on macOS in this release.`;
            } else if (id === 'llama.cpp') reason = engineHost.nvidiaSmi ? null : 'The CUDA build of llama.cpp needs an NVIDIA GPU; nvidia-smi was not found on PATH. The CPU build runs models without one.';
            else if (id !== 'llama.cpp-cpu') reason = `${runner.displayName} is not offered by the standalone pworker on Linux in this release.`;
            // The engine's llama.cpp pins the build the agent's image carries; the host installs its own lock, so the runner reports the
            // lock's version (the agent path, which never goes through this table, keeps its own).
            const locked = hostLockVersion(id, arch, { fsApi, platformName });
            const pinned = locked ? { pinnedVersion: locked } : {};
            // The CUDA build is for the NVIDIA profiles alone: on a machine that commits to the cpu profile (nvidia-smi present, no usable
            // device) the engine hides it and refuses Run and Install, and the CPU build serves. The Metal build is for the metal profile alone.
            if (id === 'llama.cpp' && reason === null) table[id] = Object.freeze({ ...runner, ...pinned, profiles: darwin ? METAL_PROFILES : NVIDIA_PROFILES });
            else table[id] = reason === null ? Object.freeze({ ...runner, ...pinned }) : Object.freeze({ ...runner, ...pinned, supported: false, unsupportedReason: reason });
        }
        return Object.freeze(table);
    }

    // The privateRoot of the macOS platform: the engine directory as the file system resolves it (it may not exist yet).
    function realEngineRoot() {
        try { return fsApi.realpathSync(layout.root); } catch { return layout.root; }
    }

    function buildEngineHost({ listen = true } = {}) {
        // There is no nvidia-smi on a Mac.
        const nvidiaSmi = darwin ? null : findOnPath('nvidia-smi', env.PATH);
        return {
            nvidiaSmi,
            host: defineHost({
                dataDir: layout.root,
                runRoot: layout.run,
                sharedModelsRoot: null,
                imageContractFile: null,
                imageRunnerLockFile: hostLockFile(arch, platformName),
                agentRunnerLockFile: () => null,
                seedCatalogFile: SEED_CATALOG_FILE,
                nvidiaSmi,
                nvidiaLibDir: null,
                executables: {},
                vulkan: null,
                lmStudioSdkDir: null,
                uv: null,
                python: null,
                toolPath: SYSTEM_PATH.join(':'),
                gpuAccess: null,
                platform: darwin ? platformFor('darwin', { privateRoot: realEngineRoot(), uid, exec: darwinExec }) : linuxPlatform({ procDir }),
                listen: listen ? { socketDir: layout.socketDir, pathHint: PATH_HINT } : null,
            }),
        };
    }

    const layoutDirs = [layout.root, layout.models, layout.runners, layout.run, layout.state, layout.logs, layout.engineHome, layout.launchDir];

    // macOS (N1, R-2): the real home and each directory of the layout below it, as the file system resolves them. The home is
    // resolved first, so a `/var` that is a link to `/private/var` is the file system's own and not a refusal.
    function checkLayoutDirs({ requireAll }) {
        const realHome = fsApi.realpathSync(home);
        const real = [realHome, ...layoutDirs.map((dir) => path.join(realHome, path.relative(home, dir)))];
        checkPrivateDirs(real, { fsApi, uid, requireAll });
    }

    function prepareLayout() {
        // Before anything is created below the home: a home or an engine directory that is loose is refused with nothing added to it.
        if (darwin) checkLayoutDirs({ requireAll: false });
        for (const dir of layoutDirs) {
            fsApi.mkdirSync(dir, { recursive: true, mode: 0o700 });
        }
        // With every directory there, each is checked: a link to somewhere else, another owner, or a mode others can write.
        if (darwin) checkLayoutDirs({ requireAll: true });
        prepareSocketDir(layout.socketDir, { fsApi });
    }

    async function acquireOnce({ waitMs }) {
        if (!enabled) return { owned: false, port: null, reaped: null, reason: 'disabled' };
        if (!supported.ok) return { owned: false, port: null, reaped: null, reason: supported.reason };
        try {
            fsApi.mkdirSync(home, { recursive: true, mode: 0o700 });
            prepareLayout();
        } catch (error) {
            throw fail(error instanceof LocalLlmError ? error : new LocalLlmError('invalid_host', `The local-llm directories under ${home} could not be prepared: ${error?.code ?? error?.message}`, { field: 'layout' }));
        }
        const realHome = fsApi.realpathSync(home);
        const deadline = Date.now() + waitMs;
        const shuttingDown = () => new LocalLlmError('shutting_down', 'The local-llm host is shutting down.');
        for (;;) {
            if (closing) throw shuttingDown();
            try {
                ownership = await acquireOwnership({
                    realHome, port: ownerPort ?? ownerPortFor(realHome), ownerFile: layout.ownerFile, netImpl, fsApi, log: hostLog, platformName,
                });
                break;
            } catch (error) {
                // A predecessor that is still draining holds the port for a while; any other refusal is final.
                if (error?.code === 'owner_held' && Date.now() < deadline) {
                    await new Promise((resolve) => setTimeout(resolve, OWNER_WAIT_STEP_MS));
                    continue;
                }
                throw fail(error);
            }
        }
        if (closing) {
            // Asked to shut down while the bind was in flight: hold nothing and reap nothing.
            const held = ownership;
            ownership = null;
            await held.release();
            throw shuttingDown();
        }
        phase = 'owned';
        // Only the owner reaps: a process that did not get the port never reads a launch record to signal anything.
        let reaped;
        try {
            reaped = await reapOrphans({
                launchDir: layout.launchDir, socketDir: layout.socketDir, procDir, platformName, fsApi, log: hostLog, ...(procs ? { procs } : {}),
                ...(killImpl ? { killImpl } : {}), ...reapOptions,
            });
        } catch (error) {
            // The ownership is held and stays held: a failed reap must not leave a listener behind that refuses its own process.
            hostLog(`local-llm: orphaned runners could not be looked for (${error?.code ?? error?.message}); the launch records are left as they are`);
            reaped = { reaped: [], skipped: [{ reason: `the launch records could not be read (${error?.code ?? error?.message})` }], failed: [] };
        }
        return { owned: true, port: ownership.port, reaped };
    }

    function acquire({ waitMs = 0 } = {}) {
        if (draining) return Promise.reject(new LocalLlmError('shutting_down', 'The local-llm host is shutting down.'));
        if (refusal && refusal.code !== 'owner_held') return Promise.reject(refusal);
        if (!acquiring) {
            refusal = null;
            acquiring = acquireOnce({ waitMs }).then(
                (value) => value,
                (error) => { acquiring = null; throw error; },
            );
        }
        return acquiring;
    }

    async function getController() {
        if (draining) throw new LocalLlmError('shutting_down', 'The local-llm host is shutting down.');
        if (!enabled) throw new LocalLlmError('local_llm_disabled', 'The local-llm engine is disabled in the configuration (localLlm.enabled).');
        if (!supported.ok) throw new LocalLlmError('local_llm_unsupported', supported.reason);
        await acquire();
        if (draining) throw new LocalLlmError('shutting_down', 'The local-llm host is shutting down.');
        if (controller) return controller;
        const { host, nvidiaSmi } = buildEngineHost();
        const recorder = createLaunchRecorder({ dir: layout.launchDir, procDir, fsApi, platformName, ...(procs ? { procs } : {}) });
        const startRunner = createRunnerStarter({
            recorder, socketDir: layout.socketDir, platform: host.platform, killImpl, log: hostLog,
            crashHook: crashHook ?? envCrashHook(env),
        });
        controller = createLocalLlmController({
            host, messages: PWORKER_MESSAGES, env: engineEnvironment(nvidiaSmi), runners: hostRunners(host), startRunner,
            ...(darwin ? darwinSeams() : {}), ...controllerOptions,
        });
        return controller;
    }

    // What a Mac gives the engine in place of what it reads from /proc and nvidia-smi: the fixed profile, the hardware snapshot (the
    // Metal device from the runner's own --list-devices), and the memory and pressure the guard samples (the kernel's pressure level).
    function darwinSeams() {
        return {
            profile: darwinProfile,
            readHardwareSnapshot: createDarwinSnapshotReader({ exec: darwinExec, dataDir: layout.root, probeDevice: probeMetalDevices, metal: darwinProfile === 'metal' }),
            readMemory: () => readDarwinMemory({ exec: darwinExec }),
            readPressure: () => readDarwinPressure({ exec: darwinExec }),
        };
    }

    // The Metal device of the installed llama.cpp, as `llama-server --list-devices` prints it, or null while it is not installed (the
    // snapshot then has no device limit and admission refuses). The program is the one the installer made ready under the run root,
    // run with the engine's filtered environment less the token; a failure rejects, and the snapshot treats it as no answer.
    function probeMetalDevices({ signal } = {}) {
        const entry = hostLockEntry('llama.cpp', arch, platformName, fsApi);
        const relative = entry?.check?.executable;
        if (typeof entry?.version !== 'string' || typeof relative !== 'string' || !relative || path.isAbsolute(relative)
            || path.normalize(relative) !== relative || relative.startsWith('..')) return Promise.resolve(null);
        const dir = path.join(layout.run, 'llama.cpp', entry.version);
        const program = path.join(dir, relative);
        try {
            // The installer writes the marker last, so a copy without it is not installed (or not whole).
            if (!fsApi.statSync(path.join(dir, '.ready.json')).isFile() || !fsApi.statSync(program).isFile()) return Promise.resolve(null);
        } catch {
            return Promise.resolve(null);
        }
        // The device list needs no token: the probe gets the path, home and locale of the engine's environment and nothing else.
        const { PATH, HOME, LANG } = engineEnvironment(null);
        return new Promise((resolve, reject) => {
            execFileImpl(program, ['--list-devices'], {
                cwd: layout.engineHome, env: { PATH, HOME, LANG }, timeout: PROBE_TIMEOUT_MS, signal, maxBuffer: 1024 * 1024, encoding: 'utf8',
            }, (error, stdout, stderr) => (error ? reject(error) : resolve(`${stdout}\n${stderr}`)));
        });
    }

    async function drain() {
        closing = true;
        draining ??= (async () => {
            // An acquisition in flight finishes first, so its listener is released too.
            try { await acquiring; } catch { /* a refusal holds nothing */ }
            try {
                if (controller) await controller.drain();
            } finally {
                if (ownership) await ownership.release();
                phase = 'closed';
            }
        })();
        return draining;
    }

    // Never throws: an unusable socket directory is reported as `socketDirProblem`, and the runners are listed from a host without a socket.
    function describe() {
        let socketDirProblem = null;
        let built;
        try {
            built = buildEngineHost();
        } catch (error) {
            socketDirProblem = error?.message ?? String(error);
            built = buildEngineHost({ listen: false });
        }
        const runners = hostRunners(built.host);
        return {
            enabled,
            supported,
            ownership: { state: phase, port: ownership?.port ?? null, refusal: refusal ? { code: refusal.code, message: refusal.message } : null },
            socketDir: layout.socketDir,
            socketDirProblem,
            layout,
            runners: RUNNER_IDS.map((id) => ({ id, supported: runners[id].supported, reason: runners[id].supported ? null : runners[id].unsupportedReason })),
            seedCatalogFile: SEED_CATALOG_FILE,
            runnerLockFile: hostLockFile(arch, platformName),
        };
    }

    return Object.freeze({
        layout,
        enabled,
        supported,
        get refusal() { return refusal; },
        acquire,
        controller: getController,
        current: () => controller,
        drain,
        describe,
        engineEnvironment: () => engineEnvironment(darwin ? null : findOnPath('nvidia-smi', env.PATH)),
        runnerVersion: (runnerId) => hostLockVersion(runnerId, arch, { fsApi, platformName }),
        /** The secret values the host knows (the Hugging Face token), for the proxy to keep out of its logs. */
        secretValues: () => [readToken()].filter(Boolean),
        redact(text) {
            const secret = readToken();
            return secret ? String(text).split(secret).join('***') : String(text);
        },
    });
}
