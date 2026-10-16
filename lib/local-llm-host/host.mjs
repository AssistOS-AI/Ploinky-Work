// The local-llm host of one pworker home: what `pworker serve` gives the engine (lib/local-llm) so that it runs as a standalone
// server. It owns the home's layout, the ownership of the home, the recovery of runners an earlier process left behind, and one
// lazily created engine controller. It is Linux-only in this release: on another platform it reports itself unsupported and
// creates nothing.
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
//                                    owner_port_taken, invalid_socket_dir. `host.refusal` holds the remembered error.
//   await host.controller()          the engine controller (createLocalLlmController), created on first use after acquire().
//                                    Throws the remembered refusal, local_llm_disabled, local_llm_unsupported or shutting_down.
//   host.current()                   the controller if it was created, else null (never creates one)
//   await host.drain()               drain the controller if there is one, then release the ownership. Idempotent; later
//                                    controller() calls throw shutting_down.
//   host.describe()                  { enabled, supported, ownership: { state, port, refusal }, socketDir, socketDirProblem, layout,
//                                    runners: [{ id, supported, reason }], seedCatalogFile, runnerLockFile }; no side effects, never throws
//   host.engineEnvironment()         the filtered environment the engine and its runners get: PATH (system directories and the
//                                    directory of nvidia-smi when it is on PATH), HOME, LANG and HF_TOKEN (the process environment,
//                                    then keys/huggingface.env) when there is one; nothing else
//   host.redact(text)                `text` with the Hugging Face token replaced by ***
//
// How the gate and the routes (slice 2) plug in: the gate calls `await host.controller()` and then `controller.chatTarget()`,
// `controller.run(...)` and `controller.stop()`; the routes call the controller's overview, status, run, stop, cancelDownload,
// deleteWeights, addModel, lookupModel, updateModel, removeModel, installRunner and uninstallRunner. A chat target is
// { runnerId, modelId, baseUrl: 'http://localhost', socketPath, apiKey, model, requestOptions }: send it with
// httpFetch(url, { socketPath }) and the per-start apiKey as the bearer token.
//
// Seams for tests (all optional): platformName, arch, procDir, ownerPort, which, killImpl, fsApi, netImpl, log, crashHook,
// reapOptions, controllerOptions (passed to createLocalLlmController, so a test can inject snapshot, download, ...).

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
import { parseEnvFile } from '../settings.mjs';
import { createLaunchRecorder } from './launch.mjs';
import { acquireOwnership, ownerPortFor } from './ownership.mjs';
import { reapOrphans } from './reaper.mjs';
import { createRunnerStarter, envCrashHook } from './starter.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const SEED_CATALOG_FILE = path.join(HERE, 'catalog', 'models.json');
const LOCK_ARCH = Object.freeze({ x64: 'amd64', arm64: 'arm64' });
/** Where the host's runner locks are, by CPU architecture name of this process (`process.arch`). */
export const hostLockFile = (arch) => (LOCK_ARCH[arch] ? path.join(HERE, 'locks', `runners.lock.host-linux-${LOCK_ARCH[arch]}.json`) : null);
const SYSTEM_PATH = Object.freeze(['/usr/local/bin', '/usr/bin', '/bin']);
const OWNER_WAIT_STEP_MS = 250;
export const OWNER_WAIT_MS = 45_000;
const NVIDIA_PROFILES = Object.freeze(['dedicated', 'unified']);
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

export function createLocalLlmHost({
    config = {}, home, env = process.env, platformName = process.platform, arch = process.arch, procDir = '/proc',
    ownerPort = null, which = null, killImpl = null, fsApi = fs, netImpl = undefined, log = () => {}, crashHook = null,
    reapOptions = {}, controllerOptions = {}, homeDir = os.homedir(),
} = {}) {
    if (typeof home !== 'string' || !path.isAbsolute(home)) throw new LocalLlmError('invalid_host', 'The pworker home must be an absolute path.', { field: 'home' });
    const section = config.localLlm && typeof config.localLlm === 'object' ? config.localLlm : {};
    const enabled = section.enabled !== false;
    const supported = platformName === 'linux'
        ? { ok: true, reason: null }
        : { ok: false, reason: `The local-llm engine is not supported on ${platformName} in this release.` };
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
    const hostLog = (line) => { try { log(line); } catch { /* a log sink must not break the host */ } };
    const findOnPath = which ?? ((name, pathList) => whichIn(name, pathList, fsApi));

    let phase = 'idle';
    let acquiring = null;
    let ownership = null;
    let refusal = null;
    let controller = null;
    let draining = null;
    let token = null;

    // A refusal is remembered. Only `owner_held` is tried again by the next acquire(): the other holder may have been draining.
    function fail(error) {
        refusal = error;
        phase = 'refused';
        hostLog(`local-llm: ${error.message}`);
        return error;
    }

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

    // The runners of v1 on Linux: llama.cpp (the CUDA build, only with an NVIDIA GPU) and its CPU build. The rest are listed as unsupported with a reason.
    function hostRunners(engineHost) {
        const all = createRunners({ host: engineHost, messages: PWORKER_MESSAGES });
        const table = {};
        for (const id of RUNNER_IDS) {
            const runner = all[id];
            let reason = null;
            if (id === 'llama.cpp') reason = engineHost.nvidiaSmi ? null : 'The CUDA build of llama.cpp needs an NVIDIA GPU; nvidia-smi was not found on PATH. The CPU build runs models without one.';
            else if (id !== 'llama.cpp-cpu') reason = `${runner.displayName} is not offered by the standalone pworker on Linux in this release.`;
            // The CUDA build is for the NVIDIA profiles alone: on a machine that commits to the cpu profile (nvidia-smi present, no usable
            // device) the engine hides it and refuses Run and Install, and the CPU build serves.
            if (id === 'llama.cpp' && reason === null) table[id] = Object.freeze({ ...runner, profiles: NVIDIA_PROFILES });
            else table[id] = reason === null ? runner : Object.freeze({ ...runner, supported: false, unsupportedReason: reason });
        }
        return Object.freeze(table);
    }

    function buildEngineHost({ listen = true } = {}) {
        const nvidiaSmi = findOnPath('nvidia-smi', env.PATH);
        return {
            nvidiaSmi,
            host: defineHost({
                dataDir: layout.root,
                runRoot: layout.run,
                sharedModelsRoot: null,
                imageContractFile: null,
                imageRunnerLockFile: hostLockFile(arch),
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
                platform: linuxPlatform({ procDir }),
                listen: listen ? { socketDir: layout.socketDir, pathHint: PATH_HINT } : null,
            }),
        };
    }

    function prepareLayout() {
        for (const dir of [layout.root, layout.models, layout.runners, layout.run, layout.state, layout.logs, layout.engineHome, layout.launchDir]) {
            fsApi.mkdirSync(dir, { recursive: true, mode: 0o700 });
        }
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
        for (;;) {
            try {
                ownership = await acquireOwnership({
                    realHome, port: ownerPort ?? ownerPortFor(realHome), ownerFile: layout.ownerFile, netImpl, fsApi, log: hostLog,
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
        phase = 'owned';
        // Only the owner reaps: a process that did not get the port never reads a launch record to signal anything.
        let reaped;
        try {
            reaped = await reapOrphans({
                launchDir: layout.launchDir, socketDir: layout.socketDir, procDir, platformName, fsApi, log: hostLog,
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
        const recorder = createLaunchRecorder({ dir: layout.launchDir, procDir, fsApi, platformName });
        const startRunner = createRunnerStarter({
            recorder, socketDir: layout.socketDir, platform: host.platform, killImpl, log: hostLog,
            crashHook: crashHook ?? envCrashHook(env),
        });
        controller = createLocalLlmController({
            host, messages: PWORKER_MESSAGES, env: engineEnvironment(nvidiaSmi), runners: hostRunners(host), startRunner, ...controllerOptions,
        });
        return controller;
    }

    async function drain() {
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
            runnerLockFile: hostLockFile(arch),
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
        engineEnvironment: () => engineEnvironment(findOnPath('nvidia-smi', env.PATH)),
        redact(text) {
            const secret = readToken();
            return secret ? String(text).split(secret).join('***') : String(text);
        },
    });
}
