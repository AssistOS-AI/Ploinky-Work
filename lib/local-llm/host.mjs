// What the engine needs to know about the machine it runs on, as one frozen
// value (`defineHost`), and the operating-system strategies the engine calls
// (`linuxPlatform`). Nothing here reads the file system or the environment:
// every path is data the caller chose, and a path that is not given stays
// null, which turns the feature that needs it off instead of guessing.

import path from 'node:path';

import { LocalLlmError } from './errors.mjs';

/** The PATH of the processes the engine starts when its own environment names none. */
export const DEFAULT_TOOL_PATH = '/usr/local/bin:/usr/bin:/bin';

const HOST_KEYS = Object.freeze([
    'dataDir', 'runRoot', 'sharedModelsRoot', 'imageContractFile', 'imageRunnerLockFile', 'agentRunnerLockFile',
    'seedCatalogFile', 'hfBaseUrl', 'nvidiaSmi', 'nvidiaLibDir', 'executables', 'vulkan', 'lmStudioSdkDir',
    'uv', 'python', 'shmDir', 'toolPath', 'gpuAccess', 'platform', 'listen',
]);
const REQUIRED_PATHS = Object.freeze(['dataDir', 'runRoot']);
const OPTIONAL_PATHS = Object.freeze([
    'sharedModelsRoot', 'imageContractFile', 'imageRunnerLockFile', 'seedCatalogFile', 'nvidiaSmi', 'nvidiaLibDir',
    'lmStudioSdkDir', 'uv', 'python',
]);
const VULKAN_KEYS = Object.freeze(['serverPath', 'vulkaninfo', 'icdDir']);
const LISTEN_KEYS = Object.freeze(['socketDir', 'pathHint']);
// sizeof(sockaddr_un.sun_path) - 1 on Linux; a platform with a shorter path says so (`socketPathMax`).
export const DEFAULT_SOCKET_PATH_MAX = 107;
// The runners that launch an executable the host itself holds; the others run from a lock's runnable copy.
const EXECUTABLE_IDS = Object.freeze(['llama.cpp', 'ik_llama.cpp', 'llama.cpp-vulkan', 'ollama']);

function invalid(field, message) {
    return new LocalLlmError('invalid_host', `Invalid host ${field}: ${message}`, { field });
}

// An absolute, normalized path with no ':' (the engine joins paths into lists with it).
function checkedPath(field, value) {
    if (typeof value !== 'string' || !value) throw invalid(field, 'it must be a non-empty string');
    if (!path.posix.isAbsolute(value)) throw invalid(field, `${value} is not an absolute path`);
    if (path.posix.normalize(value) !== value) throw invalid(field, `${value} is not a normalized path`);
    if (value.includes(':')) throw invalid(field, `${value} contains ':'`);
    return value;
}

function plain(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value) && Object.getPrototypeOf(value) === Object.prototype;
}

function onlyKeys(field, value, allowed) {
    for (const key of Object.keys(value)) {
        if (!allowed.includes(key)) throw invalid(field ? `${field}.${key}` : key, 'unknown key');
    }
}

function checkedExecutables(value) {
    if (!plain(value)) throw invalid('executables', 'it must be an object of runner ids to absolute paths or null');
    onlyKeys('executables', value, EXECUTABLE_IDS);
    const executables = {};
    for (const [id, file] of Object.entries(value)) {
        executables[id] = file === null ? null : checkedPath(`executables.${id}`, file);
    }
    return Object.freeze(executables);
}

function checkedVulkan(value) {
    if (value === null) return null;
    if (!plain(value)) throw invalid('vulkan', 'it must be null or an object');
    onlyKeys('vulkan', value, VULKAN_KEYS);
    for (const key of VULKAN_KEYS) checkedPath(`vulkan.${key}`, value[key]);
    return Object.freeze({ serverPath: value.serverPath, vulkaninfo: value.vulkaninfo, icdDir: value.icdDir });
}

function checkedGpuAccess(value) {
    if (value === null) return null;
    if (!plain(value) || typeof value.report !== 'function' || typeof value.source !== 'string' || !value.source) {
        throw invalid('gpuAccess', 'it must be null or { report(env), source }');
    }
    onlyKeys('gpuAccess', value, ['report', 'source']);
    return Object.freeze({ report: value.report, source: value.source });
}

function checkedListen(value) {
    if (value === null) return null;
    if (!plain(value)) throw invalid('listen', 'it must be null or { socketDir, pathHint? }');
    onlyKeys('listen', value, LISTEN_KEYS);
    const socketDir = checkedPath('listen.socketDir', value.socketDir);
    // llama-server reads `--host` as a comma-separated list.
    if (socketDir.includes(',')) throw invalid('listen.socketDir', `${socketDir} contains ','`);
    if (value.pathHint !== undefined && (typeof value.pathHint !== 'string' || !value.pathHint)) {
        throw invalid('listen.pathHint', 'it must be a non-empty string');
    }
    return Object.freeze({ socketDir, pathHint: value.pathHint ?? null });
}

function checkedBaseUrl(value) {
    let url;
    try {
        url = new URL(value);
    } catch {
        throw invalid('hfBaseUrl', `${String(value)} is not a URL`);
    }
    if (!['https:', 'http:'].includes(url.protocol)) throw invalid('hfBaseUrl', `${String(value)} is not an http(s) URL`);
    return value;
}

/**
 * The strategies that differ between operating systems. The Linux ones are the
 * behavior the engine always had; their argv and paths are byte-identical.
 *
 *   openedPathOf(fsApi, fd)       where the kernel says an open descriptor points
 *   openatPath(fd, name)          `name` inside an open directory, the way openat(2) names it
 *   unreadableReason              why the descriptor's path is unknown, when openedPathOf rejects
 *   setOomScoreAdj(pid, v, opts)  make a process the OOM killer's first choice
 *   tarFlags(into, strip)         the extraction flags of the runner installer
 *   socketPathMax                 the longest Unix socket path in bytes (sun_path less its terminator)
 */
export function linuxPlatform({ procDir = '/proc', devDriDir = '/dev/dri' } = {}) {
    checkedPath('platform.procDir', procDir);
    checkedPath('platform.devDriDir', devDriDir);
    return Object.freeze({
        name: 'linux',
        procDir,
        devDriDir,
        socketPathMax: DEFAULT_SOCKET_PATH_MAX,
        openedPathOf: (fsApi, fd) => fsApi.promises.readlink(`${procDir}/self/fd/${fd}`),
        openatPath: (fd, name) => `${procDir}/self/fd/${fd}/${name}`,
        unreadableReason: `${procDir}/self/fd is not readable`,
        setOomScoreAdj: (pid, value, { fsApi }) => { fsApi.writeFileSync(`${procDir}/${pid}/oom_score_adj`, String(value)); },
        // File times are not restored (--touch): on a fuse-overlayfs, setting a directory's time fails with EPERM.
        tarFlags: (into, strip) => ['-C', into, `--strip-components=${strip}`, '--no-same-owner', '--touch'],
    });
}

function checkedPlatform(value) {
    const required = ['openedPathOf', 'openatPath', 'setOomScoreAdj', 'tarFlags'];
    if (!plain(value) || required.some((key) => typeof value[key] !== 'function')
        || typeof value.procDir !== 'string' || typeof value.devDriDir !== 'string' || typeof value.unreadableReason !== 'string') {
        throw invalid('platform', 'it must come from linuxPlatform() or provide the same strategies');
    }
    if (value.socketPathMax !== undefined && !(Number.isInteger(value.socketPathMax) && value.socketPathMax > 0)) {
        throw invalid('platform.socketPathMax', 'it must be a positive byte count');
    }
    return value;
}

/**
 * The frozen host of one engine instance. `dataDir` and `runRoot` are required;
 * every other path is absolute or null, and null switches its feature off:
 *
 *   sharedModelsRoot      null: no workspace-shared weights
 *   imageContractFile     null: no image contract, every runner is available
 *   imageRunnerLockFile   null: an empty image lock
 *   agentRunnerLockFile   (arch) => path or null; the default offers none
 *   seedCatalogFile       null: the caller must pass the seed catalog itself
 *   nvidiaSmi             null: the NVIDIA GPU is absent
 *   nvidiaLibDir          null: no driver library path and no libcuda link
 *   executables           { runnerId: path or null } for llama.cpp, ik_llama.cpp, llama.cpp-vulkan and ollama;
 *                         null or absent: the runner is offered only through a lock
 *   vulkan                null, or { serverPath, vulkaninfo, icdDir }
 *   lmStudioSdkDir        null: LM Studio is unsupported
 *   uv, python            null: Python runners cannot be installed
 *   gpuAccess             null, or { report(env) -> { unavailable, reason }, source }
 *   listen                null: a runner listens on loopback TCP, on the port its adapter names (the default);
 *                         { socketDir, pathHint? }: a runner listens on a Unix socket `<socketDir>/<8 hex>.sock`
 *                         and has no TCP port. `pathHint` is appended to the error of a path that is too long.
 *
 * Unknown keys and invalid values throw `invalid_host`. There is no file system access.
 */
export function defineHost(spec) {
    if (!plain(spec)) throw invalid('(host)', 'it must be an object');
    onlyKeys('', spec, HOST_KEYS);
    const host = {};
    for (const field of REQUIRED_PATHS) host[field] = checkedPath(field, spec[field]);
    for (const field of OPTIONAL_PATHS) host[field] = spec[field] === undefined || spec[field] === null ? null : checkedPath(field, spec[field]);
    if (spec.agentRunnerLockFile !== undefined && typeof spec.agentRunnerLockFile !== 'function') {
        throw invalid('agentRunnerLockFile', 'it must be a function of the CPU architecture');
    }
    host.agentRunnerLockFile = spec.agentRunnerLockFile ?? (() => null);
    host.hfBaseUrl = checkedBaseUrl(spec.hfBaseUrl ?? 'https://huggingface.co');
    host.executables = checkedExecutables(spec.executables ?? {});
    host.vulkan = checkedVulkan(spec.vulkan ?? null);
    // The Vulkan build is launched from one path: the executable and the Vulkan layout may name it twice, but must agree.
    const vulkanExecutable = host.executables['llama.cpp-vulkan'] ?? null;
    if (vulkanExecutable !== null && (host.vulkan === null || host.vulkan.serverPath !== vulkanExecutable)) {
        throw invalid('executables.llama.cpp-vulkan', 'it needs a vulkan layout whose serverPath is the same file');
    }
    host.shmDir = checkedPath('shmDir', spec.shmDir ?? '/dev/shm');
    // A PATH list: its ':' separators are the point.
    host.toolPath = spec.toolPath ?? DEFAULT_TOOL_PATH;
    if (typeof host.toolPath !== 'string' || !host.toolPath) throw invalid('toolPath', 'it must be a non-empty PATH string');
    host.gpuAccess = checkedGpuAccess(spec.gpuAccess ?? null);
    host.platform = checkedPlatform(spec.platform ?? linuxPlatform());
    host.listen = checkedListen(spec.listen ?? null);
    return Object.freeze(host);
}
