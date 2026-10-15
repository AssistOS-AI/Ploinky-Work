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
    'uv', 'python', 'shmDir', 'toolPath', 'gpuAccess', 'platform',
]);
const REQUIRED_PATHS = Object.freeze(['dataDir', 'runRoot']);
const OPTIONAL_PATHS = Object.freeze([
    'sharedModelsRoot', 'imageContractFile', 'imageRunnerLockFile', 'seedCatalogFile', 'nvidiaSmi', 'nvidiaLibDir',
    'lmStudioSdkDir', 'uv', 'python',
]);
const VULKAN_KEYS = Object.freeze(['serverPath', 'vulkaninfo', 'icdDir']);
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
 */
export function linuxPlatform({ procDir = '/proc', devDriDir = '/dev/dri' } = {}) {
    checkedPath('platform.procDir', procDir);
    checkedPath('platform.devDriDir', devDriDir);
    return Object.freeze({
        name: 'linux',
        procDir,
        devDriDir,
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
    return Object.freeze(host);
}
