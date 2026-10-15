// The container host, wording and runner table that the moved engine tests ran against before the engine took them as input:
// the paths of the image the engine was written for, and the Ploinky wording of its messages. The values are the ones the
// agent passes, so a test that asserts one of them keeps its assertion. A test that does not care about them passes its own.
import { createController as createEngineController } from '../../../lib/local-llm/controller/deployments.mjs';
import { defineHost } from '../../../lib/local-llm/host.mjs';
import { defineMessages } from '../../../lib/local-llm/messages.mjs';
import { createRunners } from '../../../lib/local-llm/runners/index.mjs';
import { SEED_CATALOG_FILE } from './seedCatalog.mjs';

export const GATEWAY_MODEL_NAME = 'soul_gateway/local-llms/local-llm/default';

/** The Ploinky wording of every message key, byte for byte as the engine had it before the keys existed. */
export const PLOINKY_MESSAGES = defineMessages({
    notReady: () => 'No local model is ready. Run one from Settings > Agents > Local LLMs.',
    gpuAccessReported: (reason) => `Ploinky reports: ${reason || 'GPU access is unavailable; the cause is unknown. Check host `ploinky gpu status`.'}`,
    gpuAbsent: () => 'No GPU is available to this agent. On the host, `ploinky gpu status` shows why: '
        + 'revoked: run `ploinky gpu grant --agent local-llms/local-llm`; '
        + 'not applied yet: run `ploinky start`; or the host has no usable GPU.',
    hfTokenHint: () => 'set a token with `ploinky var HF_TOKEN <token>` and restart local-llm',
    vulkanRuntimeMissing: () => 'A GPU render node is attached, but this image lacks a supported declared Vulkan runtime; update the local-llm image.',
    apuSwitchOff: (deviceName, switchName, switchValue) => `An AMD integrated GPU (${deviceName}) is attached. `
        + `APU use is off: the operator enables it with ploinky var ${switchName} ${switchValue} and a restart of local-llm.`,
    vllmUnifiedSwitchHint: (switchName, switchValue) => `The operator can turn it on with ploinky var ${switchName} ${switchValue}; `
        + 'it has not been measured on this hardware.',
    lmStudioEnableHint: (switchName, switchValue) => `This deployment's operator enables it with "ploinky var ${switchName} ${switchValue}" and a restart of local-llm.`,
    lmStudioStagedUpdate: () => 'LM Studio has staged an update in its home; restart local-llm to rebuild LM Studio from its verified download.',
    profileRestartGpuRecovered: () => 'GPU access is available now; restart local-llm to select it. Stopping and starting the model does not change the compute profile.',
    profileRestartChanged: () => 'After changing GPU access, machine or provider, restart the agent or Box through Ploinky to select a new compute profile.',
    restartToUseDevice: (profile) => `This agent started with the ${profile} profile; restart local-llm to use the current device or run on the CPU.`,
    restartToRunOnCpu: (profile) => `This agent started with the ${profile} profile; restart local-llm to run on the CPU.`,
    restartForMemoryModel: (current, selected) => `The GPU now reports ${current} memory, but this agent started with the ${selected} profile; restart local-llm to use it.`,
});

/** The environment reader the container's GPU grant used: the engine was told why the device is missing. */
export const PLOINKY_GPU_ACCESS = Object.freeze({
    report: (env) => ({ unavailable: env.PLOINKY_GPU_STATUS === 'unavailable', reason: env.PLOINKY_GPU_REASON }),
    source: 'ploinky-environment',
});

export const HOST_VULKAN = Object.freeze({
    serverPath: '/opt/llama.cpp-vulkan/llama-server',
    vulkaninfo: '/usr/bin/vulkaninfo',
    icdDir: '/opt/local-llm/vulkan/icd.d',
});

export const AGENT_HOST_SPEC = Object.freeze({
    dataDir: '/data',
    runRoot: '/opt/runners',
    sharedModelsRoot: '/shared/models',
    imageContractFile: '/opt/local-llm/source.contract',
    imageRunnerLockFile: '/opt/local-llm/runners.lock.json',
    agentRunnerLockFile: () => null,
    seedCatalogFile: SEED_CATALOG_FILE,
    hfBaseUrl: 'https://huggingface.co',
    nvidiaSmi: '/usr/local/nvidia/bin/nvidia-smi',
    nvidiaLibDir: '/usr/local/nvidia/lib64',
    executables: Object.freeze({
        'llama.cpp': '/opt/llama.cpp/llama-server',
        'ik_llama.cpp': '/opt/ik_llama.cpp/llama-server',
        'llama.cpp-vulkan': '/opt/llama.cpp-vulkan/llama-server',
        ollama: '/opt/ollama/bin/ollama',
    }),
    vulkan: HOST_VULKAN,
    lmStudioSdkDir: '/opt/local-llm/lmstudio-sdk',
    uv: '/usr/local/bin/uv',
    python: '/usr/bin/python3',
    shmDir: '/dev/shm',
    toolPath: '/usr/local/nvidia/bin:/usr/local/bin:/usr/bin:/bin',
    gpuAccess: PLOINKY_GPU_ACCESS,
});

/** The host of the image the engine was written for, with `overrides` on top. */
export function agentLikeHost(overrides = {}) {
    return defineHost({ ...AGENT_HOST_SPEC, ...overrides });
}

export const HOST = agentLikeHost();

/** The frozen runner table of that host and wording: what the module-level table was. */
export const RUNNERS = createRunners({ host: HOST, messages: PLOINKY_MESSAGES });

export function getRunner(id) {
    if (typeof id !== 'string' || !Object.hasOwn(RUNNERS, id)) throw Object.assign(new Error(`Unknown runner: ${String(id)}`), { code: 'unknown_runner', details: { runner: id } });
    return RUNNERS[id];
}

/**
 * `createController` with the host, wording and gateway name this suite has always run with. The host takes the
 * test's `dataDir`; a test passes `host` (and `messages`) to run against another one, and `runRoot` or `hostOverrides`
 * to change one value of this one.
 */
export function createController({ runRoot, hostOverrides = {}, ...options } = {}) {
    const host = options.host ?? agentLikeHost({
        ...(options.dataDir ? { dataDir: options.dataDir } : {}),
        ...(runRoot ? { runRoot } : {}),
        ...hostOverrides,
    });
    return createEngineController({ messages: PLOINKY_MESSAGES, gatewayModelName: GATEWAY_MODEL_NAME, ...options, host });
}
