// The wording a deployment gives to the engine's messages that name its own
// surroundings: how an operator turns a switch on, what to restart, where to
// look for the GPU grant. The engine defaults are neutral; a deployment that
// speaks differently passes `defineMessages(overrides)`, one function per key
// returning the whole sentence, so its wording can be kept byte for byte.
// Wording that only describes the engine's own behavior stays in the engine.

import { LocalLlmError } from './errors.mjs';

const DEFAULTS = {
    // The chat endpoint has no ready model.
    notReady: () => 'No local model is ready; run one first.',
    // nvidia-smi failed and the host says why GPU access is unavailable; `reason` is null when it gives none.
    gpuAccessReported: (reason) => `The host reports: ${reason || 'GPU access is unavailable; the cause is unknown.'}`,
    // nvidia-smi is not there and the host reported nothing.
    gpuAbsent: () => 'No GPU is available to this server.',
    // The end of a Hugging Face refusal sentence for a gated or private repository.
    hfTokenHint: () => 'provide a Hugging Face token (HF_TOKEN) and restart the server',
    // A GPU render node is attached, but the host declares no Vulkan runtime.
    vulkanRuntimeMissing: () => 'A GPU render node is attached, but this host declares no supported Vulkan runtime.',
    // An AMD integrated GPU is attached and its operator switch is off.
    apuSwitchOff: (deviceName, switchName, switchValue) => `An AMD integrated GPU (${deviceName}) is attached. `
        + `APU use is off: set ${switchName}=${switchValue} and restart to enable it.`,
    // The sentence that ends vLLM's "not enabled on unified memory" reasons.
    vllmUnifiedSwitchHint: (switchName, switchValue) => `Set ${switchName}=${switchValue} to turn it on; it has not been measured on this hardware.`,
    // The sentence that ends LM Studio's "not enabled" reason.
    lmStudioEnableHint: (switchName, switchValue) => `Set ${switchName}=${switchValue} and restart to enable it.`,
    // LM Studio staged an update in its home, which would change the pinned bytes.
    lmStudioStagedUpdate: () => 'LM Studio has staged an update in its home; restart to rebuild LM Studio from its verified download.',
    // The two restart hints of a compute diagnostic: the GPU came back, or something else changed.
    profileRestartGpuRecovered: () => 'GPU access is available now; restart to select it. Stopping and starting the model does not change the compute profile.',
    profileRestartChanged: () => 'After changing GPU access, machine or provider, restart to select a new compute profile.',
    // Admission refusals that say the committed profile no longer matches the hardware.
    restartToUseDevice: (profile) => `This engine started with the ${profile} profile; restart it to use the current device or run on the CPU.`,
    restartToRunOnCpu: (profile) => `This engine started with the ${profile} profile; restart it to run on the CPU.`,
    restartForMemoryModel: (current, selected) => `The GPU now reports ${current} memory, but this engine started with the ${selected} profile; restart it to use it.`,
};

export const MESSAGE_KEYS = Object.freeze(Object.keys(DEFAULTS));

/** The messages of an engine instance: the neutral defaults with `overrides` (a function per key) on top. Unknown keys throw. */
export function defineMessages(overrides = {}) {
    if (overrides === null || typeof overrides !== 'object' || Array.isArray(overrides)) {
        throw new LocalLlmError('invalid_messages', 'Messages must be an object of functions', { key: '(messages)' });
    }
    const messages = { ...DEFAULTS };
    for (const [key, value] of Object.entries(overrides)) {
        if (!Object.hasOwn(DEFAULTS, key)) throw new LocalLlmError('invalid_messages', `Unknown message key ${key}`, { key });
        if (typeof value !== 'function') throw new LocalLlmError('invalid_messages', `Message ${key} must be a function`, { key });
        messages[key] = value;
    }
    return Object.freeze(messages);
}

export const DEFAULT_MESSAGES = defineMessages();
