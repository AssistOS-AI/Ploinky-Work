// The public entry of the local-llm engine (`pworker/local-llm`): one controller
// for one on-demand model runner, composed from what a host gives it. The host
// says where things are (`defineHost`), the messages how the engine words what
// names its surroundings (`defineMessages`) and the limits what the machine
// leaves it (`defineLimits`). With the defaults the engine knows no container,
// no gateway and no limit.
//
// `./local-llm/internal/*` reaches the modules below this one; it is unstable
// and for the agent that ships with the engine only.

import path from 'node:path';

import { createController } from './controller/deployments.mjs';
import { createHardwareQueryReaper } from './controller/hardware.mjs';
import { createRunnerInstaller } from './controller/runnerInstaller.mjs';
import { loadRunnerLocks } from './controller/runnerLock.mjs';
import { DRAIN_QUEUE_WAIT_MS, DRAIN_RUNNER_GRACE_MS, HARDWARE_QUERY_REAP_MS } from './drainBudget.mjs';
import { defineHost } from './host.mjs';
import { IDENTITY_LIMITS, defineLimits } from './limits.mjs';
import { DEFAULT_MESSAGES, defineMessages } from './messages.mjs';
import { createRunners } from './runners/index.mjs';

/** The stages of a drain that belong to the engine; a host adds its own on top. */
export const ENGINE_DRAIN = Object.freeze({
    queueWaitMs: DRAIN_QUEUE_WAIT_MS,
    hardwareQueryReapMs: HARDWARE_QUERY_REAP_MS,
    runnerGraceMs: DRAIN_RUNNER_GRACE_MS,
});

/**
 * A controller for `host`. `host`, `messages` and `limits` may be the values
 * `defineHost`, `defineMessages` and `defineLimits` return, or the plain
 * objects they take (each is validated, and the call is idempotent).
 * `gatewayModelName` is the model name the overview publishes (null: none).
 * `env` is used by reference and never copied, so a switch the operator
 * changes is seen at the next read.
 *
 * `seams` are the options of `createController`, and win over what this
 * function composes: the runner table, the runner locks, the installer, the
 * hardware reaper and every reader or clock a test replaces.
 */
export function createLocalLlmController({ host, messages = DEFAULT_MESSAGES, limits = IDENTITY_LIMITS, gatewayModelName = null, env = process.env, ...seams } = {}) {
    const definedHost = defineHost(host);
    const definedMessages = defineMessages(messages);
    const definedLimits = defineLimits(limits);
    const dataDir = seams.dataDir ?? definedHost.dataDir;
    const hostArch = seams.hostArch ?? process.arch;
    // One reaper for every hardware query of this controller, so one drain waits for all of them.
    const reaper = seams.reaper ?? createHardwareQueryReaper();
    const runners = seams.runners ?? createRunners({ host: definedHost, messages: definedMessages, limits: definedLimits });
    // The image's lock and the host's lock for this architecture; the engine has no default location for either.
    const runnerLocks = seams.runnerLocks ?? loadRunnerLocks({ image: definedHost.imageRunnerLockFile, agent: definedHost.agentRunnerLockFile(hostArch) });
    const installWarnings = seams.installWarnings ?? { write: () => {} };
    const installer = seams.installer ?? createRunnerInstaller({
        lock: runnerLocks,
        cacheRoot: path.join(dataDir, 'runners'),
        runRoot: definedHost.runRoot,
        uv: definedHost.uv,
        python: definedHost.python,
        platform: definedHost.platform,
        onWarning: (message) => installWarnings.write(message),
    });
    return createController({
        host: definedHost,
        messages: definedMessages,
        limits: definedLimits,
        gatewayModelName,
        env,
        hostArch,
        dataDir,
        reaper,
        runners,
        runnerLocks,
        installWarnings,
        installer,
        ...seams,
    });
}

export { LocalLlmError, serializeError } from './errors.mjs';
export { OBSERVATION_ID_RE } from './promptObservations.mjs';
export { defineHost, linuxPlatform } from './host.mjs';
export { darwinPlatform, platformFor } from './darwinPlatform.mjs';
export { DEFAULT_MESSAGES, MESSAGE_KEYS, defineMessages } from './messages.mjs';
export { IDENTITY_LIMITS, LIMIT_HOOKS, defineLimits } from './limits.mjs';
export { RUNNER_IDS, createRunners, runnerSummaries } from './runners/index.mjs';
export { LMSTUDIO_SWITCH } from './runners/lmStudio.mjs';
export { VLLM_UNIFIED_SWITCH } from './runners/vllm.mjs';
export { createStateStore } from './controller/stateStore.mjs';
export { loadSeedCatalog, validateModel } from './controller/catalog.mjs';
export { createRunnerInstaller } from './controller/runnerInstaller.mjs';
export { loadRunnerLocks } from './controller/runnerLock.mjs';
export { readSnapshot } from './controller/hardware.mjs';
export { computeDiagnostic } from './controller/computeDiagnostics.mjs';
export { createObservationClock, createStartupObservations } from './controller/startupObservations.mjs';
export { createController } from './controller/deployments.mjs';
