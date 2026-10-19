// The limits a deployment puts on the engine, as one frozen value
// (`defineLimits`): a set of hooks the engine calls at fixed points and whose
// results it uses as they are. The engine itself knows no limit. The default
// (`IDENTITY_LIMITS`) caps nothing, refuses nothing, decorates nothing, adds
// nothing to a runner's environment and watches nothing, so an engine without
// a limits value behaves as on a machine that has none.
//
// Every hook receives what the engine holds at its call point, so a deployment
// can reproduce its own results, key order and wording exactly. A hook keeps
// its state in its own closure, one per engine instance: the engine keeps none
// for it. A hook that throws is the deployment's own fault and propagates,
// except `guardSample`, whose throw the engine counts as "no reading".
//
// | Hook | Called by | Inputs | Result |
// | --- | --- | --- | --- |
// | capCores | physicalCoreCount, performanceCoreCount | count, { fsApi } | the count to use |
// | observeSnapshot | readSnapshot | { fsApi, env } | an observation or null |
// | decorateSnapshot | readSnapshot | { snapshot, fsApi, env, observation } | the snapshot to publish |
// | memoryPool | cpuPool | memory, snapshot, { amount } | null, or the capped pool |
// | refuseProfile | admit, the APU policy | { stage, profile, snapshot, refuse } | a refusal or null |
// | admit | admit | { stage, ... } (see below) | a refusal, a share or null |
// | effective | admit | { snapshot } | { memory } |
// | refuseResourceRequest | the Ollama and vLLM policies | { policy, gpu, ..., refuse, gib } | a refusal or null |
// | overview | the controller's overview | { snapshot, env } | an object or null |
// | runnerEnvironment | the controller's runner environment | { env, profile, adapterEnvironment } | the adapter layer |
// | guardSample | the memory guard | () | a sample or null |
// | guardBreach | the memory guard | sample, { check, ... } | see below |
// | vllmQualification | admit (a runner that requires it) | { ...runnerContext, runner, gpu } | { qualified, code, fix } |
//
// capCores(count, { fsApi }) -> number
//   Called with the core count the engine found (never null) at the end of
//   physicalCoreCount and of performanceCoreCount (not when the latter finds no
//   distinct performance class). `fsApi` is the reader the count used.
//
// observeSnapshot({ fsApi, env }) -> observation | null
//   Called once per snapshot, after the disk and GPU reads and before the
//   engine reads memory and counts cores. A deployment that must record a fact
//   even when a later read of the same snapshot throws takes it here; the
//   engine does not look at the result and hands it to decorateSnapshot as
//   `observation`. `fsApi` is never undefined (the engine passes `fs` when none
//   was given).
//
// decorateSnapshot({ snapshot, fsApi, env, observation }) -> snapshot
//   `observation` is what observeSnapshot returned for this snapshot. It is not
//   called when a read between the two throws. `snapshot` has the keys at, gpu, memory, disk, cpus, cores, in this order.
//   A deployment that adds a key appends it (a key assigned to an existing
//   property keeps its place), may replace `gpu` when it knows the GPU cannot
//   be used, and may attach non-enumerable facts for its other hooks.
//
// memoryPool(memory, snapshot, { amount }) -> null | { totalBytes, availableBytes, capped, warning }
//   `memory` has finite totalBytes and availableBytes (the engine handles
//   unreadable memory itself). Null means the physical figures stand. `warning`
//   is a sentence the CPU policies add to their warnings when `capped`; `amount`
//   formats a byte count as the policies do.
//   A result may carry `reserveTotalBytes` and `reserveAvailableBytes`: the
//   capped figures then bound only what a model may need, while the host
//   reserve and the floor are computed from these figures. A result without
//   them has its own figures decide, and `physicalTotalBytes` keeps its
//   meaning (the denominator of a vLLM share).
//
// refuseProfile({ stage, profile, snapshot, refuse }) -> result | null
//   `stage` is 'admission' (before any policy runs, every profile) or 'policy'
//   (inside the APU policy, after its device-memory check). `refuse(status,
//   constraint, reason)` builds the result in the shape of that call point.
//
// admit(context) -> by stage
//   Common inputs: stage, profile, snapshot, env, result (the admission
//   result builder). Stages, in the order of the checks:
//   'before-policy'  after the runner and source checks and refuseProfile: a
//                    refusal (a result) or null.
//   'vulkan'         in a Vulkan profile, after the runner has been found to
//                    have a policy for it: a refusal or null.
//   'gpu'            on a dedicated or unified GPU, after the engine has checked
//                    the GPU is usable and matches the profile: also `selected`
//                    and `gpu` (the GPU as read). Returns null (no share
//                    applies), { refusal } (a result), or { share: { gpu } },
//                    where `gpu` is the view the policies then size against.
//                    The engine then asks the runner's qualification (below)
//                    on the GPU as read, and substitutes `share.gpu`.
//
// effective({ snapshot }) -> { memory }
//   The memory view the dedicated, unified and Vulkan policies size against.
//   The view may carry `reserveTotalBytes` and `reserveAvailableBytes` as a
//   memoryPool result does; the unified policies take their reserve and floor
//   from them.
//
// refuseResourceRequest(context) -> result | null
//   `policy` is 'ollama' (pinned GPU layers; also `estimate` and `marginBytes`)
//   or 'vllm' (also `params`, `gpuMemoryUtilization`, `denominator`). All carry
//   `gpu` as sized, `refuse(status, reason)` (the policy's result with its own
//   estimate and warnings) and `gib` (the policy's byte formatter).
//
// overview({ snapshot, env }) -> object | null
//   Published as the `budget` key of the pool limits when not null.
//
// runnerEnvironment({ env, profile, adapterEnvironment }) -> object
//   `adapterEnvironment` is a copy of what the runner adapter asked for. The
//   result replaces it in the runner's environment, so a deployment removes and
//   adds variables in one place. A throw refuses the start.
//
// guardSample() -> sample | null
//   A fresh reading of what the guard watches. A throw counts as null. The first
//   sample, taken when a guard starts, decides its cadence: `sample.watched`
//   true makes the guard sample on its short interval and call guardBreach
//   with check 'headroom' on every tick.
//
// guardBreach(sample, context) -> by check
//   'watchable'  (APU guard, every tick, `profile` given): null, or { message }
//                to stop the runner with that message.
//   'headroom'   (`memory` and `available` given, only for a watched guard):
//                null (nothing to adjust), { available } (the figure to
//                compare with the floor), or { unreadable: message }. After two
//                consecutive unreadable results the runner is stopped with the
//                last message.
//
// vllmQualification
//   A function, or null. A runner flagged `requiresShareQualification` is
//   admitted under a GPU share only when it returns { qualified: true }; a
//   refusal carries `code` and `fix`. With no function bound, the engine refuses
//   such a runner under a share.

import { LocalLlmError } from './errors.mjs';

const HOOKS = {
    capCores: (count) => count,
    observeSnapshot: () => null,
    decorateSnapshot: ({ snapshot }) => snapshot,
    memoryPool: () => null,
    refuseProfile: () => null,
    admit: () => null,
    effective: ({ snapshot }) => ({ memory: snapshot?.memory || {} }),
    refuseResourceRequest: () => null,
    overview: () => null,
    runnerEnvironment: ({ adapterEnvironment }) => adapterEnvironment,
    guardSample: () => null,
    guardBreach: () => null,
};

export const LIMIT_HOOKS = Object.freeze([...Object.keys(HOOKS), 'vllmQualification']);

/** The limits of an engine instance: the identity hooks with `overrides` (a function per hook) on top. Unknown keys throw. */
export function defineLimits(overrides = {}) {
    if (overrides === null || typeof overrides !== 'object' || Array.isArray(overrides)) {
        throw new LocalLlmError('invalid_limits', 'Limits must be an object of functions', { key: '(limits)' });
    }
    const limits = { ...HOOKS, vllmQualification: null };
    for (const [key, value] of Object.entries(overrides)) {
        if (!LIMIT_HOOKS.includes(key)) throw new LocalLlmError('invalid_limits', `Unknown limits hook ${key}`, { key });
        const allowed = typeof value === 'function' || (key === 'vllmQualification' && value === null);
        if (!allowed) throw new LocalLlmError('invalid_limits', `Limits hook ${key} must be a function`, { key });
        limits[key] = value;
    }
    return Object.freeze(limits);
}

export const IDENTITY_LIMITS = defineLimits();
