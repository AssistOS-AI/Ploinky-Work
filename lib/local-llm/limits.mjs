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
// | capThreads | the llama-server runners' default thread count | threads, { profile } | the thread count to use |
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
// | capParams | the controller's parameter resolution | { runnerId, profile, params, explicit, model } | the params to use |
// | fitParams | every admission of the controller | { runnerId, profile } | null, or { order } |
// | vllmQualification | admit (a runner that requires it) | { ...runnerContext, runner, gpu } | { qualified, code, fix } |
//
// capCores(count, { fsApi }) -> number
//   Called with the core count the engine found (never null) at the end of
//   physicalCoreCount and of performanceCoreCount (not when the latter finds no
//   distinct performance class). `fsApi` is the reader the count used.
//
// capThreads(threads, { profile }) -> number
//   Called with the thread count a llama-server runner resolved by default
//   (the high-performance cores, or the physical cores less a margin), on every
//   profile, after the core counts and `capCores` (except for a count the
//   platform supplies itself, macOS `performanceCores`). It is not called for a
//   `threads` parameter the caller set: the engine hands those to `capParams`.
//
// capParams({ runnerId, profile, params, explicit, model }) -> params
//   Called once at the end of the controller's parameter resolution, so for a
//   preview, a Run, the overview and the work a Run redoes when the downloaded
//   file changes the model's training context. `params` is the normalized set:
//   the caller's values or, with none, the saved ones, over the model's
//   recommended values and the schema defaults; `explicit` is a frozen array of the keys
//   the caller itself set in this request (saved values and defaults are not in
//   it). The result is validated against the runner's schema again, and a
//   result outside it is refused as invalid parameters. The engine uses the
//   result for the admission and the launch and saves `params`, not the result,
//   so removing a cap restores the earlier values. A hook that refuses an
//   explicit value throws a LocalLlmError with code `invalid_params`.
//
// fitParams({ runnerId, profile }) -> null | { order }
//   Called by the controller before each admission it makes for a preview, the
//   overview, a Run and the checks a Run repeats before it starts. Null (the
//   identity) means the request is admitted as it is, with no extra work.
//   `{ order }` is an array of the parameters the engine may lower (`ctxSize`,
//   `parallel`; `nGpuLayers` is not lowered and is ignored), first to last: when the admission is
//   `incompatible` because of the request's size (never `insufficient-now`), the
//   engine lowers the first of them that can go lower, one halving at a time, and
//   admits again until the result is not `incompatible` or none can. It never
//   lowers a key the caller set in the request (`explicit` of capParams), and the
//   parameters a Run saves are the ones before fitting. A result that was lowered
//   carries `fit: { lowered: [{ param, from, to }] }` and a warning; one that
//   still does not fit names the smallest size tried. The steps are bounded.
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
//   with check 'headroom' on every tick. A runner whose admission recorded no
//   floor has no guard unless that first sample says both `watched: true` and
//   `standalone: true`: the guard then runs with no floor, compares no available
//   figure with one, and acts only on a `{ stop }` answer (below). The other
//   samples are read as before, so a sample with `watched` alone starts no guard
//   on a runner with no floor.
//
// guardBreach(sample, context) -> by check
//   'watchable'  (APU guard, every tick, `profile` given): null, or { message }
//                to stop the runner with that message.
//   'headroom'   (`memory` and `available` given, only for a watched guard):
//                null (nothing to adjust), { available } (the figure to
//                compare with the floor), { unreadable: message }, or
//                { stop: message }. After two consecutive unreadable results the
//                runner is stopped with the last message. After two consecutive
//                `stop` results it is stopped with the last `stop` message, on
//                every profile, whatever the floor and the pressure; any other
//                result (null, available, unreadable) starts that count again.
//
// vllmQualification
//   A function, or null. A runner flagged `requiresShareQualification` is
//   admitted under a GPU share only when it returns { qualified: true }; a
//   refusal carries `code` and `fix`. With no function bound, the engine refuses
//   such a runner under a share.

import { LocalLlmError } from './errors.mjs';

const HOOKS = {
    capCores: (count) => count,
    capThreads: (threads) => threads,
    capParams: ({ params }) => params,
    fitParams: () => null,
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
