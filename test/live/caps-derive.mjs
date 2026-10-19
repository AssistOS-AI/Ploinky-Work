// Pure helpers of test/live/caps-live.mjs: how the live caps suite derives every cap value from the preview estimates of the machine
// it runs on (never from numbers measured on another machine), and how it reads the text and argv the product prints. No process,
// no file, no network: test/local-llm-host/caps-live-guard.test.mjs runs them on fixed inputs.
//
// The thresholds below are the dedicated llama.cpp policy's own (lib/local-llm/controller/admission.mjs, admitLlamaServer; the other dedicated runners' policies keep the same two margins): a request is
// `incompatible` when its GPU need is above 0.97 of the GPU, and `insufficient-now` when it is above the free GPU memory less a 256 MiB
// margin. A cap bounds what a model may plan for; the margins and the memory the machine really has free stay real, so under a
// `gpuMemory` cap the total is the cap and the free figure is the cap plus that margin (caps.mjs, `admit`): on an idle GPU a need is
// `incompatible` above 0.97 of the cap and `ok` otherwise, and there is no band that is refused for now, where fit never acts. The
// `ram` cap works the same way for the dedicated policy's 1 GiB margin (`effective`). The helpers say what each plan expects the
// preview to show, so a mismatch is a finding, not noise.

import { FIT_CONTEXT_FLOOR } from '../../lib/local-llm/controller/fit.mjs';
import { RAM_MARGIN_BYTES } from '../../lib/local-llm/controller/admission.mjs';

export { FIT_CONTEXT_FLOOR, RAM_MARGIN_BYTES };
export const MIB = 1024 ** 2;
export const GIB = 1024 ** 3;
export const GPU_FIT_RATIO = 0.97;
const MIN_CONTEXT = 512;

/** The need a preview's estimate puts on host memory: `needBytes` where the policy has one (Metal), else `ramBytes`. */
export const ramNeedOf = (estimate) => (Number.isFinite(estimate?.needBytes) ? estimate.needBytes : estimate?.ramBytes);

/** The context fit gives a default `ctx` at its first halving. */
export const halvedContext = (ctx) => Math.max(MIN_CONTEXT, Math.floor(ctx / 2));

/** The least a `gpuMemory` cap can be for a request needing `gpuBytes` to be admitted outright (`ok`), with `slack` (a fraction) to spare. */
export const gpuCapAdmitting = (gpuBytes, slack = 0.03) => Math.ceil((gpuBytes / GPU_FIT_RATIO) * (1 + slack));

/** What a dedicated preview says of a request that needs `gpuBytes` under a `gpuMemory` cap (the GPU itself being large and idle). */
export function gpuVerdict(gpuBytes, cap) {
    return gpuBytes > cap * GPU_FIT_RATIO ? 'incompatible' : 'ok';
}

/**
 * The memory the available figure of a refusal for now carries beyond the room under a `ram` cap: the dedicated runners' policies keep
 * 1 GiB free, and the cap adds that margin to the figure it gives the policy (the unified and pool policies keep none, and the cap adds none).
 */
export const ramAdmitMargin = (profile) => (profile === 'dedicated' ? RAM_MARGIN_BYTES : 0);

/**
 * L5 on the pool profiles (cpu, metal): a `ram` cap between the need of the default context and of the halved one, so the default is
 * `incompatible`, the halved context fits, and fit lowers `ctxSize` once. Null when the two needs do not differ.
 */
export function planRamFit({ ctx, needDefault, needHalved }) {
    const halved = halvedContext(ctx);
    if (halved >= ctx || !(needHalved + MIB < needDefault)) return null;
    return { cap: Math.floor((needDefault + needHalved) / 2), from: ctx, to: halved };
}

/**
 * L5 on dedicated: fit halves a default context only down to FIT_CONTEXT_FLOOR while layers can still move (the GPU being what refuses).
 * A cap that makes the default context `incompatible` and the halved one `ok` needs the KV difference to be wider than the slack
 * `gpuCapAdmitting` leaves above the halved need. `gpuAt(ctx)` is the GPU need of the default layers at a context.
 * Returns `{ cap, from, to }` or `{ skip }` with the reason.
 */
export function planGpuFitContext({ ctx, gpuAt }) {
    const to = halvedContext(ctx);
    if (ctx <= FIT_CONTEXT_FLOOR) return { skip: `the default context ${ctx} is not above the fit context floor ${FIT_CONTEXT_FLOOR}, so a GPU refusal makes fit move layers first (case L6), not the context` };
    if (to < FIT_CONTEXT_FLOOR) return { skip: `halving ${ctx} goes below the fit context floor ${FIT_CONTEXT_FLOOR}` };
    const cap = gpuCapAdmitting(gpuAt(to));
    if (gpuVerdict(gpuAt(ctx), cap) !== 'incompatible') {
        return { skip: `with a gpuMemory cap that admits ctx ${to} (${cap} bytes) the default ctx ${ctx} is ${gpuVerdict(gpuAt(ctx), cap)}, not incompatible: the KV cache of this model is too small to open a gap wider than the slack of the cap` };
    }
    return { cap, from: ctx, to };
}

/**
 * L6: a `gpuMemory` cap under which the default layers cannot all be offloaded at the context fit stops at, so fit lowers `nGpuLayers`.
 * `gpuAtLayers(n)` is the GPU need at n layers (and that context); `layers` the model's layer count. Fit takes layers one at a time from
 * `layers - 1` while the result is `incompatible`, so it ends at the largest n with `gpuAtLayers(n) <= 0.97 * cap`, which is `ok`
 * (launchable). The plan with the end state closest to a third of the layers is returned. Null when no cap makes layers move.
 */
export function planGpuFitLayers({ layers, gpuAtLayers }) {
    const plans = [];
    for (let target = 1; target < layers; target += 1) {
        const cap = gpuCapAdmitting(gpuAtLayers(target));
        if (gpuVerdict(gpuAtLayers(layers), cap) !== 'incompatible') break; // a larger target only raises the cap
        let end = null;
        for (let n = layers - 1; n >= 1; n -= 1) if (gpuVerdict(gpuAtLayers(n), cap) !== 'incompatible') { end = n; break; }
        if (end !== null) plans.push({ cap, expectLayers: end, expectStatus: gpuVerdict(gpuAtLayers(end), cap) });
    }
    const wanted = Math.max(1, Math.floor(layers / 3));
    return plans.sort((a, b) => Math.abs(a.expectLayers - wanted) - Math.abs(b.expectLayers - wanted))[0] ?? null;
}

/**
 * L5's explicit `--ctx A` under `mode: "fit"`: the pool profiles (cpu, metal) refuse it, because fit never lowers an explicit value and
 * ctxSize is all they can lower; the dedicated profile keeps the explicit ctx and moves layers to the CPU instead (DS007, DS008).
 */
export const explicitCtxExpectation = (profile) => (profile === 'dedicated' ? 'admitted' : 'refused');

/** The `param from -> to` steps a preview prints after "fit lowered:" (empty when it prints none). */
export function fitLoweredSteps(stdout) {
    const line = /fit lowered: ([^\n]*)/.exec(String(stdout ?? ''))?.[1] ?? '';
    return [...line.matchAll(/(\w+) (\d+) -> (\d+)/g)].map((match) => ({ param: match[1], from: Number(match[2]), to: Number(match[3]) }));
}

/**
 * Whether the preview of an explicit ctx (`stdout`, the CLI's text for `model`) says what `explicitCtxExpectation` expects: refused as
 * incompatible with no fit (pool profiles), or admitted (`ok`) with fit lowering `nGpuLayers` and never `ctxSize` (dedicated).
 */
export function explicitCtxPreviewMatches({ profile, stdout, model }) {
    const text = String(stdout ?? '');
    const head = new RegExp(`^${String(model).replace(/[.]/g, '\\.')}: (\\S+?)[,\\s]`, 'm').exec(text)?.[1] ?? null;
    const steps = fitLoweredSteps(text);
    if (explicitCtxExpectation(profile) === 'refused') return head === 'incompatible' && steps.length === 0;
    const layers = steps.find((step) => step.param === 'nGpuLayers');
    return head === 'ok' && Boolean(layers) && layers.to < layers.from && !steps.some((step) => step.param === 'ctxSize');
}

/** Whether a run admitted with an explicit ctx launched with it and with fewer GPU layers than the model has (`parsed` is parseRunnerArgv's). */
export const explicitCtxLaunchMatches = ({ parsed, ctx, layers }) => parsed?.ctx === ctx && Number.isFinite(parsed?.layers) && Number.isFinite(layers) && parsed.layers < layers;

/** What identifies a deployment in a status answer (`{ deployment }`), or null when there is none. */
export const deploymentIdentity = (status) => {
    const deployment = status?.deployment;
    if (!deployment) return null;
    return { id: deployment.id ?? null, phase: deployment.phase ?? null, updatedAt: deployment.updatedAt ?? null, hasRunner: Boolean(deployment.runner) };
};

/**
 * Whether a refused run left the deployment as it was: no deployment before and none after, or the same one (id and timestamp), in the
 * phase it had, and still without a runner. A refusal must neither create a deployment nor change the remembered one.
 */
export function deploymentUnchanged(before, after) {
    if (!before || !after) return !before && !after;
    return before.id === after.id && before.updatedAt === after.updatedAt && before.phase === after.phase && before.hasRunner === after.hasRunner && after.hasRunner === false && after.phase === 'idle';
}

/**
 * Bytes from the figure of a refusal for now: "3.35 GiB is available now" (RAM), "3.35 GiB is free now" or "Only 3.35 GiB of GPU memory is
 * free" (GPU), in GiB or MiB. Null when the reason has none (a disk shortage, for example).
 */
export function availableNowBytes(reason) {
    const text = String(reason ?? '');
    const match = /(\d+(?:\.\d+)?) (GiB|MiB) is (?:available|free|reported free) now/.exec(text) ?? /Only (\d+(?:\.\d+)?) (GiB|MiB) of GPU memory is free/.exec(text);
    if (!match) return null;
    return Math.round(Number(match[1]) * (match[2] === 'GiB' ? GIB : MIB));
}

/**
 * How far below the physical figure the figure a refusal reports must be to count as the cap's doing: the machine's free memory moves
 * between two snapshots, so a smaller gap is noise.
 */
export const CAP_LIMIT_SLACK_BYTES = 128 * MIB;

/**
 * Whose doing is a refusal for now (`insufficient-now`)? The policies are given the physical figure, or the cap's figure when that is
 * smaller (caps.mjs), and print the one they got. A reported figure clearly below the physical one of the snapshot (`hardware`, the
 * overview's: the GPU's free memory for a reason about GPU memory, else the available RAM) is therefore the cap's room, and the cap, not
 * the machine, is what limits the request: `cause: 'cap'`. Otherwise it is the machine's state (`'machine'`); with no figure on either
 * side nothing can be said (`'unknown'`: a disk shortage, for example, which is the machine's too).
 */
export function insufficientNowCause(admission, hardware) {
    const reason = admission?.reason ?? '';
    const gpuReason = /GPU memory/.test(reason);
    const physical = gpuReason ? hardware?.gpu?.freeBytes : hardware?.memory?.availableBytes;
    const reported = availableNowBytes(reason);
    if (reported === null || !Number.isFinite(physical)) return { cause: 'unknown', reported, physical: Number.isFinite(physical) ? physical : null, kind: gpuReason ? 'gpu' : 'ram' };
    return { cause: reported < physical - CAP_LIMIT_SLACK_BYTES ? 'cap' : 'machine', reported, physical, kind: gpuReason ? 'gpu' : 'ram' };
}

/**
 * The status of a finished case that ended `fail`. The engine's memory guard stopping the runner for the machine's memory (`guardStopped`)
 * means the case met a busy machine, not a caps result, so it is `inconclusive`: but only when no check failed. A check that failed is
 * a finding about the product and stays a `fail`, whatever the guard did afterwards.
 */
export function settleFailedCase({ status, checks, guardStopped }) {
    const failedCheck = (checks ?? []).some((item) => item?.status === 'fail');
    return status === 'fail' && guardStopped === true && !failedCheck ? 'inconclusive' : status;
}

/**
 * L7's `ram` cap: the need of the model plus 64 MiB, and never less than 1.25 times the resident memory the runner was measured to hold
 * earlier in the run (`rssBytes`, from L11, when there is one), so the runner's own steady RSS (a CUDA runner's can exceed its estimate) never
 * trips the live ram guard. A second instance of the model is then refused for now because of `ourUse` when its need is above the room
 * the cap leaves, `cap - rss` (the 1 GiB the dedicated policy keeps free is added to the figure the cap gives it and taken off again by the policy).
 * Returns `{ cap, room, secondRefused }`; `secondRefused` is false when the cap leaves room for a second instance, so the case cannot show ourUse.
 */
export function planOurUseCap({ need, rssBytes }) {
    const measured = Number.isFinite(rssBytes) && rssBytes > 0 ? Math.ceil(rssBytes * 1.25) : 0;
    const cap = Math.max(Math.ceil(need + 64 * MIB), measured);
    const room = cap - (Number.isFinite(rssBytes) && rssBytes > 0 ? rssBytes : 0);
    return { cap, room, secondRefused: Number.isFinite(rssBytes) && rssBytes > 0 ? need > room : null };
}

/** The values of a runner's command line (a string, or its tokens) that the caps suite checks. Each is a number or null. */
export function parseRunnerArgv(command) {
    const tokens = Array.isArray(command) ? command.map(String) : String(command).trim().split(/\s+/);
    const after = (...names) => {
        for (let index = 0; index < tokens.length - 1; index += 1) {
            if (names.includes(tokens[index]) && /^-?\d+$/.test(tokens[index + 1])) return Number(tokens[index + 1]);
        }
        return null;
    };
    return { ctx: after('-c', '--ctx-size'), threads: after('--threads', '-t'), parallel: after('-np', '--parallel'), layers: after('-ngl', '--n-gpu-layers') };
}

/** Whether `actual` is within `fraction` of `reference` (both whole numbers of bytes). */
export const within = (actual, reference, fraction) => Number.isFinite(actual) && Number.isFinite(reference) && reference > 0 && Math.abs(actual - reference) <= reference * fraction;
