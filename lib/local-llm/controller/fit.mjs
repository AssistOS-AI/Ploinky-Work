// Fit: lower the parameters a request did not set until admission stops calling it `incompatible` (DS007, `fitParams`).
//
// A pure helper: the controller hands it the parameters, the keys the caller set, a way to admit a parameter set and a way to
// normalize one, and gets back the admission result and the parameters it belongs to. Nothing here reads a limit; the limits
// only say whether, and in which order, the controller may fit (`fitParams`).
//
// What it lowers, in the order the limits give (the order DS007 documents is ctxSize, then nGpuLayers, then parallel), one step at a time:
//
//   ctxSize      halved, rounded down, never below the schema's minimum; while nGpuLayers comes later in the order and can still
//                move, never below FIT_CONTEXT_FLOOR either
//   nGpuLayers   one layer fewer, from the model's layer count down (a default of 99 means all of them), never below 1; only while the
//                refusal names the GPU (`reasonCode` `gpu_capacity`): fewer layers only raise the RAM a request needs
//   parallel     halved, rounded down, never below the schema's minimum; a context the lowered slots cannot hold goes down with them
//
// So with the DS007 order a default context goes down to the floor, then the layers move to the CPU until the GPU holds the rest,
// and only then does the context go lower: a model whose weights are the larger part keeps a useful context at a few layers fewer,
// and one whose KV cache is the larger part keeps its layers at a smaller context. At 0 layers the estimate counts no compute buffer
// and no CUDA context, so fit stops at 1.
//
// A key is lowered to its end (or to the floor) before the next one is touched. After each step the set is normalized again and admitted again, until
// the result is not `incompatible` or no key can be lowered; the number of steps is bounded. It never changes a key the caller set
// (`explicit`), and never runs on a result that is not `incompatible` (`insufficient-now` is the machine's state now, not the request's
// size) or whose estimate has no weights (a refusal of the runner or the source).
//
// A request that ends `incompatible` is returned as it was asked, with the smallest size tried in the reason. So is a refusal that
// lowering never changed the wording of: it was not about the sizes (a Metal device whose limit is unknown, memory that cannot be read),
// and fit says nothing about it.
//
//   fitAdmission(input) -> { admission, params }

// The layer count of a model whose entry does not say: the one the llama.cpp estimate sizes it with.
import { DEFAULT_LAYERS } from './admission.mjs';

/** The parameters fit can lower; a `fitParams` order names some of them. */
export const FIT_KEYS = Object.freeze(['ctxSize', 'nGpuLayers', 'parallel']);

/** The context a default goes down to before any layer moves, while there are layers to move (a context below it is not raised). */
export const FIT_CONTEXT_FLOOR = 8192;

/**
 * The most steps one fit takes. Every step lowers one key strictly, so a fit ends by itself well before this (about a dozen for a
 * 128k context and 16 slots, plus one per layer moved); the bound is the backstop against a schema or a hook that never lets it end.
 */
export const MAX_FIT_STEPS = 256;

// The figure a refusal of the hardware's size names, by its reasonCode (admission.mjs); a refusal with no such code names none.
const NAMED_FIGURE = Object.freeze({ gpu_capacity: 'gpu', ram_capacity: 'ram' });

const isWhole = (value) => Number.isSafeInteger(value) && value >= 0;

// A result the sizes of the request decided: a policy's estimate always carries the weights, a refusal of the runner, the
// source or the machine's state does not.
const isSized = (admission) => Number.isFinite(admission?.estimate?.weightsBytes);

const halve = (value, minimum) => Math.max(minimum, Math.floor(value / 2));

function minimumOf(schema, key, fallback) {
    const minimum = schema?.properties?.[key]?.minimum;
    return Number.isSafeInteger(minimum) ? minimum : fallback;
}

/**
 * `params` with `key` one step lower, or null when it cannot go lower (set by the caller, at its minimum, not an integer, not in the
 * schema; the layers also while the refusal is not the GPU's). `floor` is the least the context goes to at this step.
 */
function lowerKey(key, params, { schema, model, explicit, named }, floor = 0) {
    if (explicit.includes(key) || !schema?.properties?.[key] || !isWhole(params[key])) return null;
    if (key === 'nGpuLayers') {
        if (named !== 'gpu') return null;
        const layers = Number.isSafeInteger(model?.memory?.layers) && model.memory.layers > 0 ? model.memory.layers : DEFAULT_LAYERS;
        const next = Math.min(params.nGpuLayers, layers) - 1;
        return next >= Math.max(1, minimumOf(schema, key, 1)) ? { ...params, nGpuLayers: next } : null;
    }
    const next = halve(params[key], Math.max(floor, minimumOf(schema, key, 1)));
    if (next >= params[key]) return null;
    if (key !== 'parallel') return { ...params, [key]: next };
    // The model's training context is per slot: fewer slots hold less. A context the caller set stays theirs, so these slots
    // cannot be lowered; any other goes down to what the lowered slots hold (a value below the schema minimum fails to normalize).
    const room = Number.isSafeInteger(model?.contextLength) ? model.contextLength * next : null;
    if (room === null || params.ctxSize === undefined || params.ctxSize <= room) return { ...params, parallel: next };
    if (explicit.includes('ctxSize')) return null;
    return { ...params, parallel: next, ctxSize: Math.max(minimumOf(schema, 'ctxSize', 1), room) };
}

/** The next parameter set to try: the first key of `order` that can go one step lower and still normalizes, else null. */
function nextStep(order, params, context) {
    // The layers move before the context goes below the floor: while they still can, the context stops there.
    const layersLater = order.indexOf('nGpuLayers') > order.indexOf('ctxSize') && order.includes('ctxSize');
    const floor = layersLater && lowerKey('nGpuLayers', params, context) ? FIT_CONTEXT_FLOOR : 0;
    for (const key of order) {
        const lower = lowerKey(key, params, context, key === 'ctxSize' ? floor : 0);
        if (!lower) continue;
        try {
            return context.normalize(lower);
        } catch (error) {
            // This key cannot go lower here (a context the slots cannot hold); another may. Anything else is a fault.
            if (error?.code !== 'invalid_params') throw error;
        }
    }
    return null;
}

// What an admission says a request needs, on the GPU and in the RAM; a figure the estimate does not carry is null.
const needOf = (admission) => ({ gpu: admission?.estimate?.gpuBytes ?? null, ram: admission?.estimate?.ramBytes ?? null });
const namedBy = (admission) => NAMED_FIGURE[admission?.reasonCode] ?? null;
const fell = (now, before, figure) => now[figure] !== null && before[figure] !== null && now[figure] < before[figure];
// Whether the need went down: the figure the refusal names when it names one (fewer layers lower the GPU figure and raise the RAM
// one), else either figure.
const wentDown = (now, before, figure) => (figure ? fell(now, before, figure) : fell(now, before, 'gpu') || fell(now, before, 'ram'));

const sayLowered = (changes) => changes.map(({ param, from, to }) => `${param} from ${from} to ${to}`).join(', ');

/**
 * Admits `params` and, when the result is `incompatible` because of their size, lowers them until it is not.
 *
 * @param {object} input
 * @param {object} input.params         the parameters to admit, already normalized (and capped)
 * @param {string[]} input.explicit     the keys the caller set: never lowered
 * @param {string[]} input.order        the keys fit may lower, first to last (a subset of FIT_KEYS; others are ignored)
 * @param {object} input.schema         the runner's parameter schema for the profile (the minimums)
 * @param {object} input.model          the model entry (its training context and its layer count)
 * @param {(params: object) => object} input.admitWith  admission of a parameter set
 * @param {(params: object) => object} input.normalize  validates a parameter set, throwing an `invalid_params` error when it is not one
 * @param {number} [input.maxSteps]     the bound on the steps (MAX_FIT_STEPS)
 * @returns {{ admission: object, params: object }} a result that was lowered carries a `fit` of the changes and a warning that names them, with the lowered parameters; a request that still does not fit comes back as it was asked, its reason followed by the smallest size tried
 */
export function fitAdmission({ params, explicit, order, schema, model, admitWith, normalize, maxSteps = MAX_FIT_STEPS }) {
    const first = admitWith(params);
    if (first.status !== 'incompatible' || !isSized(first)) return { admission: first, params };
    const keys = FIT_KEYS.filter((key) => order.includes(key)).sort((a, b) => order.indexOf(a) - order.indexOf(b));
    const context = { schema, model, explicit, normalize };
    const firstNeed = needOf(first);
    const firstNamed = namedBy(first);
    let current = params;
    let admission = first;
    // Whether lowering ever moved the refusal: the need the estimate carries went down and the wording changed with it. A refusal that
    // is about the sizes names a need that changes with them; one that is not (a Metal device whose limit is unknown, memory that
    // cannot be read, a RAM figure that lowering the context does not touch) says the same at every size. A refusal that names its
    // figure is judged by that one: a RAM refusal the layers only raised did not move, a GPU refusal they lowered did.
    let reasonMoved = false;
    for (let step = 0; step < maxSteps && admission.status === 'incompatible'; step += 1) {
        const next = nextStep(keys, current, { ...context, named: namedBy(admission) });
        if (!next) break;
        current = next;
        admission = admitWith(current);
        if ((admission.reason !== first.reason || admission.reasonCode !== first.reasonCode) && wentDown(needOf(admission), firstNeed, firstNamed)) reasonMoved = true;
    }
    const changes = keys.filter((key) => current[key] !== params[key]).map((key) => ({ param: key, from: params[key], to: current[key] }));
    if (admission.status === 'incompatible') {
        // Not fitted: the request is refused as it was asked, so what is shown and recorded is the request, and the reason says how far fit went.
        // With nothing lowered, or nothing that moved the reason, the refusal is not one fit can speak to and stays as it was.
        if (!changes.length || !reasonMoved) return { admission: first, params };
        const smallest = keys.filter((key) => isWhole(current[key])).map((key) => `${key} ${current[key]}`).join(', ');
        const reason = `${first.reason} Fit also tried lowering ${sayLowered(changes)}, and it still does not fit; the smallest size tried was ${smallest}.`;
        return { admission: Object.freeze({ ...first, reason }), params };
    }
    if (!changes.length) return { admission, params };
    // Not `ok` (`insufficient-now`): the sizes fit this machine, but not what is free now.
    const said = admission.status === 'ok'
        ? `Fit lowered ${sayLowered(changes)} so that the model fits; a value you set is never lowered.`
        : `Fit lowered ${sayLowered(changes)} so that the model fits this machine's size; it does not fit in what is free now.`;
    const warnings = [...admission.warnings, said];
    const fit = Object.freeze({ lowered: Object.freeze(changes.map((change) => Object.freeze(change))) });
    return { admission: Object.freeze({ ...admission, warnings: Object.freeze(warnings), fit }), params: current };
}
