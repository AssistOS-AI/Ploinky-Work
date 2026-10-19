// The hardware caps of one pworker home: how much GPU memory and RAM local models may plan for, set in the configuration as
// `localLlm.caps`. They size and refuse: admission is run against the capped figures, so a model that does not fit the config is
// refused even when the machine could hold it. Nothing here limits a running process except the `ram` backstop below, which stops a runner that uses more than its cap (see DS008, "Caps").
//
//   parseCaps(section)         pure; null when there are no caps, else the frozen parsed caps, or a LocalLlmError `invalid_caps`
//                              that names the key. `mode` is "refuse" (the default) or "fit"; on its own it caps nothing.
//   defineCapLimits(caps, o)   the engine limits (defineLimits) that apply parsed caps; with no caps the host passes none at all. `o` says how
//                              the runners' memory is read for the `ram` guard (below); without it that guard is not bound
//   capsDigest(caps)           a short stable digest of the caps, for the identity of a cached answer
//
// Caps keys: `gpuMemory` and `ram`, each as bytes (a safe integer of at least 1 MiB), a string with a binary unit ("12GiB",
// "512MiB", "1.5GiB") or a fraction 0 < f <= 1 of the physical figure (the first one read, then frozen). `gpuMemory` applies to the
// dedicated and unified NVIDIA profiles only (the host refuses it on macOS); `ram` to every profile. `threads`, `maxContext` and
// `maxParallel` are whole numbers of at least 1 that bound the llama-server flags: `--threads` (the default is lowered by
// `capThreads`), `--ctx-size` (`ctxSize`) and `-np` (`parallel`). A default or saved value above one is lowered to it; a value the
// caller set above it is refused (`invalid_params`, naming the cap), by `capParams` before the model starts. `maxContext` may not be
// below the smallest context any runner pworker offers accepts.
//
// `mode` says what happens to a request that does not fit. "refuse" refuses it, naming what it hit. "fit" lowers the parameters the
// caller did not set (`ctxSize` down to 8192, then `nGpuLayers` while the GPU refuses, then the rest of `ctxSize`, then `parallel`; see fit.mjs) until the admission is no longer
// `incompatible`, and refuses only when nothing more can be lowered (`fitParams`, DS007). It acts on every `incompatible` result, whether a
// cap or the machine caused it, and never on `insufficient-now`; a value the caller set is never lowered.
//
// With a `ram` cap the memory the runners this host started already hold (runnerMemory.mjs) counts against it: the memory a model may
// plan for now is the smaller of what the machine has available and the cap less that use. The reserve figures stay physical.
//
// The `ram` cap is also a backstop while a runner runs: the guard samples the resident memory of the runners this host started
// (runnerMemory.mjs) and stops the runner after two consecutive readings above the cap (`guardBreach` answers `{ stop }`, DS007). It
// never answers `{ available }`, so the floor check on the machine's available memory is unchanged. It is a backstop, not a limit: a
// load can allocate faster than it samples.
//
// A cap bounds what a model may plan for. The host reserve and the memory-guard floor stay those of the physical machine: the
// pool and the memory view carry the reserve figures (`reserveTotalBytes`, `reserveAvailableBytes`) beside the capped ones (DS007, `memoryPool` and `effective`).

import { createHash } from 'node:crypto';

import { LocalLlmError, defineLimits } from '../local-llm/index.mjs';
import { createRunnerMemoryReader } from './runnerMemory.mjs';
import {
    LLAMA_SERVER_CPU_PARAM_SCHEMA, LLAMA_SERVER_METAL_PARAM_SCHEMA, LLAMA_SERVER_PARAM_SCHEMA, LLAMA_SERVER_UNIFIED_PARAM_SCHEMA,
} from '../local-llm/runners/llamaServer.mjs';

const KIB = 1024;
const MIN_BYTES = 1024 ** 2;
const UNITS = Object.freeze({ kib: KIB, mib: KIB ** 2, gib: KIB ** 3, tib: KIB ** 4 });
const SIZE_KEYS = Object.freeze(['gpuMemory', 'ram']);
// A cap on a count, and the run parameter it bounds (the llama-server names, as the schemas of the runners pworker offers have them).
const COUNT_PARAMS = Object.freeze({ threads: 'threads', maxContext: 'ctxSize', maxParallel: 'parallel' });
const COUNT_KEYS = Object.freeze(Object.keys(COUNT_PARAMS));
const SUPPORTED_KEYS = Object.freeze([...SIZE_KEYS, ...COUNT_KEYS]);
const MODES = Object.freeze(['refuse', 'fit']);
// What `fit` may lower, in this order (the engine knows how: fit.mjs).
const FIT_ORDER = Object.freeze(['ctxSize', 'nGpuLayers', 'parallel']);
// The smallest context any runner pworker offers (llama.cpp CUDA, llama.cpp-cpu, Metal) accepts: a lower cap leaves no valid value.
const MIN_CONTEXT = Math.min(...[LLAMA_SERVER_PARAM_SCHEMA, LLAMA_SERVER_UNIFIED_PARAM_SCHEMA, LLAMA_SERVER_CPU_PARAM_SCHEMA, LLAMA_SERVER_METAL_PARAM_SCHEMA]
    .map((schema) => schema.properties.ctxSize.minimum));

const invalid = (key, why) => new LocalLlmError('invalid_caps', `localLlm.caps.${key} ${why}.`, { key });

/** One cap value: { bytes } for a size, { fraction } for a share of the physical figure. */
function parseSize(key, value) {
    if (typeof value === 'number') {
        if (!Number.isFinite(value) || value <= 0) throw invalid(key, 'must be a positive size: bytes, a string such as "12GiB", or a fraction above 0 and at most 1');
        if (value <= 1) return Object.freeze({ fraction: value });
        if (!Number.isSafeInteger(value)) throw invalid(key, 'must be a whole number of bytes (a safe integer), a size string such as "12GiB", or a fraction');
        if (value < MIN_BYTES) throw invalid(key, `is ${value} bytes, under the 1 MiB minimum; a bare number is bytes, so write the unit, for example "12GiB"`);
        return Object.freeze({ bytes: value });
    }
    if (typeof value === 'string') {
        const match = /^\s*(\d+(?:\.\d+)?)\s*(KiB|MiB|GiB|TiB)\s*$/i.exec(value);
        if (!match) throw invalid(key, `must be bytes, a fraction, or a size with a binary unit such as "12GiB" or "512MiB" (got "${value}")`);
        const bytes = Math.floor(Number(match[1]) * UNITS[match[2].toLowerCase()]);
        if (!Number.isSafeInteger(bytes) || bytes < MIN_BYTES) throw invalid(key, `must be at least 1 MiB (got "${value}")`);
        return Object.freeze({ bytes });
    }
    throw invalid(key, 'must be bytes, a size string such as "12GiB", or a fraction above 0 and at most 1');
}

/** A count cap: a safe integer of at least 1 (`maxContext` at least the smallest context a runner accepts). */
function parseCount(key, value) {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1) throw invalid(key, `must be a whole number of at least 1 (got ${JSON.stringify(value)})`);
    if (key === 'maxContext' && value < MIN_CONTEXT) throw invalid(key, `is ${value}, under ${MIN_CONTEXT}, the smallest context a runner accepts`);
    return value;
}

/** The parsed caps of `localLlm.caps`, or null when there are none. Throws `invalid_caps`, naming the key. */
export function parseCaps(section) {
    if (section === undefined || section === null) return null;
    if (typeof section !== 'object' || Array.isArray(section)) throw invalid('(caps)', 'must be an object');
    const parsed = {};
    for (const [key, value] of Object.entries(section)) {
        if (key === 'mode') {
            if (!MODES.includes(value)) throw invalid(key, `must be "refuse" or "fit" (got ${JSON.stringify(value)})`);
            parsed.mode = value;
            continue;
        }
        if (!SUPPORTED_KEYS.includes(key)) throw invalid(key, `is not a known cap (known: ${SUPPORTED_KEYS.join(', ')})`);
        parsed[key] = SIZE_KEYS.includes(key) ? parseSize(key, value) : parseCount(key, value);
    }
    if (!SUPPORTED_KEYS.some((key) => parsed[key])) return null;
    return Object.freeze({ ...parsed, mode: parsed.mode ?? 'refuse' });
}

/** A short stable digest of parsed caps: the first 12 hex digits of the SHA-256 of their canonical JSON. */
export function capsDigest(caps) {
    if (!caps) return null;
    const canonical = JSON.stringify(SUPPORTED_KEYS.filter((key) => caps[key]).map((key) => (COUNT_KEYS.includes(key) ? [key, caps[key], null] : [key, caps[key].bytes ?? null, caps[key].fraction ?? null])).concat([['mode', caps.mode ?? 'refuse']]));
    return createHash('sha256').update(canonical).digest('hex').slice(0, 12);
}

const finite = (value) => Number.isFinite(value) && value > 0;
const gib = (bytes) => `${(bytes / KIB ** 3).toFixed(2)} GiB`;

/**
 * The engine limits that apply `caps` (parsed, not null). Every hook it does not bind stays the identity. A fraction is resolved
 * against the first physical figure the engine shows it and then kept for the life of the limits. The state is in this closure:
 * one per engine instance.
 *
 * With a `ram` cap and a way to read the runners' memory, the guard hooks are bound: `runnerMemory` (a function returning bytes, 0 or
 * null) or, to build one, `launchDir` with `platformName`, `procDir`, `exec`, `fsApi` and `procs` (runnerMemory.mjs). With neither,
 * or with no `ram`, `guardSample` and `guardBreach` stay the identity.
 */
export function defineCapLimits(caps, { runnerMemory = null, launchDir = null, ...readerOptions } = {}) {
    if (!caps) throw new LocalLlmError('invalid_caps', 'defineCapLimits needs parsed caps; with none, use the identity limits.', { key: '(caps)' });
    const frozen = { ram: null, gpuMemory: null };

    // The cap in bytes for `key` against `physical` (the total it is a share of), or null while a fraction has nothing to resolve against.
    function resolve(key, physical) {
        const spec = caps[key];
        if (!spec) return null;
        if (frozen[key] !== null) return frozen[key];
        if (spec.bytes !== undefined) frozen[key] = spec.bytes;
        else if (finite(physical)) frozen[key] = Math.max(1, Math.floor(spec.fraction * physical));
        return frozen[key];
    }

    const ramOf = (memory) => resolve('ram', memory?.totalBytes);
    // The NVIDIA GPU of the dedicated and unified profiles: it has a memory model and no backend (a Metal or Vulkan device names one).
    const nvidiaGpu = (gpu) => gpu?.available === true && !gpu.backend && (gpu.memoryModel === 'dedicated' || gpu.memoryModel === 'unified');
    const unifiedGpu = (snapshot) => nvidiaGpu(snapshot?.gpu) && snapshot.gpu.memoryModel === 'unified';
    // A unified GPU has no memory of its own: its cap is a share of system memory.
    const gpuOf = (snapshot) => (nvidiaGpu(snapshot?.gpu) ? resolve('gpuMemory', unifiedGpu(snapshot) ? snapshot?.memory?.totalBytes : snapshot?.gpu?.totalBytes) : null);
    const readUsed = runnerMemory ?? (launchDir ? createRunnerMemoryReader({ launchDir, ...readerOptions }) : null);
    // The memory the runners this host started hold now, which counts against the `ram` cap (Q3: the cap is on what the pworker
    // models use, so the memory a model may plan for is the cap less this). A reading that fails counts as 0: the figure only
    // lowers the capped available figure, so a failed read admits a little more, and the guard backstop (below) still stops a
    // runner that is over the cap; the other choice would refuse every preview whenever one `ps` fails. A Run stops every
    // runner before it is admitted, so the figure is nonzero only for a preview and an overview while a model runs. The same
    // snapshot is asked about many times (one admission per fit step, one per model and runner in the overview, a `ps` per read on
    // macOS), so the reading is kept per memory object of a snapshot.
    const usedFor = new WeakMap();
    function ourUse(memory) {
        if (!readUsed || memory === null || typeof memory !== 'object') return 0;
        if (usedFor.has(memory)) return usedFor.get(memory);
        let bytes = 0;
        try {
            const value = readUsed();
            if (Number.isSafeInteger(value) && value >= 0) bytes = value;
        } catch { /* counts as 0 */ }
        usedFor.set(memory, bytes);
        return bytes;
    }
    // What a model may plan for of the available memory under the `ram` cap `limit`.
    const roomUnder = (memory, limit) => Math.max(0, limit - ourUse(memory));

    // The capped view, with the figures the reserve and the floor come from beside it: the policies keep the host reserve and the floor of the machine.
    // `availableLimit` is the cap on the available figure when it is not the cap on the total (the `ram` cap less the runners' own use).
    const limitMemory = (memory, limit, availableLimit = limit) => ({
        ...memory,
        reserveTotalBytes: memory.reserveTotalBytes ?? memory.totalBytes,
        reserveAvailableBytes: memory.reserveAvailableBytes ?? memory.availableBytes,
        ...(Number.isFinite(memory.totalBytes) ? { totalBytes: Math.min(memory.totalBytes, limit) } : {}),
        ...(Number.isFinite(memory.availableBytes) ? { availableBytes: Math.min(memory.availableBytes, availableLimit) } : {}),
    });

    // A refusal of a value the caller set above a cap: invalid_params reaches HTTP as a 400 and the CLI as the message.
    const refusal = (key, param, cap, value) => new LocalLlmError('invalid_params',
        `Invalid parameter ${param}: ${value} is above localLlm.caps.${key} (${cap}); lower ${param} to ${cap} or less, or raise the cap in the configuration.`,
        { field: param, cap: `localLlm.caps.${key}`, limit: cap, value });

    // The guard of the `ram` cap. The sample says the runner is watched and stands alone, so a runner with no memory floor (dedicated
    // llama.cpp) is guarded too (DS007). A reading that failed is a sample with `usedBytes: null`, not null: the guard decides once, on its first
    // sample, whether it watches, and one failed read must not leave the cap unenforced for the life of the runner.
    const guardHooks = !caps.ram || !readUsed ? {} : {
        guardSample() {
            let usedBytes = null;
            try { usedBytes = readUsed(); } catch { /* no reading */ }
            return { watched: true, standalone: true, usedBytes: Number.isSafeInteger(usedBytes) && usedBytes >= 0 ? usedBytes : null };
        },
        // Only the cap is checked here: the answer is never `{ available }`, so the floor check on the machine's available memory is the engine's own.
        guardBreach(sample, { check, memory } = {}) {
            if (check !== 'headroom') return null;
            const used = sample?.usedBytes;
            const limit = ramOf(memory);
            if (!Number.isFinite(used) || limit === null || used <= limit) return null;
            return { stop: `stopped: the local runners use ${gib(used)} of memory, above the pworker ram cap (localLlm.caps.ram) of ${gib(limit)}` };
        },
    };

    return defineLimits({
        ...guardHooks,
        // Fit mode: the engine may lower these, in this order, for a request whose admission is `incompatible` (DS007).
        ...(caps.mode === 'fit' ? { fitParams: () => ({ order: FIT_ORDER }) } : {}),
        // The default thread count is lowered to the cap; a count the caller set is refused in capParams.
        capThreads: (threads) => (caps.threads ? Math.min(threads, caps.threads) : threads),
        // A default or saved value above a count cap is lowered to it; one the caller set is refused. null (the default threads) is left to capThreads.
        capParams({ params, explicit, model }) {
            let result = params;
            for (const key of COUNT_KEYS) {
                const cap = caps[key];
                const param = COUNT_PARAMS[key];
                if (!cap || !Number.isFinite(params[param]) || params[param] <= cap) continue;
                if (explicit.includes(param)) throw refusal(key, param, cap, params[param]);
                result = { ...result, [param]: cap };
            }
            // The training context of a model is per slot, so a context sized for more slots than the cap allows is lowered with them.
            if (result.parallel !== params.parallel && Number.isInteger(model?.contextLength)) {
                const room = model.contextLength * result.parallel;
                // A context the caller set stays theirs: if the lowered slots cannot hold it, the cap is what refuses it.
                if (explicit.includes('ctxSize') && result.ctxSize > room) {
                    throw new LocalLlmError('invalid_params',
                        `Invalid parameter ctxSize: ${result.ctxSize} does not fit localLlm.caps.maxParallel (${caps.maxParallel}): with ${result.parallel} slot${result.parallel === 1 ? '' : 's'} `
                        + `the model's training context allows ${room} tokens (${model.contextLength} per slot); lower ctxSize to ${room} or less, or raise the cap in the configuration.`,
                        { field: 'ctxSize', cap: 'localLlm.caps.maxParallel', limit: caps.maxParallel, value: result.ctxSize });
                } else if (!explicit.includes('ctxSize')) {
                    result = { ...result, ctxSize: Math.min(result.ctxSize, room) };
                }
            }
            return result;
        },
        // cpu and metal: the pool is the RAM cap. The warning is what the CPU policies add to theirs.
        memoryPool(memory, snapshot, { amount }) {
            const limit = ramOf(memory);
            if (limit === null) return null;
            return {
                totalBytes: Math.min(memory.totalBytes, limit),
                availableBytes: Math.min(memory.availableBytes, roomUnder(memory, limit)),
                capped: true,
                reserveTotalBytes: memory.totalBytes,
                reserveAvailableBytes: memory.availableBytes,
                warning: `Memory is capped by the pworker ram cap (localLlm.caps.ram) of ${amount(limit)}; this machine reports ${amount(memory.totalBytes)}.`,
            };
        },
        // dedicated and unified: the memory view the policies size against. On a unified GPU the GPU cap folds in, because those
        // policies size only against memory.
        effective({ snapshot }) {
            let memory = snapshot?.memory || {};
            const ram = ramOf(memory);
            if (ram !== null) memory = limitMemory(memory, ram, roomUnder(memory, ram));
            if (unifiedGpu(snapshot)) {
                const gpu = gpuOf(snapshot);
                if (gpu !== null) memory = limitMemory(memory, gpu);
            }
            return { memory };
        },
        // dedicated: the share of the GPU the policies size against. Every other field of the GPU is kept.
        admit(context) {
            if (context?.stage !== 'gpu' || context.selected !== 'dedicated') return null;
            const gpu = context.gpu;
            const limit = gpuOf({ gpu, memory: context.snapshot?.memory });
            if (limit === null || !Number.isFinite(gpu?.totalBytes) || !Number.isFinite(gpu?.freeBytes)) return null;
            return { share: { gpu: { ...gpu, totalBytes: Math.min(gpu.totalBytes, limit), freeBytes: Math.min(gpu.freeBytes, limit) } } };
        },
        // What the overview publishes as `limits.budget`.
        overview({ snapshot }) {
            const resolved = {};
            const ram = ramOf(snapshot?.memory);
            if (ram !== null) resolved.ram = ram;
            const gpu = gpuOf(snapshot);
            if (gpu !== null) resolved.gpuMemory = gpu;
            for (const key of COUNT_KEYS) if (caps[key]) resolved[key] = caps[key];
            return { caps: resolved, mode: caps.mode };
        },
    });
}
