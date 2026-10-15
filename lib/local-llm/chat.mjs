// The pure helpers of a chat-completions responder: the fields that are
// forwarded to a runner, the completion budget and runner deadline from the
// environment, and the runner's speed figures. Nothing here reads a socket, a
// process stream or a controller, so a host builds its own responder around
// them. The environment names and the defaults are part of the contract.

const FORWARDED_FIELDS = new Set([
    'messages', 'stream', 'stream_options', 'max_tokens', 'max_completion_tokens', 'temperature', 'top_p',
    'top_k', 'min_p', 'repeat_penalty', 'stop', 'seed', 'presence_penalty', 'frequency_penalty', 'n', 'tools',
    'tool_choice', 'parallel_tool_calls', 'response_format', 'reasoning_effort', 'chat_template_kwargs',
    'logprobs', 'top_logprobs',
]);

// Runner-reported speed for the Settings test prompt: llama.cpp adds
// `timings` to a response and to the last stream chunk; Ollama's /v1 gives
// token counts in `usage` only.
export function completionStats(body) {
    if (!body || typeof body !== 'object') return null;
    const timings = body.timings && typeof body.timings === 'object' ? body.timings : null;
    const usage = body.usage && typeof body.usage === 'object' ? body.usage : null;
    if (!timings && !usage) return null;
    const count = (value) => (Number.isFinite(value) && value >= 0 ? value : null);
    return {
        promptTokens: count(timings?.prompt_n) ?? count(usage?.prompt_tokens),
        completionTokens: count(timings?.predicted_n) ?? count(usage?.completion_tokens),
        promptTokensPerSecond: count(timings?.prompt_per_second),
        generationTokensPerSecond: count(timings?.predicted_per_second),
        source: timings ? 'runner timings' : 'usage',
    };
}

// Watches SSE `data:` lines as they pass through and keeps the last stats.
export function createStreamStatsReader() {
    const decoder = new TextDecoder();
    let pending = '';
    let stats = null;
    const scan = (line) => {
        if (!line.startsWith('data:') || !/"(timings|usage)"/.test(line)) return;
        try {
            stats = completionStats(JSON.parse(line.slice(5).trim())) || stats;
        } catch {}
    };
    return {
        push(chunk) {
            pending += typeof chunk === 'string' ? chunk : decoder.decode(chunk, { stream: true });
            let newline;
            while ((newline = pending.indexOf('\n')) >= 0) {
                scan(pending.slice(0, newline).trim());
                pending = pending.slice(newline + 1);
            }
        },
        finish() {
            scan(pending.trim());
            return stats;
        },
    };
}

// Limits for an authorized caller: one choice per request, at most the
// completion budget in generated tokens (larger values are clamped, and a
// request that sets no limit gets it as `max_tokens`), and a runner call that
// ends inside a 600 s command limit. The operator may set both through the
// environment, within these bounds; the caller's own deadline still applies,
// so a long budget needs that deadline raised too.
export const MAX_COMPLETION_TOKENS = 8192;
export const MAX_COMPLETION_TOKENS_LIMIT = 32768;
export const RUNNER_TIMEOUT_MS = 570_000;
const MIN_RUNNER_TIMEOUT_MS = 10_000;
const TOKEN_LIMIT_FIELDS = ['max_tokens', 'max_completion_tokens'];

function boundedInteger(value, fallback, min, max) {
    if (!/^\d{1,9}$/.test(String(value ?? ''))) return fallback;
    const number = Number(value);
    return number >= min && number <= max ? number : fallback;
}

// On the cpu profile a completion is much slower than on a GPU, and the caller's own deadline
// is not raised for it, so the default budget is smaller.
export const CPU_MAX_COMPLETION_TOKENS = 2048;

/** The completion budget on the cpu profile: 2,048 tokens unless the operator set one, whose value wins. */
export function cpuCompletionBudget(env = process.env) {
    return boundedInteger(env.LOCAL_LLM_MAX_COMPLETION_TOKENS, CPU_MAX_COMPLETION_TOKENS, 1, MAX_COMPLETION_TOKENS_LIMIT);
}

/** The completion budget and runner deadline from the environment, each within its bounds. */
export function chatLimits(env = process.env) {
    return {
        maxCompletionTokens: boundedInteger(env.LOCAL_LLM_MAX_COMPLETION_TOKENS, MAX_COMPLETION_TOKENS, 1, MAX_COMPLETION_TOKENS_LIMIT),
        runnerTimeoutMs: boundedInteger(env.LOCAL_LLM_RUNNER_TIMEOUT_MS, RUNNER_TIMEOUT_MS, MIN_RUNNER_TIMEOUT_MS, RUNNER_TIMEOUT_MS),
    };
}

export function requestLimitProblem(request) {
    if (request.n !== undefined && request.n !== 1) return 'n must be 1; local models return one choice per request';
    for (const field of TOKEN_LIMIT_FIELDS) {
        const value = request[field];
        if (value !== undefined && value !== null && (!Number.isInteger(value) || value < 1)) {
            return `${field} must be a positive integer`;
        }
    }
    return null;
}

export function buildRunnerRequest(request, target, { maxCompletionTokens = MAX_COMPLETION_TOKENS } = {}) {
    const body = {};
    for (const [key, value] of Object.entries(request || {})) {
        if (FORWARDED_FIELDS.has(key)) body[key] = value;
    }
    for (const field of TOKEN_LIMIT_FIELDS) {
        if (Number.isInteger(body[field]) && body[field] > maxCompletionTokens) body[field] = maxCompletionTokens;
    }
    if (!TOKEN_LIMIT_FIELDS.some((field) => Number.isInteger(body[field]))) body.max_tokens = maxCompletionTokens;
    // A stream ends with a chunk that carries `usage`, unless the caller says otherwise.
    if (body.stream === true) {
        const options = body.stream_options && typeof body.stream_options === 'object' && !Array.isArray(body.stream_options)
            ? body.stream_options : {};
        body.stream_options = { include_usage: true, ...options };
    }
    body.model = target.model;
    return body;
}
