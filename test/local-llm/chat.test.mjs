// The pure chat helpers (lib/local-llm/chat.mjs): request building, the completion budget, the deadline and the
// runner's speed figures. The responder that wraps them lives with its host.
import assert from 'node:assert/strict';
import test from 'node:test';

import {
    CPU_MAX_COMPLETION_TOKENS, MAX_COMPLETION_TOKENS, MAX_COMPLETION_TOKENS_LIMIT, RUNNER_TIMEOUT_MS,
    buildRunnerRequest, chatLimits, completionStats, cpuCompletionBudget, createStreamStatsReader, requestLimitProblem,
} from '../../lib/local-llm/chat.mjs';

const TARGET = { baseUrl: 'http://127.0.0.1:18080', apiKey: 'k', model: 'gpt-oss-20b' };

test('chat: a stream ends with a usage chunk unless the caller chose otherwise', () => {
    assert.deepEqual(buildRunnerRequest({ messages: [], stream: true }, { model: 'gpt-oss:20b' }),
        { messages: [], stream: true, max_tokens: 8192, stream_options: { include_usage: true }, model: 'gpt-oss:20b' });
    assert.deepEqual(buildRunnerRequest({ messages: [], stream: true, stream_options: { include_usage: false } }, { model: 'm' }).stream_options,
        { include_usage: false });
    assert.equal('stream_options' in buildRunnerRequest({ messages: [] }, { model: 'm' }), false);
});

test('chat: only the forwarded fields reach the runner, and the model is the target model', () => {
    const body = buildRunnerRequest({ messages: [], top_k: 20, min_p: 0.05, repeat_penalty: 1.1, parallel_tool_calls: false, user: 'x', model: 'other' }, TARGET,
        { maxCompletionTokens: 1024 });
    assert.deepEqual(body, { messages: [], top_k: 20, min_p: 0.05, repeat_penalty: 1.1, parallel_tool_calls: false, max_tokens: 1024, model: 'gpt-oss-20b' });
    assert.deepEqual(buildRunnerRequest(null, TARGET), { max_tokens: 8192, model: 'gpt-oss-20b' });
});

test('chat: completions are capped, and a request without a token limit gets the default cap', () => {
    assert.equal(buildRunnerRequest({ messages: [], max_tokens: 100_000 }, TARGET).max_tokens, 8192);
    assert.equal(buildRunnerRequest({ messages: [], max_completion_tokens: 9000 }, TARGET).max_completion_tokens, 8192);
    assert.equal(buildRunnerRequest({ messages: [], max_tokens: 256 }, TARGET).max_tokens, 256);
    assert.equal(buildRunnerRequest({ messages: [], max_tokens: 100_000 }, TARGET, { maxCompletionTokens: 32000 }).max_tokens, 32000);
    assert.equal(buildRunnerRequest({ messages: [] }, TARGET).max_tokens, 8192);
    // A caller that sets either limit keeps only its own field.
    const onlyNew = buildRunnerRequest({ messages: [], max_completion_tokens: 64 }, TARGET);
    assert.equal(onlyNew.max_completion_tokens, 64);
    assert.equal('max_tokens' in onlyNew, false);
});

test('chat: one choice per request and positive integer token limits', () => {
    assert.equal(requestLimitProblem({ messages: [] }), null);
    assert.equal(requestLimitProblem({ messages: [], n: 1 }), null);
    assert.match(requestLimitProblem({ messages: [], n: 4 }), /n must be 1/);
    for (const request of [{ messages: [], max_tokens: -5 }, { messages: [], max_tokens: 1.5 }, { messages: [], max_completion_tokens: 0 }]) {
        assert.match(requestLimitProblem(request), /must be a positive integer/, JSON.stringify(request));
    }
    assert.equal(requestLimitProblem({ messages: [], max_tokens: null }), null);
});

test('chat: the completion budget and runner deadline come from the environment, within their bounds', () => {
    assert.deepEqual(chatLimits({}), { maxCompletionTokens: 8192, runnerTimeoutMs: 570_000 });
    assert.deepEqual(chatLimits({ LOCAL_LLM_MAX_COMPLETION_TOKENS: '32000', LOCAL_LLM_RUNNER_TIMEOUT_MS: '110000' }),
        { maxCompletionTokens: 32000, runnerTimeoutMs: 110_000 });
    // Out of bounds or not a whole number: the default. The deadline never passes the 600 s command limit.
    for (const [tokens, timeout] of [['0', '600000'], ['40000', '5000'], ['1e3', 'soon'], ['-1', '570001']]) {
        assert.deepEqual(chatLimits({ LOCAL_LLM_MAX_COMPLETION_TOKENS: tokens, LOCAL_LLM_RUNNER_TIMEOUT_MS: timeout }),
            { maxCompletionTokens: 8192, runnerTimeoutMs: RUNNER_TIMEOUT_MS });
    }
    assert.equal(MAX_COMPLETION_TOKENS, 8192);
    assert.equal(MAX_COMPLETION_TOKENS_LIMIT, 32768);
    assert.equal(RUNNER_TIMEOUT_MS, 570_000);
});

test('chat: the cpu profile defaults to 2048 completion tokens unless the operator set a budget', () => {
    assert.equal(CPU_MAX_COMPLETION_TOKENS, 2048);
    assert.equal(cpuCompletionBudget({}), 2048);
    assert.equal(cpuCompletionBudget({ LOCAL_LLM_MAX_COMPLETION_TOKENS: '4096' }), 4096);
    assert.equal(cpuCompletionBudget({ LOCAL_LLM_MAX_COMPLETION_TOKENS: '1' }), 1);
    assert.equal(cpuCompletionBudget({ LOCAL_LLM_MAX_COMPLETION_TOKENS: '32768' }), 32768);
    for (const bad of ['0', '32769', 'abc', '', '-5', '1e3']) assert.equal(cpuCompletionBudget({ LOCAL_LLM_MAX_COMPLETION_TOKENS: bad }), 2048, bad);
});

test('chat: the budget helpers read process.env when given no environment', (t) => {
    const before = process.env.LOCAL_LLM_MAX_COMPLETION_TOKENS;
    t.after(() => { if (before === undefined) delete process.env.LOCAL_LLM_MAX_COMPLETION_TOKENS; else process.env.LOCAL_LLM_MAX_COMPLETION_TOKENS = before; });
    process.env.LOCAL_LLM_MAX_COMPLETION_TOKENS = '3000';
    assert.equal(cpuCompletionBudget(), 3000);
    assert.equal(chatLimits().maxCompletionTokens, 3000);
});

test('chat: runner timings come from a response, and usage is the fallback', () => {
    const timings = { prompt_n: 20, prompt_per_second: 812.5, predicted_n: 7, predicted_per_second: 61.25 };
    assert.deepEqual(completionStats({ timings, usage: { prompt_tokens: 20, completion_tokens: 7 } }), {
        promptTokens: 20, completionTokens: 7, promptTokensPerSecond: 812.5, generationTokensPerSecond: 61.25, source: 'runner timings',
    });
    assert.deepEqual(completionStats({ usage: { prompt_tokens: 3, completion_tokens: 4 } }), {
        promptTokens: 3, completionTokens: 4, promptTokensPerSecond: null, generationTokensPerSecond: null, source: 'usage',
    });
    assert.equal(completionStats({ choices: [] }), null);
    assert.equal(completionStats(null), null);
    assert.equal(completionStats({ usage: { prompt_tokens: -1, completion_tokens: 'x' } }).promptTokens, null);
});

test('chat: the stream reader keeps the stats of the last chunk that carries them, across split chunks', () => {
    const reader = createStreamStatsReader();
    const timings = { prompt_n: 5, predicted_n: 9, prompt_per_second: 10, predicted_per_second: 20 };
    const line = `data: ${JSON.stringify({ choices: [], timings })}\n\n`;
    reader.push('data: {"choices":[{"delta":{"content":"hi"}}]}\n\n');
    reader.push(Buffer.from(line.slice(0, 20)));
    reader.push(Buffer.from(line.slice(20)));
    reader.push('data: [DONE]');
    assert.deepEqual(reader.finish(), { promptTokens: 5, completionTokens: 9, promptTokensPerSecond: 10, generationTokensPerSecond: 20, source: 'runner timings' });
    assert.equal(createStreamStatsReader().finish(), null);
    const unfinished = createStreamStatsReader();
    unfinished.push(`data: ${JSON.stringify({ usage: { prompt_tokens: 1, completion_tokens: 2 } })}`);
    assert.equal(unfinished.finish().source, 'usage');
    const broken = createStreamStatsReader();
    broken.push('data: {"usage": nope\n');
    assert.equal(broken.finish(), null);
});
