import assert from 'node:assert/strict';
import test from 'node:test';
import { cpuFallbackExplanation, computeDiagnostic } from '../../lib/local-llm/controller/computeDiagnostics.mjs';
import { readGpu } from '../../lib/local-llm/controller/hardware.mjs';
import { HOST, PLOINKY_GPU_ACCESS, PLOINKY_MESSAGES } from './fixtures/host.mjs';

test('Ploinky explanations are bounded, sanitized and attributed without guessing Apple identity', async () => {
    const gpu = await readGpu({ env: { PLOINKY_GPU_STATUS: 'unavailable', PLOINKY_GPU_REASON: 'wrong provider\n\u202efor machine' },
        nvidiaSmi: HOST.nvidiaSmi, gpuAccess: PLOINKY_GPU_ACCESS, messages: PLOINKY_MESSAGES,
        execFileImpl(command, args, options, callback) { callback(Object.assign(new Error('missing'), { code: 'ENOENT' }), '', ''); } });
    assert.equal(gpu.diagnostic.code, 'gpu_access_unavailable'); assert.equal(gpu.diagnostic.driver, null);
    assert.match(gpu.diagnostic.message, /^Ploinky reports:/); assert.doesNotMatch(gpu.diagnostic.message, /[\n\u202e]/);
});

test('an unavailable Ploinky status without explanatory text retains attribution and an unknown cause', async () => {
    for (const reason of [undefined, '', '\n']) {
        const gpu = await readGpu({ env: { PLOINKY_GPU_STATUS: 'unavailable', PLOINKY_GPU_REASON: reason },
            nvidiaSmi: HOST.nvidiaSmi, gpuAccess: PLOINKY_GPU_ACCESS, messages: PLOINKY_MESSAGES,
            execFileImpl(command, args, options, callback) { callback(Object.assign(new Error('missing'), { code: 'ENOENT' }), '', ''); } });
        assert.equal(gpu.diagnostic.code, 'gpu_access_unavailable'); assert.equal(gpu.diagnostic.source, 'ploinky-environment');
        assert.equal(gpu.diagnostic.backend, null); assert.equal(gpu.diagnostic.driver, null);
        assert.match(gpu.diagnostic.message, /^Ploinky reports: GPU access is unavailable; the cause is unknown/);
        assert.doesNotMatch(gpu.diagnostic.message, /Apple|denied|wrong provider/);
    }
});
