// The engine for the agent is unchanged by the metal profile: the runner table, the ports and the overview of every other profile
// are what they were before `metal` existed. The expected runner ids and ports are the ones the table had at the move (S1,
// 4668a3f); the overview comparison builds the same catalog with and without `metal` keys and demands deep equality.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog, validateModel } from '../../lib/local-llm/controller/catalog.mjs';
import { CATALOG_V4, METADATA_PROFILES } from '../../lib/local-llm/controller/venusMetadata.mjs';
import { PROFILES } from '../../lib/local-llm/controller/profiles.mjs';
import { catalogProfileOf, vulkanContractOf } from '../../lib/local-llm/controller/vulkan.mjs';
import { RUNNER_IDS, defaultPorts, runnerSummary, schemaOf } from '../../lib/local-llm/runners/index.mjs';
import { SEED_CATALOG_FILE } from './fixtures/seedCatalog.mjs';
import { HOST_VULKAN, RUNNERS, createController } from './fixtures/host.mjs';
import { VULKAN_KEYS } from './fixtures/sharedFixtures.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const contract = vulkanContractOf(VULKAN_KEYS, HOST_VULKAN);

const S1_RUNNER_IDS = ['llama.cpp', 'ik_llama.cpp', 'ollama', 'vllm', 'tabbyapi', 'lmstudio', 'llama.cpp-cpu', 'llama.cpp-vulkan'];
const S1_PORTS = { 'llama.cpp': 18080, 'ik_llama.cpp': 18081, ollama: 18434, vllm: 18082, tabbyapi: 18083, lmstudio: 18084, 'llama.cpp-cpu': 18085, 'llama.cpp-vulkan': 18086 };

test('agent-unchanged-metal: the runner table has the eight ids of the move, in order, and the ports are the same', () => {
    assert.deepEqual(Object.keys(RUNNERS), S1_RUNNER_IDS);
    assert.deepEqual([...RUNNER_IDS], S1_RUNNER_IDS);
    assert.deepEqual({ ...defaultPorts(RUNNERS) }, S1_PORTS);
    // Only llama.cpp has a metal policy; no other runner gained a schema or a hook for it.
    for (const [id, runner] of Object.entries(RUNNERS)) {
        assert.equal(typeof runner.admitMetal === 'function', id === 'llama.cpp', id);
        assert.equal(schemaOf(runner, 'metal') !== null, id === 'llama.cpp', id);
    }
});

test('agent-unchanged-metal: PROFILES is unchanged and metal is a metadata profile that the catalog maps to cpu', () => {
    assert.deepEqual([...PROFILES], ['dedicated', 'unified', 'cpu']);
    assert.deepEqual([...METADATA_PROFILES], ['dedicated', 'unified', 'cpu', 'venus', 'metal']);
    assert.equal(catalogProfileOf('metal'), 'cpu');
    for (const [profile, mapped] of [['dedicated', 'dedicated'], ['unified', 'unified'], ['cpu', 'cpu'], ['vulkan', 'dedicated'], ['venus', 'cpu'], ['apu', 'cpu']]) {
        assert.equal(catalogProfileOf(profile), mapped, profile);
    }
});

test('agent-unchanged-metal: a runner summary on every other profile is the same, and the other runners say they are not on the Apple GPU', () => {
    for (const runner of Object.values(RUNNERS)) {
        for (const profile of ['dedicated', 'unified', 'cpu', 'vulkan', 'venus', 'apu']) {
            assert.doesNotMatch(String(runnerSummary(runner, profile).profileUnsupportedReason ?? ''), /Apple GPU/, `${runner.id} on ${profile}`);
        }
        const summary = runnerSummary(runner, 'metal');
        assert.equal(summary.paramSchema !== null, runner.id === 'llama.cpp', runner.id);
        if (runner.id !== 'llama.cpp') assert.match(summary.profileUnsupportedReason, /not available on the Apple GPU/);
    }
});

// ------------------------------------------------------------------ overview with and without metal keys

const rawModels = JSON.parse(fs.readFileSync(SEED_CATALOG_FILE, 'utf8')).models;
const plain = loadSeedCatalog(SEED_CATALOG_FILE);
// The same catalog with a `metal` recommendation and a `metal` validation label on every model.
const withMetal = Object.freeze(rawModels.map((entry) => validateModel({
    ...entry,
    recommended: { ...entry.recommended, metal: { 'llama.cpp': { ctxSize: 2048 } } },
    validated: { ...entry.validated, metal: { 'llama.cpp': 'measured on an M1 (test label)' } },
}, { seed: true, schema: CATALOG_V4 })));

const small = plain.find((model) => model.id === 'qwen2.5-0.5b-instruct-q4_k_m');
const gpus = {
    dedicated: { available: true, name: 'NVIDIA RTX', memoryModel: 'dedicated', totalBytes: 12 * GIB, freeBytes: 11 * GIB, usedBytes: GIB, processes: [] },
    unified: { available: true, name: 'NVIDIA GB10', memoryModel: 'unified', totalBytes: null, usedBytes: null, freeBytes: null, processes: [] },
    cpu: { available: false, reason: 'No GPU is available to this agent.' },
    vulkan: { available: true, backend: 'vulkan', driver: 'radv', memoryModel: 'dedicated', name: 'AMD Radeon RX', totalBytes: 16 * GIB, freeBytes: 15 * GIB, icdFile: contract.vulkan_icd_radv, processes: [] },
    venus: { available: true, backend: 'vulkan', driver: 'venus', memoryModel: 'venus', name: 'Virtio-GPU Venus (Apple M1)', totalBytes: 16 * GIB, freeBytes: 12 * GIB, icdFile: contract.vulkan_icd_venus, processes: [] },
    apu: { available: true, backend: 'vulkan', driver: 'radv', memoryModel: 'shared', name: 'AMD Radeon Graphics', totalBytes: 16 * GIB, freeBytes: 16 * GIB, icdFile: contract.vulkan_icd_radv, processes: [] },
    metal: { available: true, backend: 'metal', name: 'Apple M1', memoryModel: 'unified', deviceCapBytes: 12 * GIB },
};

async function overviewOf(t, catalog, profile) {
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'local-llm-unchanged-metal-'));
    const memory = { totalBytes: 32 * GIB, availableBytes: 24 * GIB, swapFreeBytes: 0, freeBytes: 20 * GIB, cachedBytes: 4 * GIB, pressureLevel: 1 };
    const controller = createController({
        dataDir, env: { PATH: '/usr/bin' }, seedCatalog: catalog, imageContract: { architecture: 'arm64', gpu_compute_capabilities: '12.1', ...VULKAN_KEYS },
        hostArch: 'arm64', fileExists: () => true, sharedModelsRoot: null, profile,
        detectRunner: () => ({ installed: true, version: 'b11159', reason: null }),
        snapshot: async () => ({ at: '2026-10-16T00:00:00.000Z', gpu: structuredClone(gpus[profile]), memory: { ...memory }, disk: { freeBytes: 200 * GIB, totalBytes: 500 * GIB }, cpus: 8, cores: 8 }),
        inspect: async () => ({ state: 'absent', bytes: 0 }),
    });
    t.after(async () => { await controller.drain(); fs.rmSync(dataDir, { recursive: true, force: true }); });
    return controller.overview();
}

test('agent-unchanged-metal: the overview on dedicated, unified, cpu, vulkan, venus and apu is deep-equal with and without metal keys in the catalog', async (t) => {
    assert.deepEqual(withMetal.find((model) => model.id === small.id).recommended.metal, { 'llama.cpp': { ctxSize: 2048 } }, 'the metal keys are in the catalog');
    for (const profile of ['dedicated', 'unified', 'cpu', 'vulkan', 'venus', 'apu']) {
        const without = await overviewOf(t, plain, profile);
        const with_ = await overviewOf(t, withMetal, profile);
        assert.equal(without.profile, profile);
        assert.ok(without.models.length > 0, `${profile} offers models`);
        // The label of the model is shown per profile: a metal label is never shown on another one.
        assert.deepEqual(with_, without, profile);
        assert.equal(JSON.stringify(with_).includes('measured on an M1 (test label)'), false, profile);
    }
});

test('agent-unchanged-metal: on the metal profile the metal recommendation is applied and its label is shown', async (t) => {
    const view = await overviewOf(t, withMetal, 'metal');
    assert.equal(view.profile, 'metal');
    const model = view.models.find((entry) => entry.id === small.id);
    assert.deepEqual(model.validated, { 'llama.cpp': 'measured on an M1 (test label)' });
    assert.equal(model.runners['llama.cpp'].params.ctxSize, 2048);
    // The catalog of the cpu profile is what the metal profile offers.
    assert.deepEqual(view.models.map((entry) => entry.id), (await overviewOf(t, withMetal, 'cpu')).models.map((entry) => entry.id));
});
