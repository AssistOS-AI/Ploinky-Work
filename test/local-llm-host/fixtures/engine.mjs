// Shared by the host and server tests: a runner adapter that starts the stand-in process (fake-runner.mjs) on the start's Unix
// socket, the hardware snapshot the engine sees, and small helpers.
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { admissionResult } from '../../../lib/local-llm/controller/admission.mjs';
import { deepFreeze, normalizeWith } from '../../../lib/local-llm/runners/params.mjs';

export const FAKE_RUNNER = fileURLToPath(new URL('./fake-runner.mjs', import.meta.url));
export const MIB = 1024 * 1024;
export const GIB = 1024 * MIB;
export const SNAPSHOT = Object.freeze({
    gpu: { available: true, name: 'Test GPU', totalBytes: 6144 * MIB, usedBytes: 144 * MIB, freeBytes: 6000 * MIB, processes: [] },
    memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB },
    disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
    cpus: 8,
});
export const MODEL = 'qwen2.5-0.5b-instruct-q4_k_m';

export const freePort = () => new Promise((resolve) => {
    const server = net.createServer();
    server.listen(0, '127.0.0.1', () => { const { port } = server.address(); server.close(() => resolve(port)); });
});

export function scratch(t) {
    const dir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'lh-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

/** A runner adapter that starts the stand-in process on the start's Unix socket. */
export function fakeAdapter() {
    const paramSchema = deepFreeze({
        type: 'object', additionalProperties: false,
        properties: { contextTokens: { type: 'integer', minimum: 512, maximum: 65536, default: 4096, title: 'Context', description: 'Context tokens.' } },
    });
    return Object.freeze({
        id: 'fake-gguf', displayName: 'Fake GGUF runner', weightFormat: 'gguf', pinnedVersion: 'v1', supported: true, executable: '/opt/fake/server',
        port: 18999, listensOnSocket: true, apiKey: true, paramSchema, basicParams: Object.freeze(['contextTokens']),
        normalizeParams: (params, { model } = {}) => normalizeWith(paramSchema, 'fake-gguf', params, model),
        describeContext: ({ contextTokens }) => ({ totalContext: contextTokens, perRequestContext: contextTokens, parallel: 1, kvUnified: false }),
        detect: () => ({ installed: true, version: 'v1', reason: null }),
        async start(ctx) {
            const proc = ctx.launch({ command: process.execPath, args: [FAKE_RUNNER, '--host', ctx.listen.socketPath, '--api-key', ctx.apiKey], env: {} });
            await ctx.waitForHttp('http://localhost/health', { process: proc });
            await ctx.waitForHttp('http://localhost/v1/models', { headers: { authorization: `Bearer ${ctx.apiKey}` }, process: proc });
            return {};
        },
        chatModel: (deployment) => `fake/${deployment.modelId}`,
        admit: () => admissionResult('ok', null, { gpuBytes: 1 * GIB, basis: 'fake' }),
        parseReport: () => ({ device: 'CUDA0 (fake)', totalMiB: 1 }),
    });
}

export async function until(predicate, timeoutMs = 5000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 10));
    }
    throw new Error('condition not reached');
}

/** Runs the fake runner on the seeded model and waits for the ready phase. */
export async function runModel(controller, requestId = 'request-host-01') {
    await controller.run({ modelId: MODEL, runnerId: 'fake-gguf', requestId });
    await until(() => ['ready', 'error'].includes(controller.state.deployment?.phase));
    if (controller.state.deployment.phase !== 'ready') throw new Error(controller.state.deployment.error);
}

/** The controller options that run the stand-in runner on the seeded model without a network, a GPU or an installer. */
export function controllerSeams() {
    return {
        runners: { 'fake-gguf': fakeAdapter() },
        snapshot: async () => structuredClone(SNAPSHOT),
        download: async ({ artifact }) => ({ status: 'complete', path: `/models/${artifact.file}`, bytesTransferred: 0 }),
        inspect: async () => ({ state: 'complete', bytes: 1 }),
        remove: async () => 1,
        detectRunner: () => ({ installed: true, version: 'v1', reason: null }),
        pollMs: 5,
        stopGraceMs: 500,
        profile: 'dedicated',
    };
}
