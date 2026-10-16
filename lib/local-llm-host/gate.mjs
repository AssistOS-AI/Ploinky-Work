// The gate between the proxy and the local-llm engine of this home: what a request for a provider of kind `local-llm` asks before it
// is forwarded. It never creates the engine for a question it can answer without it, and it never downloads anything.
//
//   const gate = createLocalLlmGate({ host, settings: config.localLlm, busy });
//   await gate.ensure({ model, runner })   -> { ok: true, target } | { ok: false, reason }
//   await gate.stop()                      stops the engine's deployment (an explicit stop; a drain is the host's)
//   gate.status()                          -> { kind: 'local-llm', ready, phase, modelId, runnerId, startedBy, last_refusal, ... }
//   gate.identity(model, runner)           -> the cache identity of what would answer, or null when it cannot be known without the engine
//
// The deployment is the engine's single one. `pworker models run` (an admin) starts it and the gate only uses it. With
// `localLlm.onDemand.enabled` the gate itself may start one: only when the weights are complete and the runner is installed (a chat
// request never downloads), with one shared start for any number of requests, and it replaces or idle-stops only a deployment it
// started itself, and only while nothing is waiting or running (`busy`). A deployment is the gate's own when its id is the one the
// gate's run returned; any other is an admin's.
//
// `target` is the engine's chat target: { runnerId, modelId, baseUrl, socketPath, apiKey, model, requestOptions }; send it with
// httpFetch(url, { socketPath }) and the per-start apiKey as the bearer token.

import crypto from 'node:crypto';

import { loadSeedCatalog, mergeCatalog } from '../local-llm/controller/catalog.mjs';
import { SEED_CATALOG_FILE, hostLockVersion } from './host.mjs';

export const ON_DEMAND_DEFAULTS = Object.freeze({ idleStopMs: 900_000, startWaitMs: 120_000 });
const TRANSFER_PHASES = new Set(['downloading', 'copying', 'verifying']);
const STARTING_PHASES = new Set(['downloading', 'copying', 'verifying', 'starting', 'loading']);
const NOT_READY = 'No local model is ready; run one with `pworker models run <id>`.';

/** The runner a request gets when its tier entry names none: the CUDA build on an NVIDIA profile, the CPU build otherwise. */
export const defaultRunnerFor = (profile) => (profile === 'cpu' || !profile ? 'llama.cpp-cpu' : 'llama.cpp');

export function createLocalLlmGate({
    host = null, settings = {}, busy = () => false, log = () => {}, now = Date.now, pollMs = 500, arch = process.arch,
    timers = { setTimeout, clearTimeout }, seedCatalogFile = SEED_CATALOG_FILE,
} = {}) {
    const onDemand = settings?.onDemand && typeof settings.onDemand === 'object' ? settings.onDemand : {};
    const onDemandEnabled = onDemand.enabled === true;
    const idleStopMs = Number.isFinite(onDemand.idleStopMs) && onDemand.idleStopMs > 0 ? onDemand.idleStopMs : ON_DEMAND_DEFAULTS.idleStopMs;
    const startWaitMs = Number.isFinite(onDemand.startWaitMs) && onDemand.startWaitMs > 0 ? onDemand.startWaitMs : ON_DEMAND_DEFAULTS.startWaitMs;

    let own = null; // { deploymentId, modelId, runnerId } of the deployment the gate started
    let starting = null; // { key, replacing, promise } of the one start in flight; `replacing`: it stops the gate's own deployment first
    let lastRefusal = null;
    let lastUsed = 0;
    let idleTimer = null;
    const keysSeen = new Set();
    let seed = null;

    const redact = (text) => {
        let out = host ? host.redact(String(text)) : String(text);
        for (const key of keysSeen) out = out.split(key).join('***');
        return out;
    };
    const refuse = (reason) => {
        const message = redact(reason);
        lastRefusal = { reason: message, at: new Date(now()).toISOString() };
        return { ok: false, reason: message };
    };
    const sleep = (ms) => new Promise((resolve) => timers.setTimeout(resolve, ms));
    const deploymentOf = (controller) => controller?.state?.deployment ?? null;
    const isOwn = (controller) => own !== null && deploymentOf(controller)?.id === own.deploymentId;

    function chatTargetOf(controller) {
        try {
            const target = controller.chatTarget();
            if (target.apiKey) keysSeen.add(target.apiKey);
            return target;
        } catch (error) {
            if (error?.code === 'not_ready') return null;
            throw error;
        }
    }

    function clearIdle() {
        if (idleTimer) timers.clearTimeout(idleTimer);
        idleTimer = null;
    }

    // The idle stop: only a deployment the gate started, and only when nothing runs or waits for it.
    async function idleCheck() {
        idleTimer = null;
        const controller = host?.current() ?? null;
        if (!own || !controller || !isOwn(controller)) { own = null; return { stopped: false, reason: 'not the gate\'s deployment' }; }
        const idleFor = now() - lastUsed;
        if (busy() || idleFor < idleStopMs) {
            arm(busy() ? idleStopMs : idleStopMs - idleFor);
            return { stopped: false, reason: busy() ? 'busy' : 'not idle long enough' };
        }
        try {
            await controller.stop();
            log(`local-llm: stopped ${own.modelId} after ${Math.round(idleFor / 1000)} s without a request`);
        } catch (error) {
            log(`local-llm: the idle stop failed (${error?.code ?? error?.message})`);
        }
        own = null;
        return { stopped: true };
    }

    function arm(ms = idleStopMs) {
        clearIdle();
        if (!onDemandEnabled || !own) return;
        idleTimer = timers.setTimeout(() => { idleCheck().catch(() => {}); }, Math.max(1, ms));
        idleTimer.unref?.();
    }

    function touch() {
        lastUsed = now();
        if (own) arm();
    }

    function matches(target, model, runner) {
        return target.modelId === model && (!runner || target.runnerId === runner);
    }

    // Whether the engine could start `model` now without downloading: its weights are complete and its runner is installed.
    async function startable(controller, model, runner) {
        const overview = await controller.overview();
        const entry = overview.models.find((candidate) => candidate.id === model);
        if (!entry) return { why: `${model} is not in the local-llm catalog` };
        const runnerId = runner ?? defaultRunnerFor(overview.profile);
        const runnerInfo = overview.runners.find((candidate) => candidate.id === runnerId);
        if (!runnerInfo || runnerInfo.supported === false) return { why: `runner ${runnerId} is not available here${runnerInfo?.unsupportedReason ? `: ${runnerInfo.unsupportedReason}` : ''}` };
        if (!runnerInfo.installed) return { why: `runner ${runnerId} is not installed; run \`pworker models runner install ${runnerId}\`` };
        const perRunner = entry.runners?.[runnerId];
        if (!perRunner) return { why: `${model} does not run on ${runnerId}` };
        if (perRunner.download?.state !== 'complete') return { why: `weights of ${model} are not downloaded; run \`pworker models run ${model}\` once` };
        return { runnerId };
    }

    async function startOnDemand(controller, model, runner, replace) {
        const plan = await startable(controller, model, runner);
        if (plan.why) return refuse(plan.why);
        const requestId = crypto.randomUUID().replace(/-/g, '');
        const result = await controller.run({ requestId, modelId: model, runnerId: plan.runnerId, replace });
        own = { deploymentId: result.deployment?.id ?? null, modelId: model, runnerId: plan.runnerId };
        const deadline = now() + startWaitMs;
        for (;;) {
            const target = chatTargetOf(controller);
            if (target && matches(target, model, plan.runnerId)) { touch(); arm(); return { ok: true, target }; }
            const deployment = deploymentOf(controller);
            // The weights were there a moment ago. Every run passes through `downloading` (it verifies what is on disk); bytes transferred
            // mean something removed the weights since, and the run is fetching them. Stop it: a request never downloads.
            if (deployment && TRANSFER_PHASES.has(deployment.phase) && deployment.download?.transferred > 0) {
                await controller.cancelDownload().catch(() => {});
                return refuse(`weights of ${model} are not downloaded; run \`pworker models run ${model}\` once`);
            }
            if (!deployment || deployment.phase === 'error' || deployment.phase === 'idle' || deployment.phase === 'paused') {
                return refuse(`${model} did not start${deployment?.error ? `: ${typeof deployment.error === 'string' ? deployment.error : deployment.error.message ?? 'error'}` : ''}`);
            }
            if (now() >= deadline) {
                // Still loading: it stays the gate's own and keeps loading, so it is armed for the idle stop like a ready one (a request
                // that finds it ready touches it; one that never comes lets the idle stop end it).
                lastUsed = now();
                arm();
                return refuse(`start timeout: ${model} was not ready within ${Math.round(startWaitMs / 1000)} s`);
            }
            await sleep(pollMs);
        }
    }

    async function ensureInner({ model, runner } = {}) {
        if (!host) return refuse('the local-llm host is not available');
        if (!host.enabled) return refuse('the local-llm engine is disabled in the configuration (localLlm.enabled)');
        if (!host.supported.ok) return refuse(host.supported.reason);
        if (typeof model !== 'string' || !model) return refuse('a local-llm catalog model id is required');
        let controller = host.current();
        const key = `${model}\0${runner ?? ''}`;
        // Not while the gate has decided to replace this deployment: a request forwarded to it now would meet the stop of its runner.
        if (controller && !starting?.replacing) {
            const target = chatTargetOf(controller);
            if (target && matches(target, model, runner)) { touch(); return { ok: true, target }; }
        }
        // The gate's own start is in flight: a request for the same model joins it, one for another model is refused.
        if (starting) return starting.key === key ? starting.promise : refuse('local-llm is starting another model');
        if (controller) {
            const deployment = deploymentOf(controller);
            const active = deployment && (STARTING_PHASES.has(deployment.phase) || deployment.phase === 'ready');
            if (active) {
                if (!isOwn(controller)) return refuse(`admin-run deployment ${deployment.modelId} is ${deployment.phase === 'ready' ? 'active' : deployment.phase}; ${model} is not served until it is stopped (\`pworker models stop\`)`);
                // The gate's own, ready deployment, for another model: replaced only when nothing runs or waits.
                if (!onDemandEnabled) return refuse(NOT_READY);
                if (busy()) return refuse(`local-llm is serving ${deployment.modelId} and is busy; ${model} is not started`);
            }
        }
        if (!onDemandEnabled) return refuse(host.refusal ? host.refusal.message : NOT_READY);
        const replace = Boolean(controller && own && isOwn(controller));
        const promise = (async () => {
            try {
                controller = await host.controller();
                return await startOnDemand(controller, model, runner, replace);
            } catch (error) {
                return refuse(error?.message ?? String(error));
            } finally {
                starting = null;
            }
        })();
        starting = { key, replacing: replace, promise };
        return promise;
    }

    async function ensure(entry) {
        try {
            return await ensureInner(entry);
        } catch (error) {
            return refuse(error?.message ?? String(error));
        }
    }

    /** The deployment is no longer the gate's (an admin stopped it): forget it and its idle timer. */
    function forget() {
        clearIdle();
        own = null;
    }

    async function stop() {
        clearIdle();
        const controller = host?.current() ?? null;
        own = null;
        if (controller) await controller.stop();
    }

    function status() {
        const described = host ? host.describe() : null;
        const controller = host?.current() ?? null;
        const deployment = deploymentOf(controller);
        let target = null;
        try { target = controller ? controller.chatTarget() : null; } catch { target = null; }
        return {
            kind: 'local-llm',
            managed: true,
            enabled: described?.enabled ?? false,
            supported: described?.supported ?? { ok: false, reason: 'the local-llm host is not available' },
            ownership: described?.ownership ?? null,
            socketDirProblem: described?.socketDirProblem ?? null,
            ready: target !== null,
            phase: deployment?.phase ?? 'idle',
            modelId: deployment?.modelId ?? null,
            runnerId: deployment?.runnerId ?? null,
            startedBy: deployment && deployment.phase !== 'idle' ? (isOwn(controller) ? 'on-demand' : 'admin') : null,
            on_demand: onDemandEnabled,
            idle_stop_ms: onDemandEnabled ? idleStopMs : 0,
            last_refusal: lastRefusal,
        };
    }

    function sourceSha(model, controller) {
        try {
            seed ??= loadSeedCatalog(seedCatalogFile);
            const entry = mergeCatalog(seed, controller?.state?.registry ?? []).find((candidate) => candidate.id === model);
            const sha = entry?.sources?.gguf?.sha256;
            return typeof sha === 'string' && sha ? sha : null;
        } catch {
            return null;
        }
    }

    // What would answer, without asking the engine for anything: the model, the runner, the runner's pinned version and the
    // weights' sha256. The runner is the entry's, else the one serving the model now; with neither it is unknowable (no cache).
    function identity(model, runner = null) {
        if (!host || typeof model !== 'string') return null;
        const controller = host.current();
        let runnerId = runner;
        const deployment = deploymentOf(controller);
        if (!runnerId && deployment?.phase === 'ready' && deployment.modelId === model) runnerId = deployment.runnerId;
        if (!runnerId) return null;
        const version = hostLockVersion(runnerId, arch);
        const sha = sourceSha(model, controller);
        return version && sha ? `local-llm:${model}:${runnerId}:${version}:${sha}` : null;
    }

    // The engine's /v1/models, built fresh on every call from the seed catalog, the user's registry (once the engine exists) and the
    // deployment: it creates nothing and caches nothing.
    function models() {
        const controller = host?.current() ?? null;
        let target = null;
        try { target = controller ? controller.chatTarget() : null; } catch { target = null; }
        let list = [];
        try {
            seed ??= loadSeedCatalog(seedCatalogFile);
            list = mergeCatalog(seed, controller?.state?.registry ?? []);
        } catch { /* an unreadable catalog lists nothing */ }
        return {
            object: 'list',
            data: list.map((model) => ({
                id: model.id, object: 'model', owned_by: 'local-llm',
                x_local_llm: { ready: target?.modelId === model.id, runnerId: target?.modelId === model.id ? target.runnerId : null, sha256: model.sources?.gguf?.sha256 ?? null },
            })),
        };
    }

    return Object.freeze({
        ensure, stop, forget, status, identity, models, idleCheck, touch, redact,
        secrets: () => [...keysSeen],
        onDemand: onDemandEnabled,
        get lastRefusal() { return lastRefusal; },
        get ownDeployment() { return own; },
    });
}
