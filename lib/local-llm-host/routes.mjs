// The admin routes of the local-llm engine, `/v1/local-llm/*`. The server mounts them before the request admission and the
// `POST /v1/*` forwarder, behind the Host/Origin guard and the optional proxy token (nothing more: no extra credential).
//
//   GET  /v1/local-llm                         overview: runners (the unsupported ones with their reasons), models, hardware, deployment
//   GET  /v1/local-llm/status?sinceSeq=N       phase, deployment, progress, log lines since N
//   POST /v1/local-llm/preview                 {modelId, runnerId?, params?}: the admission verdict, no process
//   POST /v1/local-llm/run                     {modelId, runnerId?, params?, replace?, requestId?}: 202 {accepted | duplicate, deployment}
//   POST /v1/local-llm/stop | /cancel          stop the deployment | cancel a download
//   POST /v1/local-llm/weights/delete          {modelId, runnerId?, format?}
//   POST /v1/local-llm/models/lookup | add | update | remove
//   POST /v1/local-llm/runners/{id}/install | uninstall      {acceptLicence?}
//
// Errors are `{error: {type: <code>, message}}` with the message redacted (Hugging Face token, per-start keys): 400 for a body that
// is not a JSON object, 413 above 32 MiB, 404 for an unknown model, runner or route, 409 for a conflict, 422 for an admission
// refusal and 503 when the engine is unavailable. A read that cannot get the engine (disabled, unsupported, owned by another
// process) answers 200 with `available: false` and the refusal, so a status command can say why.

import crypto from 'node:crypto';

import { defaultRunnerFor } from './gate.mjs';

const MAX_BODY = 32 * 1024 * 1024;
const PREFIX = '/v1/local-llm';

const NOT_FOUND = new Set(['unknown_model', 'unknown_runner', 'not_found']);
const CONFLICT = new Set(['busy', 'in_use', 'weights_present', 'duplicate_model', 'runner_not_installed', 'runner_disabled', 'local_llm_disabled', 'not_downloading', 'licence_required', 'not_installable', 'runner_unsupported', 'runner_unavailable', 'unpinned', 'no_source', 'read_only']);
const UNAVAILABLE = new Set(['shutting_down', 'not_ready', 'owner_held', 'owner_port_collision', 'owner_port_taken', 'invalid_socket_dir', 'invalid_caps', 'local_llm_unsupported', 'socket_path_too_long', 'start_timeout']);

/** The HTTP status of an engine error code. */
export function statusFor(code) {
    if (NOT_FOUND.has(code)) return 404;
    if (CONFLICT.has(code)) return 409;
    if (UNAVAILABLE.has(code)) return 503;
    if (typeof code === 'string' && code.startsWith('admission_')) return 422;
    if (typeof code === 'string' && (code.startsWith('invalid_') || code === 'missing_file')) return 400;
    if (code === 'lookup_failed') return 502;
    return 500;
}

async function readJsonBody(req) {
    const chunks = [];
    let size = 0;
    for await (const chunk of req) {
        size += chunk.length;
        if (size > MAX_BODY) throw Object.assign(new Error(`request body larger than ${MAX_BODY} bytes`), { status: 413, code: 'request_too_large' });
        chunks.push(chunk);
    }
    const text = Buffer.concat(chunks).toString('utf8');
    if (!text.trim()) return {};
    let value;
    try { value = JSON.parse(text); } catch { throw Object.assign(new Error('the request body is not valid JSON'), { status: 400, code: 'invalid_json' }); }
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw Object.assign(new Error('the request body must be a JSON object'), { status: 400, code: 'invalid_json' });
    return value;
}

/**
 * `host`: the local-llm host (lib/local-llm-host/host.mjs); `gate`: the gate (or a function that returns it), so a stop also clears its idle timer; `redact`: the
 * text filter for messages (default: the host's). Returns `(req, res, url, { sendJson }) => Promise<false | undefined>`: false
 * when the request is not an engine route.
 */
export function createLocalLlmRoutes({ host, gate = null, redact = null, newRequestId = () => crypto.randomUUID().replace(/-/g, '') } = {}) {
    const gateNow = () => (typeof gate === 'function' ? gate() : gate);
    const clean = (text) => (redact ?? gateNow()?.redact ?? host.redact)(String(text));

    const unsupportedRunners = () => host.describe().runners.filter((runner) => !runner.supported).map(({ id, reason }) => ({ id, reason }));

    async function engine() {
        return host.controller();
    }

    // What a read returns (the runner's log lines, the deployment) goes through the same filter as an error text: if a runner or the
    // engine ever echoed the key of the run or the Hugging Face token, it would leave as ***.
    function scrub(value, controller) {
        let text = JSON.stringify(value);
        const secrets = [...(host.secretValues?.() ?? [])];
        try { const key = controller.chatTarget().apiKey; if (key) secrets.push(key); } catch { /* nothing is serving: no key */ }
        for (const secret of secrets) if (secret && text.includes(secret)) text = text.split(secret).join('***');
        return JSON.parse(text);
    }

    // An engine that cannot be had is data for a read, an error for an action.
    async function unavailable(error, sendJson, res) {
        if (!error || typeof error.code !== 'string') throw error;
        const described = host.describe();
        return sendJson(res, 200, {
            available: false,
            refusal: { code: error.code, message: clean(error.message) },
            ownership: described.ownership,
            host: described,
            unsupportedRunners: unsupportedRunners(),
            deployment: null,
        });
    }

    // The runner of a request that names none: the one the gate picks for the machine's profile. When this machine cannot use it the
    // answer is a 409 that tells the caller to choose with --runner, never an undefined runner handed to the engine.
    async function runnerOf(controller, body) {
        if (body.runnerId !== undefined && body.runnerId !== null) return body.runnerId;
        const overview = await controller.overview();
        const runnerId = defaultRunnerFor(overview.profile);
        const info = (overview.runners ?? []).find((candidate) => candidate.id === runnerId);
        if (!info || info.supported === false) {
            const why = info?.unsupportedReason ?? info?.reason ?? 'it is not available on this machine';
            throw Object.assign(new Error(`The default runner ${runnerId} cannot be used here: ${why}. Choose one with --runner (see \`pworker models runners\`).`), { code: 'runner_unsupported' });
        }
        return runnerId;
    }

    const actions = {
        async run(controller, body) {
            const result = await controller.run({ requestId: body.requestId ?? newRequestId(), modelId: body.modelId, runnerId: await runnerOf(controller, body), params: body.params, replace: body.replace === true });
            return [202, result];
        },
        async stop(controller) { gateNow()?.forget(); return [200, await controller.stop()]; },
        async cancel(controller) { return [200, await controller.cancelDownload()]; },
        async 'weights/delete'(controller, body) { return [200, await controller.deleteWeights({ modelId: body.modelId, runnerId: body.runnerId, format: body.format })]; },
        async 'models/lookup'(controller, body) { return [200, await controller.lookupModel(body)]; },
        async 'models/add'(controller, body) { return [200, await controller.addModel(body)]; },
        async 'models/update'(controller, body) { return [200, await controller.updateModel(body)]; },
        async 'models/remove'(controller, body) { return [200, await controller.removeModel({ modelId: body.modelId ?? body.id })]; },
    };

    return async function localLlmRoutes(req, res, url, { sendJson }) {
        const pathname = url.pathname;
        if (pathname !== PREFIX && !pathname.startsWith(`${PREFIX}/`)) return false;
        const rest = pathname.slice(PREFIX.length).replace(/^\/+/, '');
        const fail = (status, type, message) => sendJson(res, status, { error: { type, message: clean(message) } });
        try {
            if (req.method === 'GET' && rest === '') {
                let controller;
                try { controller = await engine(); } catch (error) { return await unavailable(error, sendJson, res); }
                const overview = await controller.overview();
                return sendJson(res, 200, scrub({ available: true, ...overview, ownership: host.describe().ownership, unsupportedRunners: unsupportedRunners() }, controller));
            }
            if (req.method === 'GET' && rest === 'status') {
                let controller;
                try { controller = await engine(); } catch (error) { return await unavailable(error, sendJson, res); }
                const status = await controller.status({ sinceSeq: Number(url.searchParams.get('sinceSeq')) || 0 });
                return sendJson(res, 200, scrub({ available: true, ...status, ownership: host.describe().ownership, startedBy: gateNow()?.status().startedBy ?? null, ...(host.caps ? { caps: true } : {}) }, controller));
            }
            if (req.method !== 'POST') return fail(404, 'not_found', `${req.method} ${pathname}`);
            const runner = /^runners\/([^/]+)\/(install|uninstall)$/.exec(rest);
            if (rest !== 'preview' && !Object.hasOwn(actions, rest) && !runner) return fail(404, 'not_found', `${req.method} ${pathname}`);
            const body = await readJsonBody(req);
            // Nothing is running when the engine was never used: a stop or a cancel must not create it.
            if ((rest === 'stop' || rest === 'cancel') && !host.current()) {
                return rest === 'stop' ? sendJson(res, 200, { deployment: null }) : fail(409, 'not_downloading', 'No download is in progress.');
            }
            const controller = await engine();
            if (rest === 'preview') {
                const { preview } = await controller.overview({ preview: { modelId: body.modelId, runnerId: await runnerOf(controller, body), params: body.params } });
                return sendJson(res, 200, { preview });
            }
            if (runner) {
                const runnerId = decodeURIComponent(runner[1]);
                if (runner[2] === 'install') return sendJson(res, 202, await controller.installRunner({ runnerId, acceptLicence: body.acceptLicence === true, acceptedBy: body.acceptedBy ?? null }));
                return sendJson(res, 200, await controller.uninstallRunner({ runnerId }));
            }
            const [status, result] = await actions[rest](controller, body);
            return sendJson(res, status, result);
        } catch (error) {
            if (res.headersSent) { try { res.end(); } catch { /* closed */ } return undefined; }
            const type = typeof error?.code === 'string' && !/^(E[A-Z]+)$/.test(error.code) ? error.code : 'internal_error';
            const status = error?.status === 413 || error?.status === 400 ? error.status : statusFor(type);
            return fail(status, type, error?.message ?? String(error));
        }
    };
}
