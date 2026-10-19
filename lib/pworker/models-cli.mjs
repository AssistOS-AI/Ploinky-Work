// `pworker models ...`: the verbs of the local-llm engine. `models start NAME` and `models stop NAME` (a configured local model
// server) are not here: they keep calling /v1/local/NAME/(start|stop). A bare `models`, `--provider` and `--all` are not here either.
//
//   models catalog|runners [--json]          what can run here
//   models lookup REPO [--file F] [--revision R]    read a Hugging Face model's metadata (writes nothing)
//   models add|update FILE|-   models remove ID     the user's models
//   models preview ID [--runner R] [--ctx N] [--parallel N] [--param k=v]...    the admission verdict, no process
//   models run ID [same flags] [--replace] [--detach]    start the one deployment; shows the download until it is ready
//   models stop        models cancel         stop the deployment | cancel its download
//   models status [--follow] [--json]        phase, deployment, why the engine is unavailable
//   models delete-weights ID [--runner R]    models runner install|uninstall ID [--accept-licence] [--detach]
//   models hf-token                          store a Hugging Face token (hidden input, or stdin when not a terminal)
import fs from 'node:fs';
import { hfTokenFile, readTokenInput, writeHfToken, checkToken } from './hf-token.mjs';

export const ENGINE_VERBS = Object.freeze(['catalog', 'runners', 'lookup', 'add', 'update', 'remove', 'preview', 'run', 'stop', 'status', 'cancel', 'delete-weights', 'runner', 'hf-token']);
const VALUE_FLAGS = new Set(['runner', 'ctx', 'parallel', 'param', 'file', 'revision', 'purpose', 'port', 'host']);

/** Whether `args` (everything after `pworker`) is an engine verb: `models stop` without a name is the engine's; with a name it is a local server's. */
export function isEngineCommand(args) {
  if (args[0] !== 'models' || !ENGINE_VERBS.includes(args[1])) return false;
  if (args[1] === 'stop') return args[2] === undefined || args[2].startsWith('--');
  return true;
}

/** `{ positional: [...], flags: Map-like object, params: [k=v...] }` of the arguments after the verb. */
export function parseVerbArguments(rest) {
  const positional = [], flags = {}, params = [];
  for (let i = 0; i < rest.length; i++) {
    const arg = rest[i];
    if (!arg.startsWith('--')) { positional.push(arg); continue; }
    const name = arg.slice(2);
    if (VALUE_FLAGS.has(name)) {
      const value = rest[++i];
      if (value === undefined || value.startsWith('--')) throw new Error(`--${name} needs a value`);
      if (name === 'param') params.push(value); else flags[name] = value;
    } else flags[name] = true;
  }
  return { positional, flags, params };
}

function parseValue(text) {
  if (text === 'true') return true;
  if (text === 'false') return false;
  if (text === 'null') return null;
  return /^-?\d+(\.\d+)?$/.test(text) ? Number(text) : text;
}

/** The run parameters of the flags: --ctx, --parallel and every --param key=value (numbers, booleans and null are typed). */
export function runParameters({ flags, params }) {
  const out = {};
  if (flags.ctx !== undefined) out.ctxSize = Number(flags.ctx);
  if (flags.parallel !== undefined) out.parallel = Number(flags.parallel);
  for (const pair of params) {
    const at = pair.indexOf('=');
    if (at < 1) throw new Error(`--param needs key=value, not "${pair}"`);
    out[pair.slice(0, at)] = parseValue(pair.slice(at + 1));
  }
  for (const key of ['ctxSize', 'parallel']) if (key in out && !Number.isFinite(out[key])) throw new Error(`--${key === 'ctxSize' ? 'ctx' : key} needs a number`);
  return Object.keys(out).length ? out : undefined;
}

const UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
export function formatBytes(n) {
  if (!Number.isFinite(n) || n < 0) return '?';
  let value = n, unit = 0;
  while (value >= 1024 && unit < UNITS.length - 1) { value /= 1024; unit += 1; }
  return `${unit === 0 ? value : value.toFixed(value >= 100 ? 0 : 1)} ${UNITS[unit]}`;
}
const formatEta = (seconds) => (Number.isFinite(seconds) && seconds >= 0 ? (seconds >= 3600 ? `${Math.floor(seconds / 3600)}h${String(Math.floor(seconds / 60) % 60).padStart(2, '0')}m` : seconds >= 60 ? `${Math.floor(seconds / 60)}m${String(Math.round(seconds) % 60).padStart(2, '0')}s` : `${Math.round(seconds)}s`) : '?');

/** One progress line of a status: the phase, the model and, while it transfers, the bytes, the rate and the time left. */
export function progressLine(status) {
  const deployment = status.deployment;
  if (!deployment) return `phase ${status.phase ?? 'idle'}`;
  const head = `${deployment.phase} ${deployment.modelId}${deployment.runnerId ? ` on ${deployment.runnerId}` : ''}`;
  const download = deployment.download;
  if (download && ['downloading', 'copying', 'verifying'].includes(deployment.phase) && download.total) {
    const percent = Math.min(100, Math.floor((download.bytes / download.total) * 100));
    return `${head}: ${formatBytes(download.bytes)} / ${formatBytes(download.total)} (${percent}%)${download.rate ? `, ${formatBytes(download.rate)}/s, ETA ${formatEta(download.etaSeconds)}` : ''}`;
  }
  return deployment.error ? `${head}: ${typeof deployment.error === 'string' ? deployment.error : deployment.error.message ?? 'error'}` : head;
}

function table(rows) {
  const widths = rows[0].map((_, column) => Math.max(...rows.map((row) => String(row[column] ?? '').length)));
  return rows.map((row) => row.map((cell, column) => String(cell ?? '').padEnd(widths[column])).join('  ').trimEnd()).join('\n');
}

const refusalText = (status) => (status.available === false ? `The local-llm engine is unavailable: ${status.refusal?.code ?? 'unknown'}: ${status.refusal?.message ?? ''}` : null);

const capSize = (bytes) => (bytes >= 1024 ** 3 ? `${+(bytes / 1024 ** 3).toFixed(2)} GiB` : `${Math.round(bytes / 1024 ** 2)} MiB`);

/** The line that names the hardware caps in force (`localLlm.caps`), from the overview's limits; null when there are none. */
export function capsLine(overview) {
  const budget = overview?.limits?.budget;
  const entries = Object.entries(budget?.caps ?? {}).filter(([, value]) => Number.isFinite(value));
  if (!entries.length) return null;
  // ram and gpuMemory are sizes; threads, maxContext and maxParallel are counts.
  const shown = (key, value) => (key === 'ram' || key === 'gpuMemory' ? capSize(value) : String(value));
  return `caps in force (localLlm.caps): ${entries.map(([key, value]) => `${key} ${shown(key, value)}`).join(', ')}; ${budget.mode ?? 'refuse'} mode`;
}

// The caps line for the engine now, or null: no caps, no overview, or an engine that cannot answer (the note must never fail a verb).
async function capsNote(engine) {
  try { return typeof engine.overview === 'function' ? capsLine(await engine.overview()) : null; } catch { return null; }
}

async function readJsonInput(source, { stdin }) {
  let text = '';
  if (source === '-') { for await (const chunk of stdin) text += chunk; }
  else text = fs.readFileSync(source, 'utf8');
  try { return JSON.parse(text); } catch { throw new Error(`${source === '-' ? 'stdin' : source} is not valid JSON`); }
}

/**
 * Runs one engine verb. `args` is the full argument list (`['models', 'run', 'qwen3-0.6b', '--detach']`), `client` a pworker client
 * (`client.engine`), `io` = { out(text), err(text), stdin, stdout, isTTY, signal?, home }. Resolves the exit code. `pollMs` and
 * `lineMs` are the progress intervals (1 s, and 5 s when the output is not a terminal).
 */
export async function runEngineVerb({ args, client, io, pollMs = 1000, lineMs = 5000 }) {
  const verb = args[1];
  const { positional, flags, params } = parseVerbArguments(args.slice(2));
  const json = flags.json === true;
  const engine = client.engine;
  const print = (value) => io.out(`${JSON.stringify(value, null, 2)}\n`);
  const need = (what) => { if (!positional[0]) throw new Error(`Usage: pworker models ${verb} ${what}`); return positional[0]; };

  async function follow(until) {
    let lastLine = '', lastAt = 0, seq = 0;
    for (;;) {
      if (io.signal?.aborted) return { aborted: true };
      const status = await engine.status(seq);
      seq = status.nextSeq ?? seq;
      if (status.available === false) return { status };
      const line = progressLine(status);
      const now = Date.now();
      if (io.isTTY) io.out(`\r\x1b[K${line}`);
      else if (line !== lastLine && (now - lastAt >= lineMs || line.split(':')[0] !== lastLine.split(':')[0])) { io.out(`${line}\n`); lastAt = now; }
      lastLine = line;
      const verdict = until(status);
      if (verdict) { if (io.isTTY) io.out('\n'); return { status, verdict }; }
      await new Promise((resolve) => { const timer = setTimeout(resolve, pollMs); io.signal?.addEventListener('abort', () => { clearTimeout(timer); resolve(); }, { once: true }); });
    }
  }

  switch (verb) {
    case 'catalog': {
      const overview = await engine.overview();
      if (json) { print(overview.available === false ? overview : overview.models); return 0; }
      if (overview.available === false) { io.out(`${refusalText(overview)}\n`); return 0; }
      const rows = [['ID', 'NAME', 'SIZE', 'WEIGHTS', 'RUNNERS']];
      for (const model of overview.models) {
        const weights = Object.values(model.weights ?? {})[0];
        rows.push([model.id, model.displayName, formatBytes(Object.values(model.sources ?? {})[0]?.size), weights?.download?.state ?? '-', Object.entries(model.runners ?? {}).map(([id, entry]) => `${id}:${entry.admission?.status ?? '?'}`).join(' ')]);
      }
      io.out(`${table(rows)}\n`);
      return 0;
    }
    case 'runners': {
      const overview = await engine.overview();
      if (json) { print(overview.available === false ? overview : { runners: overview.runners, unsupportedRunners: overview.unsupportedRunners }); return 0; }
      if (overview.available === false) io.out(`${refusalText(overview)}\n`);
      const known = new Set((overview.runners ?? []).map((runner) => runner.id));
      const rows = [['ID', 'STATE', 'VERSION', 'NOTE']];
      for (const runner of overview.runners ?? []) {
        rows.push([runner.id, runner.supported === false ? 'unsupported' : runner.installed ? 'installed' : 'not installed', runner.version ?? runner.pinnedVersion ?? '', runner.unsupportedReason ?? runner.reason ?? '']);
      }
      for (const runner of overview.unsupportedRunners ?? []) if (!known.has(runner.id) || overview.available === false) rows.push([runner.id, 'unsupported', '', runner.reason]);
      io.out(`${table(rows)}\n`);
      return 0;
    }
    case 'lookup': {
      const repo = need('REPO [--file F] [--revision R]');
      print(await engine.lookup({ repo, ...(flags.file ? { file: flags.file } : {}), ...(flags.revision ? { revision: flags.revision } : {}) }));
      return 0;
    }
    case 'add': print(await engine.add(await readJsonInput(need('FILE|-'), io))); return 0;
    case 'update': print(await engine.update(await readJsonInput(need('FILE|-'), io))); return 0;
    case 'remove': print(await engine.remove(need('ID'))); return 0;
    case 'preview': {
      const result = await engine.preview({ modelId: need('ID'), runnerId: flags.runner, params: runParameters({ flags, params }) });
      if (json) { print(result); return 0; }
      const admission = result.preview?.admission;
      io.out(`${positional[0]}: ${admission?.status ?? 'unknown'}${admission?.reason ? ` (${admission.reason})` : ''}\n`);
      if (admission && admission.status !== 'ok') { const caps = await capsNote(engine); if (caps) io.out(`${caps}\n`); }
      return 0;
    }
    case 'run': {
      const modelId = need('ID [--runner R] [--ctx N] [--parallel N] [--param k=v]... [--replace] [--detach]');
      let result;
      try {
        result = await engine.run({ modelId, runnerId: flags.runner, params: runParameters({ flags, params }), replace: flags.replace === true });
      } catch (error) {
        // Only a refusal by admission is about the caps.
        const caps = /^admission_/.test(String(error?.type ?? error?.code ?? '')) ? await capsNote(engine) : null;
        if (caps) io.err(`${caps}\n`);
        throw error;
      }
      if (flags.detach === true) { if (json) print(result); else io.out(`${result.duplicate ? 'already accepted' : 'accepted'}: ${modelId}; follow it with \`pworker models status --follow\`\n`); return 0; }
      const ended = await follow((status) => {
        const phase = status.deployment?.phase;
        if (phase === 'ready') return 'ready';
        if (!status.deployment || ['error', 'paused', 'idle'].includes(phase)) return 'failed';
        return null;
      });
      if (ended.aborted) { io.err(`Detached; the model keeps starting. Follow it with \`pworker models status --follow\`; cancel the download with \`pworker models cancel\`.\n`); return 130; }
      if (ended.status?.available === false) { io.err(`${refusalText(ended.status)}\n`); return 1; }
      if (ended.verdict === 'ready') { io.out(`${modelId} is ready.\n`); return 0; }
      io.err(`${modelId} did not start: ${progressLine(ended.status)}\n`);
      { const caps = await capsNote(engine); if (caps) io.err(`${caps}\n`); }
      return 1;
    }
    case 'stop': {
      const result = await engine.stop();
      if (json) print(result); else io.out('Stopped.\n');
      return 0;
    }
    case 'cancel': {
      const result = await engine.cancel();
      if (json) print(result); else io.out('Download cancelled.\n');
      return 0;
    }
    case 'status': {
      if (flags.follow === true) {
        await follow(() => null);
        return 0;
      }
      const status = await engine.status();
      if (json) { print(status); return 0; }
      const refused = refusalText(status);
      if (refused) { io.out(`${refused}\n`); return 0; }
      io.out(`${progressLine(status)}${status.startedBy ? ` (started by ${status.startedBy})` : ''}\n`);
      if (status.ownership?.refusal) io.out(`ownership: ${status.ownership.refusal.code}: ${status.ownership.refusal.message}\n`);
      const caps = status.caps === true ? await capsNote(engine) : null;
      if (caps) io.out(`${caps}\n`);
      return 0;
    }
    case 'delete-weights': {
      const result = await engine.deleteWeights({ modelId: need('ID [--runner R]'), runnerId: flags.runner });
      if (json) print(result); else io.out(`Deleted the weights of ${positional[0]}.\n`);
      return 0;
    }
    case 'runner': {
      const action = positional[0], id = positional[1];
      if (!['install', 'uninstall'].includes(action) || !id) throw new Error('Usage: pworker models runner install|uninstall ID [--accept-licence] [--detach]');
      if (action === 'uninstall') { print(await engine.uninstallRunner(id)); return 0; }
      const started = await engine.installRunner(id, flags['accept-licence'] === true);
      if (flags.detach === true || started.installed) { io.out(started.installed ? `${id} is installed.\n` : `Installing ${id}; check \`pworker models runners\`.\n`); return 0; }
      for (;;) {
        if (io.signal?.aborted) { io.err(`Detached; the install continues. Check it with \`pworker models runners\`.\n`); return 130; }
        const overview = await engine.overview();
        const info = overview.runners?.find((runner) => runner.id === id)?.install?.state;
        const phase = info?.phase;
        const download = info?.download;
        io.out(`${io.isTTY ? '\r\x1b[K' : ''}install ${id}: ${phase ?? 'queued'}${download?.total ? ` ${formatBytes(download.bytes)} / ${formatBytes(download.total)}` : ''}${io.isTTY ? '' : '\n'}`);
        if (phase === 'installed') { io.out('\n'); return 0; }
        if (phase === 'error' || phase === 'paused') { io.err(`\nThe install of ${id} stopped: ${info?.error ?? info?.pausedReason ?? phase}\n`); return 1; }
        await new Promise((resolve) => setTimeout(resolve, pollMs));
      }
    }
    case 'hf-token': {
      const token = await readTokenInput({ input: io.stdin, output: io.stdout });
      if (token === null) { io.err('Cancelled; nothing was stored.\n'); return 1; }
      const verdict = checkToken(token);
      if (!verdict.ok) { io.err(`Not stored: ${verdict.reason}.\n`); return 1; }
      const file = writeHfToken(io.home, token);
      io.out(`Hugging Face token stored in ${file} (mode 0600). Restart the proxy for the engine to use it: pworker stop && pworker start\n`);
      return 0;
    }
    default:
      throw new Error(`Unknown models command: ${verb}`);
  }
}

export { hfTokenFile };
