// Administration of a Pworker home as a library: provider keys, custom providers, tier chains, live model catalogs and the proxy
// process. The interactive menu and the CLI use these functions, and so can another program on the same machine (for example a
// settings page) without shelling out. Keys are written only to <home>/keys/<provider>.env (mode 0600) and are never returned;
// callers see whether a key is configured, never its value.
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { ensureUserHome, loadLayers } from '../config.mjs';
import { pworkerHome, resolveUpstream, readServerRecord } from '../settings.mjs';
import { liveServer } from '../server.mjs';
import { createPworkerClient } from '../client.mjs';
import { fetchModelCatalog, isModelEligibleForTier, modelExclusionReason, normalizeModelCatalog, providerSpec } from './providers.mjs';

const BIN = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', 'bin', 'pworker.mjs');
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const TIER_NAME = /^[A-Za-z][\w-]{0,39}$/;
const PROVIDER_NAME = /^[a-z][a-z0-9_-]*$/;
const KEY_VAR = /^[A-Za-z_][A-Za-z0-9_]*$/;

const configFile = (env) => path.join(pworkerHome(env), 'config.json');

/** The user layer (<home>/config.json) as stored; created from the template when missing. */
export function readUserConfig(env = process.env) {
  ensureUserHome(env);
  return JSON.parse(fs.readFileSync(configFile(env), 'utf8'));
}

/** Replaces the user layer atomically (temporary file and rename), mode 0600. */
export function saveUserConfig(config, env = process.env) {
  const file = configFile(env);
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(config, null, 2) + '\n', { mode: 0o600 });
  fs.renameSync(tmp, file);
}

const merged = (env) => loadLayers({ env, project: false }).config;

/** Writes `keyVar=value` into <home>/keys/<envFile>, keeping the file's other variables (for example a base URL override). */
function writeKeyFile(env, envFile, keyVar, key) {
  if (!KEY_VAR.test(keyVar)) throw new Error('Invalid key variable name');
  ensureUserHome(env);
  const file = path.join(pworkerHome(env), 'keys', path.basename(envFile));
  const kept = fs.existsSync(file) ? fs.readFileSync(file, 'utf8').split(/\r?\n/).filter((line) => line && !new RegExp(`^\\s*(?:export\\s+)?${keyVar}\\s*=`).test(line)) : [];
  const lines = key ? [...kept, `${keyVar}=${JSON.stringify(String(key))}`] : kept;
  fs.writeFileSync(file, lines.join('\n') + '\n', { mode: 0o600 });
  try { fs.chmodSync(file, 0o600); } catch { /* best effort */ }
}

/**
 * Every provider of the merged configuration with its public status: `{name, baseUrl, local, keyConfigured, keyVar, envFile, noKey,
 * custom}`. Never includes a key.
 */
export function listProviders(env = process.env) {
  const config = merged(env);
  const user = readUserConfig(env).providers ?? {};
  const builtIn = loadLayers({ env, project: false, user: false }).config.providers ?? {};
  return Object.entries(config.providers ?? {}).filter(([name, spec]) => !name.startsWith('_') && spec && typeof spec === 'object').map(([name, spec]) => {
    const resolved = resolveUpstream(name, spec, env);
    return {
      name, baseUrl: resolved.baseUrl, local: Boolean(spec.start), noKey: Boolean(spec.noKey),
      keyConfigured: Boolean(resolved.key), keyVar: spec.keyVar ?? null, envFile: spec.envFile ?? null,
      custom: Object.hasOwn(user, name) && !Object.hasOwn(builtIn, name),
      note: spec._note ?? null,
    };
  });
}

/**
 * Sets (or with an empty key removes) the API key of a configured provider. With `validate` (default) the provider's live model
 * catalog is requested with the key first; a key the provider refuses is not saved. Returns `{provider, models}`.
 */
export async function setProviderKey(name, key, { validate = true, env = process.env, fetchImpl = fetch } = {}) {
  if (!PROVIDER_NAME.test(String(name))) throw new Error('Invalid provider name');
  const spec = merged(env).providers?.[name];
  if (!spec) throw new Error(`Unknown provider ${name}`);
  if (spec.start || spec.noKey) throw new Error(`${name} needs no key`);
  const envFile = spec.envFile ?? `${name}.env`;
  const keyVar = spec.keyVar ?? `PWORKER_${name.toUpperCase().replace(/[^A-Z0-9]/g, '_')}_API_KEY`;
  let models = null;
  if (key && validate) models = (await fetchModelCatalog(resolveUpstream(name, spec, env), { key, fetchImpl })).length;
  if (!spec.envFile || !spec.keyVar) {
    const user = readUserConfig(env);
    user.providers ??= {};
    user.providers[name] = { ...(user.providers[name] ?? {}), envFile, keyVar };
    saveUserConfig(user, env);
  }
  writeKeyFile(env, envFile, keyVar, key || null);
  return { provider: name, keyConfigured: Boolean(key), models };
}

/** Adds or replaces a custom OpenAI-compatible provider (endpoint, optional key, requests per minute), validated by its catalog. */
export async function saveCustomProvider(name, { endpoint, key = '', rpm = 60, validate = true, env = process.env, fetchImpl = fetch } = {}) {
  const prior = merged(env).providers?.[name] ?? {};
  if (prior.start) throw new Error('A managed local provider is configured in its start block, not as a custom endpoint');
  const spec = providerSpec(name, endpoint, key, rpm, prior);
  const models = validate ? (await fetchModelCatalog(spec, { key, fetchImpl })).length : null;
  const user = readUserConfig(env);
  user.providers ??= {};
  user.providers[name] = spec;
  saveUserConfig(user, env);
  if (key) writeKeyFile(env, spec.envFile ?? `${name}.env`, spec.keyVar, key);
  return { provider: name, models };
}

/** Saves the advertised context and output limits of a catalog model in the user layer (used for batch budgets). */
export function saveModelLimits(model, env = process.env) {
  if (!Number.isSafeInteger(model?.contextLength) || !Number.isSafeInteger(model?.maxOutputTokens) || model.contextLength < 1 || model.maxOutputTokens < 1) return false;
  const user = readUserConfig(env);
  user.providers ??= {};
  const provider = user.providers[model.provider] ??= {};
  provider.modelLimits ??= {};
  provider.modelLimits[model.id] = { ...model.modelLimits, contextTokens: model.contextLength, maxOutputTokens: model.maxOutputTokens,
    source: model.modelLimits?.source ?? 'provider model catalog', verifiedAt: model.modelLimits?.verifiedAt ?? new Date().toISOString().slice(0, 10) };
  saveUserConfig(user, env);
  return true;
}

/** The tiers of the merged configuration: `{name: [{upstream, model, extraBody?, timeoutMs?}] | 'alias'}` plus batching flags. */
export function listTiers(env = process.env) {
  const config = merged(env);
  const tiers = Object.fromEntries(Object.entries(config.tiers ?? {}).filter(([k]) => !k.startsWith('_') && k !== 'maxWaitMs'));
  const batching = Object.fromEntries(Object.entries(config.batching ?? {}).map(([k, v]) => [k, Boolean(v?.enabled)]));
  return { tiers, batching };
}

const cleanEntry = (entry) => {
  const upstream = entry?.upstream ?? entry?.provider;
  if (!PROVIDER_NAME.test(String(upstream ?? ''))) throw new Error('Each tier entry needs a provider (upstream)');
  if (typeof entry.model !== 'string' || !entry.model.trim()) throw new Error('Each tier entry needs a model id');
  const out = { upstream, model: entry.model.trim() };
  if (entry.extraBody != null) {
    if (typeof entry.extraBody !== 'object' || Array.isArray(entry.extraBody)) throw new Error('extraBody must be a JSON object');
    out.extraBody = entry.extraBody;
  }
  if (entry.timeoutMs != null) {
    if (!Number.isSafeInteger(entry.timeoutMs) || entry.timeoutMs < 1) throw new Error('timeoutMs must be a positive integer');
    out.timeoutMs = entry.timeoutMs;
  }
  if (entry.minTokens != null) {
    if (!Number.isSafeInteger(entry.minTokens) || entry.minTokens < 1) throw new Error('minTokens must be a positive integer');
    out.minTokens = entry.minTokens;
  }
  if (typeof entry.prompt === 'string' && entry.prompt) out.prompt = entry.prompt;
  return out;
};

/**
 * Replaces a tier's chain in the user layer. `entries`: an ordered list of `{upstream|provider, model, extraBody?, timeoutMs?, minTokens?}`
 * (`minTokens` raises a smaller max_tokens of a request, for models that always reason before answering; the
 * first usable entry serves, the rest are fallbacks) or the name of another tier (an alias). `batch` sets batching.<tier>.enabled.
 * With `validate`, every entry's model must be in its provider's live catalog (read through the running proxy).
 */
export async function setTierChain(tier, entries, { batch, validate = false, env = process.env } = {}) {
  if (!TIER_NAME.test(String(tier)) || tier === 'maxWaitMs') throw new Error('Invalid tier name');
  const value = typeof entries === 'string' ? entries : (Array.isArray(entries) ? entries : []).map(cleanEntry);
  if (typeof value === 'string' && !TIER_NAME.test(value)) throw new Error('Invalid tier alias');
  if (Array.isArray(value) && !value.length) throw new Error('A tier needs at least one entry');
  if (validate && Array.isArray(value)) {
    for (const entry of value) {
      const catalog = await providerCatalog(entry.upstream, { env });
      const found = catalog.find((m) => m.id === entry.model);
      if (!found) throw new Error(`Model ${entry.model} is not in the live catalog of ${entry.upstream}`);
      if (!isModelEligibleForTier(found)) throw new Error(`Model ${entry.model} cannot serve a text tier: ${modelExclusionReason(found)}`);
      saveModelLimits(found, env);
    }
  }
  const user = readUserConfig(env);
  user.tiers ??= {};
  user.tiers[tier] = value;
  if (batch !== undefined) { user.batching ??= {}; user.batching[tier] = { enabled: Boolean(batch) }; }
  saveUserConfig(user, env);
  return { tier, entries: value };
}

/** Removes a user-defined tier override (the built-in default, if any, applies again). */
export function removeTier(tier, { env = process.env } = {}) {
  const user = readUserConfig(env);
  if (user.tiers && Object.hasOwn(user.tiers, tier)) { delete user.tiers[tier]; saveUserConfig(user, env); return true; }
  return false;
}

const client = (env) => createPworkerClient({ purpose: 'pworker:admin', client: 'pworker-admin', env });

/** A provider's live catalog, normalized ({id, name, provider, prices, contextLength, ...}), read through the proxy (started if needed). */
export async function providerCatalog(name, { env = process.env } = {}) {
  const response = await client(env).providerModels(name);
  const spec = merged(env).providers?.[name] ?? {};
  return normalizeModelCatalog(name, spec, response.data ?? []).map(({ raw, ...model }) => model);
}

/** The pid of this home's live proxy, or null. */
export async function proxyPid(env = process.env) { return (await liveServer(pworkerHome(env)))?.pid ?? null; }

/** Starts this home's proxy as a detached process when none runs; resolves when it answers. */
export async function startProxy({ env = process.env, config = null } = {}) {
  if (await proxyPid(env)) return false;
  const home = pworkerHome(env);
  fs.mkdirSync(path.join(home, 'logs'), { recursive: true });
  const log = fs.openSync(path.join(home, 'logs', 'pworker.log'), 'a');
  const child = spawn(process.execPath, [BIN, 'serve', ...(config ? ['--config', config] : [])], { detached: true, stdio: ['ignore', log, log], env });
  child.unref(); fs.closeSync(log);
  for (let i = 0; i < 80; i++) { await sleep(250); if (await proxyPid(env)) return true; }
  throw new Error(`The proxy did not start. See ${path.join(home, 'logs', 'pworker.log')}`);
}

/** Stops this home's proxy (SIGTERM); resolves when its record is gone or it no longer answers. */
export async function stopProxy({ env = process.env } = {}) {
  const pid = await proxyPid(env);
  if (!pid) return false;
  process.kill(pid, 'SIGTERM');
  for (let i = 0; i < 50; i++) { await sleep(100); if (!readServerRecord(pworkerHome(env)) || !(await proxyPid(env))) return true; }
  return false;
}

/** Restarts the proxy so it reads changed keys and tiers; starts it when `always` and none runs. */
export async function restartProxy({ env = process.env, always = false } = {}) {
  const running = await proxyPid(env);
  if (running) await stopProxy({ env });
  if (running || always) await startProxy({ env });
  return Boolean(running || always);
}
