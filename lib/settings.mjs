// Secrets and paths. A key is read only from an env file in the user's home (~/.pworker/keys/<provider>.env) or from the
// environment, never from a configuration file; the process environment wins over the file.
import { readFileSync, existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
/** The Ploinky Workers folder (built-in defaults and CLI assets). */
export const HOME = resolve(HERE, '..');
export const DEFAULT_URL = 'http://127.0.0.1:18080';
/** The user's Ploinky Workers home: configuration, keys, data, cache, audit, logs, runs and models (PWORKER_HOME overrides). */
export const pworkerHome = (env = process.env) => resolve(expandHome(env.PWORKER_HOME || '~/.pworker'));
/** The record `pworker serve` writes in a home ({pid, host, port, startedAt}); one proxy serves one home. */
export const serverRecordFile = (home) => join(home, 'server.json');
export function readServerRecord(home) {
  try { const r = JSON.parse(readFileSync(serverRecordFile(home), 'utf8')); return Number.isSafeInteger(r?.pid) && Number.isSafeInteger(r?.port) ? r : null; } catch { return null; }
}
/**
 * The home of a Ploinky Workers server built from `config`: `config.taskHome` (an explicit home, for tests and embedders), else the
 * Pworker home (PWORKER_HOME). It does not depend on the data folder: a `dataDir` such as "~/pw-data" must not make `~` the home of
 * the engine. The server, its local-llm host and `pworker models hf-token` all use it, so the token file the command writes is the
 * file the host reads.
 */
export const serverHome = (config = {}, env = process.env) => config.taskHome ?? pworkerHome(env);
export const secretDirs = (env = process.env) => [join(pworkerHome(env), 'keys')];

export function expandHome(p) {
  return typeof p === 'string' && p.startsWith('~') ? join(homedir(), p.slice(1)) : p;
}

export function parseEnvFile(path) {
  const out = {};
  if (!path || !existsSync(path)) return out;
  for (const line of readFileSync(path, 'utf8').split(/\r?\n/)) {
    const m = /^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/.exec(line);
    if (!m || line.trim().startsWith('#')) continue;
    out[m[1]] = m[2].replace(/^(['"])(.*)\1$/, '$2');
  }
  return out;
}

/** The env-file values of an upstream, merged over the secret folders (earlier folders win). */
export function envValues(envFile, dirs = secretDirs()) {
  if (!envFile) return {};
  const name = basename(expandHome(envFile));
  const candidates = [...dirs.map((d) => join(expandHome(d), name)), ...(envFile.includes('/') ? [expandHome(envFile)] : [])];
  const out = {};
  for (const c of candidates.reverse()) Object.assign(out, parseEnvFile(c));
  return out;
}

// Resolves secrets per upstream: process env wins over the env files.
export function resolveUpstream(name, up, env = process.env, dirs = secretDirs(env)) {
  const file = envValues(up.envFile, dirs);
  const get = (k) => (k ? (env[k] ?? file[k]) : undefined);
  return {
    ...up,
    name,
    baseUrl: String(get(up.baseUrlVar) || up.baseUrl).replace(/\/+$/, ''),
    key: get(up.keyVar) || null,
  };
}

export function resolveProxyToken(config, env = process.env) {
  if (env.PWORKER_TOKEN) return env.PWORKER_TOKEN;
  for (const up of Object.values(config.providers ?? config.upstreams ?? {})) {
    const v = envValues(up?.envFile);
    if (v.PWORKER_TOKEN) return v.PWORKER_TOKEN;
  }
  return null;
}
