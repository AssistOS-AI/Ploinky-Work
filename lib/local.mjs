// Local llama-server upstreams started on demand (config.upstreams.<name>.start).
// The server starts only when the GPU is free: no reservation lock file and no compute process of another program on the GPU.
// A start that is not allowed or fails returns false; the request then fails as unreachable and falls back down its chain.
// An idle server is stopped after start.idleStopMs so the GPU is free for other work.
// Servers of one `start.exclusiveGroup` (e.g. the larger on-demand MoE models next to the always-on tiny) never run together: starting
// one first stops the others of its group that this proxy manages and waits for them to exit (stopPeers), so GPU memory is released.
import { spawn as spawnDefault, execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, openSync, closeSync } from 'node:fs';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { homedir } from 'node:os';
import { expandHome } from './settings.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));

/** A command on PATH (a bare name such as `llama-server`), or the name itself when it is not found (reported as missing). */
export function which(name) {
  for (const d of String(process.env.PATH || '').split(':')) if (d && existsSync(join(d, name))) return join(d, name);
  return name;
}

export function gpuComputePids() {
  try {
    const out = execFileSync('nvidia-smi', ['--query-compute-apps=pid', '--format=csv,noheader'], { encoding: 'utf8', timeout: 5000 });
    return out.split('\n').map((s) => Number(s.trim())).filter(Boolean);
  } catch { return null; } // no nvidia-smi: unknown
}

// Relative paths in `start` (gguf, bin, locks, logFile) resolve against baseDir: config.baseDir, itself relative to this folder.
export function createLocalStarter(up, { spawn = spawnDefault, fetchImpl = fetch, gpuPids = gpuComputePids, now = Date.now, baseDir = HERE, logDir = join(homedir(), '.pworker', 'logs'), beforeStart = null, busy = null, killImpl } = {}) {
  const abs = (p) => { const e = expandHome(p); return isAbsolute(e) ? e : join(baseDir, e); };
  const s = up.start;
  const port = Number(new URL(up.baseUrl).port);
  let child = null, starting = null, lastUse = 0, lastRefusal = null, timer = null;

  const healthy = async () => {
    try { return (await fetchImpl(up.baseUrl + '/health', { signal: AbortSignal.timeout(2000) })).ok; } catch { return false; }
  };
  function refuse(reason) { lastRefusal = { reason, at: new Date(now()).toISOString() }; return false; }

  async function start() {
    for (const lock of s.locks || []) if (existsSync(abs(lock))) return refuse(`GPU reserved (${lock})`);
    if (s.requireFreeGpu !== false) {
      const pids = gpuPids();
      if (pids && pids.length) return refuse(`GPU busy (compute pids ${pids.join(',')})`);
    }
    // Two kinds of local server: a llama-server over a GGUF (`bin`, `gguf`), or a Node script (`script`, e.g. the small-model
    // service of the structure and formalizer tiers) run with this Node binary and `--port`.
    if (beforeStart) await beforeStart();
    let cmd, argv;
    if (s.script) {
      const script = abs(s.script);
      if (!existsSync(script)) return refuse(`missing script ${script}`);
      for (const f of s.requires || []) if (!existsSync(abs(f))) return refuse(`missing ${abs(f)}`);
      cmd = process.execPath; argv = [script, '--port', String(port), ...(s.args || [])];
    } else {
      const gguf = abs(s.gguf), bin = String(s.bin).includes('/') || String(s.bin).startsWith('~') ? abs(s.bin) : which(s.bin);
      if (!existsSync(gguf)) return refuse(`missing model ${gguf}`);
      if (!existsSync(bin)) return refuse(`missing llama-server ${bin}`);
      cmd = bin; argv = ['-m', gguf, '--host', '127.0.0.1', '--port', String(port), '-a', s.alias || up.name, ...(s.args || [])];
    }
    const logFile = abs(s.logFile || join(logDir, `${up.name}.log`));
    mkdirSync(dirname(logFile), { recursive: true });
    const fd = openSync(logFile, 'a');
    const c = spawn(cmd, argv, { stdio: ['ignore', fd, fd] });
    closeSync(fd);
    let exited = false;
    c.on('exit', () => { exited = true; if (child === c) child = null; });
    child = c;
    const deadline = now() + (s.startTimeoutMs ?? 120000);
    while (now() < deadline) {
      if (exited) return refuse(`local server exited during start; see ${logFile}`);
      if (await healthy()) { lastRefusal = null; return true; }
      await new Promise((r) => setTimeout(r, 500));
    }
    stop();
    return refuse('local server did not become healthy in time');
  }

  // Sends a signal through killImpl (process.kill by default, read at each call so a test can replace it; a test passes a recording
  // fake, so no unit test can signal a real process). A signal killImpl cannot deliver goes to the child handle instead.
  function signal(c, sig) {
    if (!Number.isInteger(c.pid)) { try { c.kill(sig); } catch {} return; }
    try { (killImpl ?? process.kill)(c.pid, sig); return; } catch { /* fall back to the handle */ }
    try { c.kill(sig); } catch {}
  }

  // Stops the managed server: SIGTERM, then SIGKILL when it has not exited after stopWaitMs (default 30 s; a llama-server busy with a
  // long generation may ignore SIGTERM); the promise resolves when it has exited.
  function stop() {
    const c = child;
    child = null;
    if (!c || c.exitCode !== null) return Promise.resolve();
    const exited = new Promise((r) => {
      c.once('exit', r);
      setTimeout(() => { if (c.exitCode === null) { signal(c, 'SIGKILL'); setTimeout(r, 5000).unref?.(); } }, s.stopWaitMs ?? 30000).unref?.();
    });
    signal(c, 'SIGTERM');
    return exited;
  }

  function armIdleStop() {
    if (!s.idleStopMs || timer) return;
    timer = setInterval(() => {
      // A server with requests in flight or queued is in use: a long generation is never cut by the idle stop.
      if (busy?.()) lastUse = now();
      if (child && now() - lastUse > s.idleStopMs) stop();
      if (!child) { clearInterval(timer); timer = null; }
    }, Math.min(60000, s.idleStopMs));
    timer.unref?.();
  }

  return {
    async ensure() {
      lastUse = now();
      if (await healthy()) return true; // ours or started by someone else on that port
      starting ??= start().finally(() => { starting = null; });
      const ok = await starting;
      if (ok) armIdleStop();
      return ok;
    },
    stop,
    status: () => ({ managed: !!child, pid: child?.pid ?? null, last_refusal: lastRefusal }),
  };
}

/** Stops the managed servers of `name`'s exclusive group other than `name` (config: upstreams.<n>.start.exclusiveGroup). */
export async function stopPeers(name, starters, upstreamsConfig) {
  const group = upstreamsConfig[name]?.start?.exclusiveGroup;
  if (!group) return;
  await Promise.all(Object.entries(starters).filter(([n]) => n !== name && upstreamsConfig[n]?.start?.exclusiveGroup === group).map(([, st]) => st.stop()));
}
