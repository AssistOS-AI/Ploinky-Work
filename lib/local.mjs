// Local llama-server upstreams started on demand (config.upstreams.<name>.start).
// The server starts only when the GPU is free: no reservation lock file and no compute process of another program on the GPU.
// A start that is not allowed or fails returns false; the request then fails as unreachable and falls back down its chain.
// An idle server is stopped after start.idleStopMs so the GPU is free for other work.
// Servers of one `start.exclusiveGroup` (e.g. the larger on-demand MoE models next to the always-on tiny) never run together: starting
// one first stops the others of its group that this proxy manages and waits for them to exit (stopPeers), so GPU memory is released.
// A server runs through the engine's runner process (lib/local-llm/controller/runnerProcess.mjs): in its own process group, with its
// output written straight to its log file (no pipe to this proxy), so a proxy that dies leaves a healthy server that the next proxy
// reuses (ensure finds it healthy on its port). A stop signals the whole group: SIGTERM, then SIGKILL after start.stopGraceMs.
import { spawn as spawnDefault, execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, openSync, closeSync, appendFileSync } from 'node:fs';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { homedir } from 'node:os';
import { expandHome } from './settings.mjs';
import { startRunnerProcess } from './local-llm/controller/runnerProcess.mjs';

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
// The runner starts in baseDir (so a relative path in `start.args` means what it means everywhere else) with this process's environment.
// `spawn` and `killImpl` are for tests: a recording killImpl keeps a fake child from ever being signalled for real (process.kill is
// the default and is read at each start, so a test that replaced it sees every signal).
export function createLocalStarter(up, { spawn = spawnDefault, fetchImpl = fetch, gpuPids = gpuComputePids, now = Date.now, baseDir = HERE, logDir = join(homedir(), '.pworker', 'logs'), beforeStart = null, busy = null, killImpl } = {}) {
  const abs = (p) => { const e = expandHome(p); return isAbsolute(e) ? e : join(baseDir, e); };
  const s = up.start;
  const port = Number(new URL(up.baseUrl).port);
  let current = null, starting = null, lastUse = 0, lastRefusal = null, timer = null;

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
    // The runner writes straight to the log file; the controller's own lines (a failed spawn, an escalation) go to the same file.
    const fd = openSync(logFile, 'a');
    let runner;
    try {
      runner = startRunnerProcess({
        command: cmd, args: argv, env: process.env, cwd: existsSync(baseDir) ? baseDir : '/', killImpl, oomScoreAdj: null,
        log: { append: (stream, text) => { try { appendFileSync(logFile, `${new Date(now()).toISOString()} [${stream}] ${text}\n`); } catch { /* the log is best effort */ } } },
        spawnImpl: (command, args, options) => spawn(command, args, { ...options, stdio: ['ignore', fd, fd] }),
      });
    } catch (error) {
      return refuse(`could not start the local server: ${error.message}`);
    } finally {
      closeSync(fd);
    }
    let exit = null;
    runner.exited.then((result) => { exit = result; if (current === runner) current = null; });
    current = runner;
    const deadline = now() + (s.startTimeoutMs ?? 120000);
    while (now() < deadline) {
      if (exit) return refuse(`local server exited during start${exit.error ? ` (${exit.error.message})` : ''}; see ${logFile}`);
      if (await healthy()) { lastRefusal = null; return true; }
      await new Promise((r) => setTimeout(r, 500));
    }
    await stop();
    return refuse('local server did not become healthy in time');
  }

  // Stops the managed server and its whole process group: SIGTERM, then SIGKILL when it has not exited after start.stopGraceMs (default
  // 30 s; a llama-server busy with a long generation may ignore SIGTERM). The promise resolves when it has exited, or 5 s after the
  // SIGKILL at the latest (a process that cannot be killed must not block the start of its exclusive-group peer for ever).
  function stop() {
    const runner = current;
    current = null;
    if (!runner || !runner.running) return Promise.resolve();
    const graceMs = s.stopGraceMs ?? 30000;
    let cap;
    const capped = new Promise((resolve) => { cap = setTimeout(resolve, graceMs + 5000); cap.unref?.(); });
    return Promise.race([runner.stop({ graceMs }), capped]).then(() => { clearTimeout(cap); });
  }

  function armIdleStop() {
    if (!s.idleStopMs || timer) return;
    timer = setInterval(() => {
      // A server with requests in flight or queued is in use: a long generation is never cut by the idle stop.
      if (busy?.()) lastUse = now();
      if (current && now() - lastUse > s.idleStopMs) stop();
      if (!current) { clearInterval(timer); timer = null; }
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
    status: () => ({ managed: !!current, pid: current?.pid ?? null, last_refusal: lastRefusal }),
  };
}

/** Stops the managed servers of `name`'s exclusive group other than `name` (config: upstreams.<n>.start.exclusiveGroup). */
export async function stopPeers(name, starters, upstreamsConfig) {
  const group = upstreamsConfig[name]?.start?.exclusiveGroup;
  if (!group) return;
  await Promise.all(Object.entries(starters).filter(([n]) => n !== name && upstreamsConfig[n]?.start?.exclusiveGroup === group).map(([, st]) => st.stop()));
}
