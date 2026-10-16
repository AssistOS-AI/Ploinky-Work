// The local-server launcher with REAL processes (the unit tests of lib/local.mjs, with recorded signals, are in proxy.test.mjs).
// Everything here is owned by the test: a fixture server that exits by itself and a stand-in proxy process that the test kills.
import './home.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import net from 'node:net';
import { once } from 'node:events';
import { spawn, execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createLocalStarter } from '../lib/local.mjs';
import { installKillGuard, launcherHarness } from './launcher-safety.mjs';

installKillGuard();

const SERVER = fileURLToPath(new URL('./launcher-fixture-server.mjs', import.meta.url));
const PROXY = fileURLToPath(new URL('./launcher-fixture-proxy.mjs', import.meta.url));

const freePort = () => new Promise((resolve) => { const s = net.createServer(); s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => resolve(port)); }); });
const health = async (port) => { try { return (await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(1000) })).ok; } catch { return false; } };
const until = async (fn, ms = 5000) => { const end = Date.now() + ms; while (Date.now() < end) { if (await fn()) return true; await new Promise((r) => setTimeout(r, 25)); } return false; };
const firstLine = (stream, ms = 15000) => new Promise((resolve, reject) => {
  let text = '';
  const timer = setTimeout(() => reject(new Error(`no output within ${ms} ms: ${text}`)), ms);
  stream.on('data', (d) => { text += d; const i = text.indexOf('\n'); if (i >= 0) { clearTimeout(timer); resolve(text.slice(0, i)); } });
});
const psField = (field, pid) => { try { return execFileSync('ps', ['-o', `${field}=`, '-p', String(pid)], { encoding: 'utf8' }).trim(); } catch { return ''; } };

test('local: a server started by a real starter survives a kill -9 of its proxy and is reused by the next proxy without a spawn', { timeout: 60000 }, async () => {
  const dir = mkdtempSync(join(tmpdir(), 'pworker-survive-'));
  const port = await freePort();
  const proxy = spawn(process.execPath, [PROXY, SERVER, String(port), join(dir, 'fixture.log')], { stdio: ['ignore', 'pipe', 'inherit'] });
  try {
    const { runnerPid } = JSON.parse(await firstLine(proxy.stdout));
    assert.ok(Number.isInteger(runnerPid) && runnerPid > 1);
    assert.equal(await health(port), true, 'the real starter started the fixture server');
    assert.equal(psField('pgid', runnerPid), String(runnerPid), 'the server leads its own process group');
    // The test kills the proxy it spawned and owns, through its ChildProcess handle (not process.kill, which the guard blocks).
    const gone = once(proxy, 'exit');
    proxy.kill('SIGKILL');
    await gone;
    assert.equal(await health(port), true, 'the server answers after its proxy was killed with SIGKILL');
    assert.notEqual(psField('pid', runnerPid), '', 'the server process still exists');
    // The next proxy: a fresh starter finds the healthy port and reuses it.
    const h = launcherHarness();
    let spawned = 0;
    const next = createLocalStarter({ name: 'fixture', baseUrl: `http://127.0.0.1:${port}`, start: { script: SERVER, requireFreeGpu: false, logFile: join(dir, 'next.log') } },
      { gpuPids: () => [], killImpl: h.killImpl, spawn: () => { spawned += 1; throw new Error('must not spawn'); } });
    assert.equal(await next.ensure(), true);
    assert.equal(spawned, 0, 'a healthy server is reused, not spawned again');
    assert.deepEqual(h.signals, []);
    assert.deepEqual(next.status(), { managed: false, pid: null, last_refusal: null });
  } finally {
    // The stand-in proxy goes first, so it never sees its server exit; then the server is asked to exit.
    if (proxy.exitCode === null && proxy.signalCode === null) { const gone = once(proxy, 'exit'); proxy.kill('SIGKILL'); await gone; }
    await fetch(`http://127.0.0.1:${port}/shutdown`, { signal: AbortSignal.timeout(1000) }).catch(() => {});
    assert.equal(await until(async () => !(await health(port))), true, 'the fixture server is gone');
    rmSync(dir, { recursive: true, force: true });
  }
});
