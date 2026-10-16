// Launcher test safety, shared by the test files that run a local-server starter (lib/local.mjs).
import { before, after, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { fileURLToPath } from 'node:url';

// The local-server tests run a starter over fake children whose PIDs are small numbers (1, 1901, 1902). A signal sent through the
// default process.kill would reach whatever owns that PID, and process.kill(-1, ...) reaches every process of the user. So every starter
// in those tests gets a recording killImpl (launcherHarness), and installKillGuard() replaces process.kill for the whole test file by a
// guard that throws and records; afterEach fails any test that reached it. A negative-control test builds a starter WITHOUT killImpl
// to show that the guard catches it and that nothing is signalled.
const realKill = process.kill;
export const guardHits = [];
export function installKillGuard() {
  before(() => {
    process.kill = function blockedKill(pid, sig) {
      guardHits.push([pid, sig]);
      process.stderr.write(`LAUNCHER GUARD: blocked process.kill(${pid}, ${sig})\n`); // visible even when the test fails for another reason first
      throw Object.assign(new Error(`process.kill(${pid}, ${sig}) is blocked in the launcher tests`), { code: 'EPERM' });
    };
  });
  after(() => { process.kill = realKill; });
  afterEach(() => { assert.deepEqual(guardHits.splice(0), [], 'a launcher test reached the process.kill guard (no killImpl injected)'); });
}

// A real file for `bin` and `gguf` (the starter only checks that they exist; the children are fakes).
export const FAKE_BIN = process.execPath, FAKE_GGUF = fileURLToPath(import.meta.url);

/**
 * Recording signals for local-server tests. `killImpl` records [pid, signal] and delivers the signal to the tracked child of that PID
 * (|pid|, because a group signal is negative); `fake(pid, opts)` makes a fake child that exits `exitDelayMs` after a signal it honors
 * (SIGTERM is ignored with `ignoreTerm`). The fake's own kill() is the fallback the launcher uses when killImpl throws; it only records,
 * so a test that expects no fallback asserts `fallbackKills` is empty. `trackReal` registers a REAL child that the test itself spawned:
 * its signals go through the ChildProcess handle (never process.kill), so the guard does not see them.
 */
export function launcherHarness() {
  const signals = [], fallbackKills = [], children = new Map();
  const killImpl = (pid, sig) => { signals.push([pid, sig]); children.get(Math.abs(pid))?.deliver(sig); };
  const fake = (pid, { exitDelayMs = 1, ignoreTerm = false, onExit = null } = {}) => {
    const c = new EventEmitter();
    c.pid = pid; c.exitCode = null;
    c.kill = (sig) => { fallbackKills.push([pid, sig]); };
    c.deliver = (sig) => {
      if (c.exitCode !== null || (sig === 'SIGTERM' && ignoreTerm)) return;
      setTimeout(() => { if (c.exitCode !== null) return; c.exitCode = 0; onExit?.(); c.emit('exit', 0, null); }, exitDelayMs);
    };
    children.set(pid, c);
    return c;
  };
  const trackReal = (c) => { c.deliver = (sig) => c.kill(sig); children.set(c.pid, c); return c; };
  return { signals, fallbackKills, killImpl, fake, trackReal };
}
