// The recovery of an orphaned runner on a real /proc (Linux only): a pworker that is killed with SIGKILL, at the ready state or
// at the launch-record transition (through the deterministic crash hook), leaves a runner that the next owner of the home finds
// by its identity and ends. The signals go through a wrapper that only lets a signal reach the one pid this test started.
import '../home.mjs';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { createLocalLlmHost } from '../../lib/local-llm-host/host.mjs';
import { readRecords } from '../../lib/local-llm-host/launch.mjs';
import { identify, listPids, listensOn, readCmdline, readStat } from '../../lib/local-llm-host/procfs.mjs';
import { CRASH_ENV } from '../../lib/local-llm-host/starter.mjs';
import { freePort, scratch, until } from './fixtures/engine.mjs';

const OWNER_SIM = fileURLToPath(new URL('./fixtures/owner-sim.mjs', import.meta.url));
const skip = process.platform !== 'linux' && 'needs a real /proc (Linux only); the identity and signal logic runs on a fake /proc in reaper.test.mjs and host.test.mjs';

const alive = (pid) => { try { process.kill(pid, 0); return true; } catch { return false; } };

function startOwner(t, { home, ownerPort, crashAt = null }) {
    const env = { ...process.env, ...(crashAt ? { [CRASH_ENV]: crashAt } : {}) };
    const child = spawn(process.execPath, [OWNER_SIM, JSON.stringify({ home, ownerPort })], { env, stdio: ['ignore', 'pipe', 'pipe'] });
    t.after(() => { try { child.kill('SIGKILL'); } catch {} });
    let out = '';
    child.stdout.on('data', (chunk) => { out += chunk; });
    child.stderr.on('data', () => {});
    const exited = new Promise((resolve) => child.once('exit', (code, signal) => resolve({ code, signal })));
    return { child, exited, output: () => out };
}

/** The processes of this machine whose command line has `socketPath` as its --host: the runners this test may end. */
const runnersOn = (socketPath) => listPids('/proc').filter((pid) => listensOn(readCmdline('/proc', pid), socketPath));

/** A kill that only reaches pids in `allowed` (the runner this test started), and records every call. */
function guardedKill(allowed) {
    const calls = [];
    const kill = (pid, signal) => {
        calls.push([pid, signal]);
        if (!allowed.has(Math.abs(pid))) throw new Error(`the test refuses to signal pid ${pid}`);
        return process.kill(pid, signal);
    };
    kill.calls = calls;
    return kill;
}

test('real /proc: the stat start time and the command line identify the runner a launch record names', { skip }, async (t) => {
    const home = scratch(t);
    const ownerPort = await freePort();
    const owner = startOwner(t, { home, ownerPort });
    await until(() => /READY \d+/.test(owner.output()), 15_000);
    const runnerPid = Number(/READY (\d+)/.exec(owner.output())[1]);
    const [{ record }] = readRecords(path.join(home, 'local-llm', 'state', 'launch'));
    assert.equal(record.pid, runnerPid);
    assert.match(record.startTime, /^\d+$/);
    assert.equal(readStat('/proc', runnerPid).startTime, record.startTime);
    assert.deepEqual(identify('/proc', runnerPid, record.socketPath, record.startTime), { pid: runnerPid, startTime: record.startTime, pgrp: runnerPid });
    assert.equal(identify('/proc', runnerPid, record.socketPath, String(Number(record.startTime) + 1)), null);
    assert.equal(identify('/proc', process.pid, record.socketPath), null);
    owner.child.kill('SIGKILL');
    await owner.exited;
    const host = createLocalLlmHost({ config: {}, home, env: {}, platformName: 'linux', ownerPort, killImpl: guardedKill(new Set([runnerPid])) });
    t.after(() => host.drain());
    await host.acquire();
    await until(() => !alive(runnerPid));
});

test('real /proc: kill -9 of the owner at the ready state leaves a runner that the next owner reaps at acquire', { skip }, async (t) => {
    const home = scratch(t);
    const ownerPort = await freePort();
    const owner = startOwner(t, { home, ownerPort });
    await until(() => /READY \d+/.test(owner.output()), 15_000);
    const runnerPid = Number(/READY (\d+)/.exec(owner.output())[1]);
    const launchDir = path.join(home, 'local-llm', 'state', 'launch');
    const [{ record }] = readRecords(launchDir);
    owner.child.kill('SIGKILL');
    assert.equal((await owner.exited).signal, 'SIGKILL');
    assert.equal(alive(runnerPid), true, 'the runner outlived its owner');
    const kill = guardedKill(new Set([runnerPid]));
    const host = createLocalLlmHost({ config: {}, home, env: {}, platformName: 'linux', ownerPort, killImpl: kill });
    t.after(() => host.drain());
    const result = await host.acquire();
    assert.deepEqual(result.reaped.reaped, [record.startId]);
    assert.deepEqual(kill.calls[0], [-runnerPid, 'SIGTERM']);
    await until(() => !alive(runnerPid));
    assert.equal(fs.existsSync(record.socketPath), false);
    assert.deepEqual(readRecords(launchDir), []);
});

test('real /proc: kill -9 at the launch-record transition leaves no unidentifiable child (the hook stops between the spawn and the record)', { skip }, async (t) => {
    const home = scratch(t);
    const ownerPort = await freePort();
    const owner = startOwner(t, { home, ownerPort, crashAt: 'after-spawn' });
    assert.equal((await owner.exited).signal, 'SIGKILL', 'the crash hook killed the process');
    const launchDir = path.join(home, 'local-llm', 'state', 'launch');
    const [{ record }] = readRecords(launchDir);
    assert.equal(record.phase, 'intent', 'only the intent was recorded');
    // The runner exists, started right before the crash; it is found by its socket path alone.
    let found = [];
    await until(() => { found = runnersOn(record.socketPath); return found.length === 1; }, 10_000);
    const [runnerPid] = found;
    t.after(() => { try { process.kill(runnerPid, 'SIGKILL'); } catch {} });
    const kill = guardedKill(new Set([runnerPid]));
    const host = createLocalLlmHost({ config: {}, home, env: {}, platformName: 'linux', ownerPort, killImpl: kill });
    t.after(() => host.drain());
    const result = await host.acquire();
    assert.deepEqual(result.reaped.reaped, [record.startId]);
    await until(() => !alive(runnerPid));
    assert.deepEqual(readRecords(launchDir), []);
});

test('real /proc: a crash before the spawn (after the intent) leaves a record that names no process, and the next owner just clears it', { skip }, async (t) => {
    const home = scratch(t);
    const ownerPort = await freePort();
    const owner = startOwner(t, { home, ownerPort, crashAt: 'after-intent' });
    assert.equal((await owner.exited).signal, 'SIGKILL');
    const launchDir = path.join(home, 'local-llm', 'state', 'launch');
    const [{ record }] = readRecords(launchDir);
    assert.equal(record.phase, 'intent');
    assert.deepEqual(runnersOn(record.socketPath), []);
    const kill = guardedKill(new Set());
    const host = createLocalLlmHost({ config: {}, home, env: {}, platformName: 'linux', ownerPort, killImpl: kill });
    t.after(() => host.drain());
    const result = await host.acquire();
    assert.deepEqual(result.reaped.reaped, [record.startId]);
    assert.deepEqual(kill.calls, []);
    assert.deepEqual(readRecords(launchDir), []);
});
