// The macOS process table (lib/local-llm-host/darwinProcs.mjs) and the reaper and launch recorder on top of it, against a fake
// `ps` (an in-memory process list behind the `exec` seam) and a recording kill, so no test runs `ps` or signals a process and the
// file runs on Linux and macOS alike. The real `ps` is exercised by the probe script outside the test globs.
import '../home.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { createDarwinProcs, parsePsRows } from '../../lib/local-llm-host/darwinProcs.mjs';
import { LAUNCH_SCHEMA, createLaunchRecorder, readRecords } from '../../lib/local-llm-host/launch.mjs';
import { listensOn } from '../../lib/local-llm-host/procfs.mjs';
import { reapOrphans } from '../../lib/local-llm-host/reaper.mjs';

// `Fri Oct 16 03:42:14 2026` UTC, in epoch seconds (written out, not computed by the parser).
const T0 = 'Fri Oct 16 03:42:14 2026';
const T0_EPOCH = '1792122134';
const T1 = 'Fri Oct 16 03:42:15 2026';
const T1_EPOCH = '1792122135';

const row = (pid, pgid, state, start, command) => `${String(pid).padStart(5)} ${String(pgid).padStart(5)} ${state.padEnd(4)} ${start}     ${command}\n`;
const runnerCommand = (socketPath) => `/run/llama.cpp/b11295/llama-server -m /m/q.gguf --host ${socketPath} --api-key k --alias q`;

/** A `ps` over an in-memory table of { pgid, state, start, command }; `exec` is its stand-in for execFileSync. */
function fakePs(initial = {}) {
    const table = new Map(Object.entries(initial).map(([pid, spec]) => [Number(pid), { pgid: Number(pid), state: 'S', start: T0, ...spec }]));
    const calls = [];
    let garbage = null;
    const exec = (file, args, options) => {
        calls.push({ file, args, options });
        if (garbage !== null) return garbage;
        const wanted = args.includes('-p') ? [Number(args.at(-1))] : [...table.keys()];
        const text = wanted.filter((pid) => table.has(pid)).map((pid) => { const p = table.get(pid); return row(pid, p.pgid, p.state, p.start, p.command); }).join('');
        // `ps -p` for a pid that does not exist prints nothing and exits 1.
        if (!text && args.includes('-p')) throw Object.assign(new Error('Command failed: ps'), { status: 1, stdout: '' });
        return text;
    };
    return { table, calls, exec, say: (text) => { garbage = text; } };
}

/** A kill that records its calls; a SIGTERM or SIGKILL to -pgid removes the whole group, to a pid just that process. */
function groupKill(ps, { survive = false } = {}) {
    const calls = [];
    const kill = (pid, signal) => {
        calls.push([pid, signal]);
        if (survive) return;
        for (const [other, spec] of [...ps.table]) {
            if (pid < 0 ? spec.pgid === -pid : other === pid) ps.table.delete(other);
        }
    };
    kill.calls = calls;
    return kill;
}

function world(t) {
    const root = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'dp-'));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    const dirs = { root, launchDir: path.join(root, 'launch'), socketDir: path.join(root, 'r') };
    for (const dir of [dirs.launchDir]) fs.mkdirSync(dir, { recursive: true });
    fs.mkdirSync(dirs.socketDir, { recursive: true, mode: 0o700 });
    const logs = [];
    return { ...dirs, logs, log: (line) => logs.push(line) };
}

const socketOf = (w, startId) => path.join(w.socketDir, `${startId}.sock`);
const writeRecord = (w, startId, body) => fs.writeFileSync(path.join(w.launchDir, `${startId}.json`), `${JSON.stringify({ schema: LAUNCH_SCHEMA, startId, socketPath: socketOf(w, startId), ...body })}\n`);
const reap = (w, ps, extra = {}) => reapOrphans({
    launchDir: w.launchDir, socketDir: w.socketDir, platformName: 'darwin', procs: createDarwinProcs({ exec: ps.exec }), log: w.log,
    graceMs: 60, killWaitMs: 60, pollMs: 5, selfPid: 1, ...extra,
});

// ------------------------------------------------------------------ parsing

test('ps rows: pid, group, the first letter of the state, the start as UTC epoch seconds in digits, and the whole command', () => {
    const rows = parsePsRows(`${row(4100, 4100, 'Ss', T0, '/bin/x --host /r/aaaaaaaa.sock')}${row(7, 1, 'Z', 'Fri Oct  6 03:42:14 2026', '(sleep)')}${row(30811, 30811, 'SNs', T1, '/bin/zsh -c a  b')}`);
    assert.deepEqual(rows, [
        { pid: 4100, stat: { state: 'S', pgrp: 4100, startTime: T0_EPOCH }, command: '/bin/x --host /r/aaaaaaaa.sock' },
        { pid: 7, stat: { state: 'Z', pgrp: 1, startTime: '1791258134' }, command: '(sleep)' },
        { pid: 30811, stat: { state: 'S', pgrp: 30811, startTime: T1_EPOCH }, command: '/bin/zsh -c a  b' },
    ]);
    assert.match(rows[0].stat.startTime, /^\d+$/, 'what validateRecord accepts as a start time');
});

test('ps rows: a line that is not a row continues the command before it; garbage, a first line that is not a row, and an impossible start are refused', () => {
    const rows = parsePsRows(`${row(10, 10, 'S', T0, 'sh -c "echo a')}b"\n${row(11, 10, 'S', T0, 'next')}`);
    assert.equal(rows.length, 2);
    assert.equal(rows[0].command, 'sh -c "echo a\nb"');
    assert.equal(parsePsRows('this is not ps output\n'), null);
    assert.equal(parsePsRows('ps: illegal option -- Z\nusage: ps [-AaCcEefhjlMmrSTvwXx]\n'), null);
    assert.equal(parsePsRows(row(12, 12, 'S', 'Fri Feb 30 03:42:14 2026', 'x')), null, 'an unparsable lstart');
    assert.equal(parsePsRows(row(12, 12, 'S', 'Fri Foo 16 03:42:14 2026', 'x')), null, 'an unknown month');
    assert.deepEqual(parsePsRows(''), []);
    assert.deepEqual(parsePsRows(null), []);
});

test('the command is matched as text: --host <socket> followed by a space or the end, and nothing looser', () => {
    const sock = '/h/r/abcd1234.sock';
    assert.equal(listensOn(`llama-server -m /m/q.gguf --host ${sock}`, sock), true, 'at the end');
    assert.equal(listensOn(`llama-server --host ${sock} --api-key k`, sock), true, 'before another argument');
    assert.equal(listensOn(`llama-server --host ${sock}2 --api-key k`, sock), false, 'a near-miss path');
    assert.equal(listensOn(`llama-server --host ${sock}2`, sock), false, 'a near-miss path at the end');
    assert.equal(listensOn(`llama-server --host ${sock}/x`, sock), false);
    assert.equal(listensOn(`llama-server --hostname ${sock}`, sock), false);
    assert.equal(listensOn(`llama-server --host=${sock}`, sock), false);
    assert.equal(listensOn(`tail -f ${sock}`, sock), false, 'a program that merely names the socket');
    assert.equal(listensOn(`llama-server --host ${sock}2 --host ${sock}`, sock), true, 'a later exact match still counts');
    // A home with spaces is matched exactly, because the whole path is in the needle.
    const spaced = '/Users/some one/pw home/r/abcd1234.sock';
    assert.equal(listensOn(`/x/llama-server -m /a b/q.gguf --host ${spaced} --api-key k`, spaced), true);
    assert.equal(listensOn(`/x/llama-server --host ${spaced}2 --api-key k`, spaced), false);
    assert.equal(listensOn(`/x/llama-server --host /Users/some one/pw`, spaced), false);
    // More than 4096 characters, with the socket at the very end.
    const long = `llama-server ${'-x '.repeat(2500)}--host ${sock}`;
    assert.ok(long.length > 7000);
    assert.equal(listensOn(long, sock), true);
    assert.equal(listensOn(`${long}2`, sock), false);
    // The argv of a /proc process is still matched as two arguments.
    assert.equal(listensOn(['llama-server', '--host', sock], sock), true);
    assert.equal(listensOn(['llama-server', `--host ${sock}`], sock), false);
});

// ------------------------------------------------------------------ the table behind exec

test('the table runs /bin/ps -ww by absolute path with a fixed environment, and reads one pid or all of them', () => {
    const ps = fakePs({ 4100: { command: 'a --host /r/aaaaaaaa.sock' }, 4200: { command: 'b', pgid: 4100, start: T1 } });
    const procs = createDarwinProcs({ exec: ps.exec });
    assert.deepEqual(procs.listPids(), [4100, 4200]);
    assert.deepEqual(procs.readStat(4200), { state: 'S', pgrp: 4100, startTime: T1_EPOCH });
    assert.equal(procs.readCmdline(4100), 'a --host /r/aaaaaaaa.sock');
    assert.equal(procs.readStat(99), null, 'no such process');
    assert.equal(procs.readCmdline(99), null);
    assert.equal(procs.readStat(0), null);
    assert.equal(procs.readStat('4100'), null, 'a pid is a number');
    assert.deepEqual(procs.listAll().map((item) => item.pid), [4100, 4200]);
    const [list, single] = [ps.calls.find((call) => call.args.includes('-A')), ps.calls.find((call) => call.args.includes('-p'))];
    assert.deepEqual(list.args, ['-ww', '-A', '-o', 'pid=,pgid=,state=,lstart=,command=']);
    assert.deepEqual(single.args.slice(0, 4), ['-ww', '-o', 'pid=,pgid=,state=,lstart=,command=', '-p']);
    for (const call of ps.calls) {
        assert.equal(call.file, '/bin/ps');
        assert.deepEqual(call.options.env, { PATH: '/usr/bin:/bin', LC_ALL: 'C', LANG: 'C', TZ: 'UTC' });
        assert.ok(call.args.includes('-ww'), 'no truncation of the command');
    }
});

test('a listing that cannot be understood throws; a single read that cannot be understood is null', () => {
    const ps = fakePs({ 1: { command: 'launchd' } });
    const procs = createDarwinProcs({ exec: ps.exec });
    for (const text of ['', 'garbage\n', 'ps: error\n']) {
        ps.say(text);
        assert.throws(() => procs.listAll(), /did not list the processes/, JSON.stringify(text));
        assert.throws(() => procs.listPids(), /did not list the processes/);
        assert.equal(procs.readStat(1), null);
        assert.equal(procs.readCmdline(1), null);
    }
    // ps that cannot run at all.
    const broken = createDarwinProcs({ exec: () => { throw Object.assign(new Error('spawn ENOENT'), { code: 'ENOENT' }); } });
    assert.throws(() => broken.listAll(), /did not list/);
    assert.equal(broken.readStat(5), null);
    // A row for another pid than the one asked, or an unparsable start, is not an answer.
    ps.say(row(2, 2, 'S', T0, 'other'));
    assert.equal(procs.readStat(1), null);
    ps.say(row(1, 1, 'S', 'Fri Feb 30 03:42:14 2026', 'launchd'));
    assert.equal(procs.readStat(1), null);
});

// ------------------------------------------------------------------ the reaper on the table

test('a record whose pid still has the recorded start and the socket as its --host: the group is sent SIGTERM, then the record and socket go', async (t) => {
    const w = world(t);
    const sock = socketOf(w, 'aaaaaaaa');
    const ps = fakePs({ 4100: { command: runnerCommand(sock) }, 4101: { pgid: 4100, command: 'helper' }, 5000: { command: 'unrelated' } });
    writeRecord(w, 'aaaaaaaa', { phase: 'spawned', pid: 4100, startTime: T0_EPOCH });
    const kill = groupKill(ps);
    const result = await reap(w, ps, { killImpl: kill });
    assert.deepEqual(kill.calls, [[-4100, 'SIGTERM']]);
    assert.deepEqual(result.reaped, ['aaaaaaaa']);
    assert.deepEqual([...ps.table.keys()], [5000], 'only the runner and its group went');
    assert.deepEqual(readRecords(w.launchDir), []);
    assert.ok(w.logs.some((line) => /sent SIGTERM to the orphaned runner aaaaaaaa \(pid 4100\)/.test(line)));
});

test('a recorded pid with another start time is not taken for the runner by its pid (only the socket name can find one), and an unrelated process on it is left alone', async (t) => {
    const w = world(t);
    const sock = socketOf(w, 'bbbbbbbb');
    // The same pid and even the same command line, one second later: the process that was recorded is gone.
    const ps = fakePs({ 4200: { start: T1, command: runnerCommand(sock) } });
    writeRecord(w, 'bbbbbbbb', { phase: 'spawned', pid: 4200, startTime: T0_EPOCH });
    const kill = groupKill(ps);
    // The recorded pid is not signalled; the process that names the socket as its --host is found by that name, with its own start time.
    await reap(w, ps, { killImpl: kill });
    assert.deepEqual(kill.calls, [[-4200, 'SIGTERM']], 'found by the socket, not by the pid');
    assert.ok(w.logs.some((line) => /pid 4200 of launch record bbbbbbbb is not that runner any more/.test(line)));
    // An unrelated process on the recorded pid is left alone.
    const ps2 = fakePs({ 4300: { start: T1, command: '/usr/bin/vim notes.txt' } });
    writeRecord(w, 'cccccccc', { phase: 'spawned', pid: 4300, startTime: T0_EPOCH });
    const kill2 = groupKill(ps2);
    const result = await reap(w, ps2, { killImpl: kill2 });
    assert.deepEqual(kill2.calls, []);
    assert.deepEqual(result.reaped, ['cccccccc']);
    assert.deepEqual([...ps2.table.keys()], [4300]);
});

test('an intent record is matched by the socket in a command line of any length; a near-miss path and a zombie are not', async (t) => {
    const w = world(t);
    const sock = socketOf(w, 'dddddddd');
    writeRecord(w, 'dddddddd', { phase: 'intent' });
    const longCommand = `/run/llama.cpp/b11295/llama-server ${'-x '.repeat(2500)}--host ${sock}`;
    assert.ok(longCommand.length > 4096);
    const ps = fakePs({
        6001: { command: longCommand },
        6002: { command: runnerCommand(`${sock}2`) },
        6003: { command: runnerCommand(sock), state: 'Z' },
        6004: { command: `tail -f ${sock}` },
    });
    const kill = groupKill(ps);
    const result = await reap(w, ps, { killImpl: kill });
    assert.deepEqual(kill.calls, [[-6001, 'SIGTERM']]);
    assert.deepEqual(result.reaped, ['dddddddd']);
    assert.deepEqual([...ps.table.keys()].sort(), [6002, 6003, 6004]);
});

test('a runner that ignores SIGTERM gets SIGKILL for its group; a survivor keeps its record', async (t) => {
    const w = world(t);
    const sock = socketOf(w, 'eeeeeeee');
    const ps = fakePs({ 4400: { command: runnerCommand(sock) }, 4401: { pgid: 4400, command: 'helper' } });
    writeRecord(w, 'eeeeeeee', { phase: 'spawned', pid: 4400, startTime: T0_EPOCH });
    const stubborn = [];
    const kill = (pid, signal) => {
        stubborn.push([pid, signal]);
        if (signal === 'SIGKILL') { ps.table.delete(4400); ps.table.delete(4401); }
    };
    const result = await reap(w, ps, { killImpl: kill });
    assert.deepEqual(stubborn, [[-4400, 'SIGTERM'], [-4400, 'SIGKILL']]);
    assert.deepEqual(result.reaped, ['eeeeeeee']);
    // One that survives both keeps its record and is reported.
    const w2 = world(t);
    const sock2 = socketOf(w2, 'ffffffff');
    const immortal = fakePs({ 4500: { command: runnerCommand(sock2) } });
    writeRecord(w2, 'ffffffff', { phase: 'spawned', pid: 4500, startTime: T0_EPOCH });
    const survived = await reap(w2, immortal, { killImpl: groupKill(immortal, { survive: true }) });
    assert.deepEqual(survived.failed, [{ startId: 'ffffffff', pids: [4500] }]);
    assert.equal(readRecords(w2.launchDir).length, 1);
});

test('members of the group that outlive the leader are ended one by one, each only if its start time is unchanged', async (t) => {
    const w = world(t);
    const sock = socketOf(w, '12121212');
    const ps = fakePs({ 4600: { command: runnerCommand(sock) }, 4601: { pgid: 4600, command: 'worker' }, 4602: { pgid: 4600, command: 'worker2' } });
    writeRecord(w, '12121212', { phase: 'spawned', pid: 4600, startTime: T0_EPOCH });
    const calls = [];
    const kill = (pid, signal) => {
        calls.push([pid, signal]);
        if (signal === 'SIGTERM') {
            ps.table.delete(4600); // the leader exits at once, its group does not
            ps.table.get(4602).start = T1; // and one member's pid is now another process
        }
        if (pid > 0) ps.table.delete(pid);
    };
    await reap(w, ps, { killImpl: kill });
    assert.deepEqual(calls, [[-4600, 'SIGTERM'], [4601, 'SIGKILL']]);
    assert.equal(ps.table.has(4602), true, 'a member whose start time changed is not signalled');
});

test('a ps that cannot be read stops the reaper before any signal and leaves the record', async (t) => {
    const w = world(t);
    const sock = socketOf(w, '13131313');
    const ps = fakePs({ 4700: { command: runnerCommand(sock) } });
    writeRecord(w, '13131313', { phase: 'spawned', pid: 4700, startTime: T0_EPOCH });
    ps.say('ps: cannot run\n');
    const kill = groupKill(ps);
    await assert.rejects(reap(w, ps, { killImpl: kill }), /did not list the processes/);
    assert.deepEqual(kill.calls, []);
    assert.equal(readRecords(w.launchDir).length, 1, 'the record stays for the next start');
});

test('a malformed record is set aside on macOS too, and no process is looked at', async (t) => {
    const w = world(t);
    fs.writeFileSync(path.join(w.launchDir, '14141414.json'), '{"schema":1,"startId":"14141414"');
    const ps = fakePs({ 1: { command: 'launchd' } });
    const kill = groupKill(ps);
    const result = await reap(w, ps, { killImpl: kill });
    assert.deepEqual(kill.calls, []);
    assert.equal(ps.calls.length, 0);
    assert.equal(result.skipped.length, 1);
    assert.equal(fs.existsSync(path.join(w.launchDir, '14141414.json.malformed')), true);
});

// ------------------------------------------------------------------ the recorder

test('on macOS the recorder keeps the start time ps reports as epoch seconds, and null when the process cannot be read', (t) => {
    const w = world(t);
    const ps = fakePs({ 4800: { start: T1, command: 'x' } });
    const recorder = createLaunchRecorder({ dir: w.launchDir, platformName: 'darwin', procs: createDarwinProcs({ exec: ps.exec }) });
    recorder.spawned({ startId: '15151515', socketPath: socketOf(w, '15151515'), pid: 4800 });
    recorder.spawned({ startId: '15151516', socketPath: socketOf(w, '15151516'), pid: 4801 });
    const byId = Object.fromEntries(readRecords(w.launchDir).map(({ record }) => [record.startId, record]));
    assert.equal(byId['15151515'].startTime, T1_EPOCH);
    assert.equal(byId['15151516'].startTime, null, 'a pid ps does not know');
    // The record the Mac writes is the record its reaper trusts.
    assert.equal(byId['15151515'].pid, 4800);
});
