// The launch records and the reaper (plan: Ownership and recovery (2)-(4)), against a fake /proc and a recording kill, so no
// test signals a real process: a record is only acted on after the identity (start time and `--host <socket>`) is read again,
// a reused pid is never signalled, and a malformed record fails safe.
import '../home.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { LAUNCH_SCHEMA, createLaunchRecorder, readRecords, removeSocketFile, validateRecord, writeDurable } from '../../lib/local-llm-host/launch.mjs';
import { identify, listensOn, readCmdline, readStat } from '../../lib/local-llm-host/procfs.mjs';
import { reapOrphans } from '../../lib/local-llm-host/reaper.mjs';
import { recordingKill, removeProc, runnerArgv, statLine, writeProc } from './fixtures/proc.mjs';

function world(t) {
    const root = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'reap-'));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    const dirs = { root, procDir: path.join(root, 'proc'), launchDir: path.join(root, 'launch'), socketDir: path.join(root, 'r') };
    for (const dir of [dirs.procDir, dirs.launchDir]) fs.mkdirSync(dir, { recursive: true });
    fs.mkdirSync(dirs.socketDir, { recursive: true, mode: 0o700 });
    const logs = [];
    return { ...dirs, logs, log: (line) => logs.push(line) };
}

const socketOf = (w, startId) => path.join(w.socketDir, `${startId}.sock`);

function writeRecord(w, startId, body) {
    fs.writeFileSync(path.join(w.launchDir, `${startId}.json`), `${JSON.stringify({ schema: LAUNCH_SCHEMA, startId, socketPath: socketOf(w, startId), ...body })}\n`);
}

/** A real socket file at `file`, kept open until the test ends. */
async function realSocketFile(t, file) {
    const server = net.createServer();
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(file, resolve); });
    t.after(() => new Promise((resolve) => server.close(resolve)));
}

const reap = (w, extra = {}) => reapOrphans({
    launchDir: w.launchDir, socketDir: w.socketDir, procDir: w.procDir, platformName: 'linux', log: w.log, graceMs: 60, killWaitMs: 60, pollMs: 5, selfPid: 1, ...extra,
});

test('stat parsing counts from the last parenthesis: a command name with spaces and parentheses cannot shift the start time', (t) => {
    const w = world(t);
    writeProc(w.procDir, { pid: 700, comm: 'evil) (S 1 2 3', startTime: '987654321', pgrp: 700, argv: ['x'] });
    assert.deepEqual(readStat(w.procDir, 700), { state: 'S', pgrp: 700, startTime: '987654321' });
    assert.equal(readStat(w.procDir, 701), null, 'a missing process');
    fs.mkdirSync(path.join(w.procDir, '702'));
    fs.writeFileSync(path.join(w.procDir, '702', 'stat'), '702 (short) S 1 2\n');
    assert.equal(readStat(w.procDir, 702), null, 'a stat line with too few fields');
    fs.writeFileSync(path.join(w.procDir, '702', 'stat'), 'no parenthesis at all');
    assert.equal(readStat(w.procDir, 702), null);
    assert.deepEqual(readCmdline(w.procDir, 700), ['x']);
});

test('the socket is matched as the whole value of --host, not as a substring or a stray argument', (t) => {
    const sock = '/run/r/aaaaaaaa.sock';
    assert.equal(listensOn(['llama-server', '--host', sock], sock), true);
    assert.equal(listensOn(['llama-server', '--host', `${sock}.x`], sock), false);
    assert.equal(listensOn(['tail', '-f', sock], sock), false, 'a program that merely names the socket is not the runner');
    assert.equal(listensOn(['llama-server', `--host=${sock}`], sock), false);
    assert.equal(listensOn(null, sock), false);
    const w = world(t);
    writeProc(w.procDir, { pid: 800, argv: runnerArgv(sock), state: 'Z' });
    assert.equal(identify(w.procDir, 800, sock), null, 'a zombie is gone');
});

test('a matching record: the group is sent SIGTERM and, once the runner is gone, its socket and record are removed', async (t) => {
    const w = world(t);
    const sock = socketOf(w, 'aaaaaaaa');
    await realSocketFile(t, sock);
    writeProc(w.procDir, { pid: 4100, startTime: '5000', argv: runnerArgv(sock) });
    writeRecord(w, 'aaaaaaaa', { phase: 'spawned', pid: 4100, startTime: '5000' });
    const kill = recordingKill({ dies: { SIGTERM: (pid) => removeProc(w.procDir, -pid) } });
    const result = await reap(w, { killImpl: kill });
    assert.deepEqual(kill.calls, [[-4100, 'SIGTERM']]);
    assert.deepEqual(result, { reaped: ['aaaaaaaa'], skipped: [], failed: [] });
    assert.equal(fs.existsSync(sock), false, 'the socket file was removed');
    assert.equal(fs.existsSync(path.join(w.launchDir, 'aaaaaaaa.json')), false, 'the record was removed');
    assert.ok(w.logs.some((line) => /sent SIGTERM to the orphaned runner aaaaaaaa \(pid 4100\)/.test(line)), w.logs.join('\n'));
});

test('a runner that is gone before the first signal is logged as already gone: no signal is sent and no line claims it was ended', async (t) => {
    const w = world(t);
    const sock = socketOf(w, 'a1a1a1a1');
    writeProc(w.procDir, { pid: 4150, startTime: '5050', argv: runnerArgv(sock) });
    writeRecord(w, 'a1a1a1a1', { phase: 'spawned', pid: 4150, startTime: '5050' });
    // The runner disappears after the reaper found it (its first /proc listing) and before it signals (the group listing that follows).
    let listings = 0;
    const fsApi = Object.create(fs, { readdirSync: { value: (...args) => {
        const out = fs.readdirSync(...args);
        if (String(args[0]) === w.procDir && ++listings === 2) removeProc(w.procDir, 4150);
        return out;
    } } });
    const kill = recordingKill();
    const result = await reap(w, { killImpl: kill, fsApi });
    assert.ok(listings >= 2, 'the hook fired');
    assert.deepEqual(kill.calls, [], 'nothing was signalled');
    assert.deepEqual(result.reaped, ['a1a1a1a1']);
    assert.ok(w.logs.some((line) => /orphaned runner a1a1a1a1 \(pid 4150\) was already gone; no signal was sent/.test(line)), w.logs.join('\n'));
    assert.ok(w.logs.every((line) => !/sent SIG|ended the orphaned/.test(line)), w.logs.join('\n'));
});

test('a runner that ignores SIGTERM gets SIGKILL for its group after the grace period, and a survivor keeps its record', async (t) => {
    const w = world(t);
    const sock = socketOf(w, 'bbbbbbbb');
    writeProc(w.procDir, { pid: 4200, startTime: '5100', argv: runnerArgv(sock) });
    writeRecord(w, 'bbbbbbbb', { phase: 'spawned', pid: 4200, startTime: '5100' });
    const kill = recordingKill({ dies: { SIGKILL: (pid) => removeProc(w.procDir, -pid) } });
    const result = await reap(w, { killImpl: kill });
    assert.deepEqual(kill.calls, [[-4200, 'SIGTERM'], [-4200, 'SIGKILL']]);
    assert.deepEqual(result.reaped, ['bbbbbbbb']);

    writeProc(w.procDir, { pid: 4201, startTime: '5101', argv: runnerArgv(socketOf(w, 'cccccccc')) });
    writeRecord(w, 'cccccccc', { phase: 'spawned', pid: 4201, startTime: '5101' });
    const stubborn = recordingKill();
    const failed = await reap(w, { killImpl: stubborn });
    assert.deepEqual(stubborn.calls, [[-4201, 'SIGTERM'], [-4201, 'SIGKILL']]);
    assert.deepEqual(failed.failed, [{ startId: 'cccccccc', pids: [4201] }]);
    assert.equal(fs.existsSync(path.join(w.launchDir, 'cccccccc.json')), true, 'the record of a survivor is kept');
    assert.ok(w.logs.some((line) => /survived SIGKILL/.test(line)));
});

test('a reused pid is never signalled: the start time or the command line is not the recorded runner\'s', async (t) => {
    const w = world(t);
    const sockA = socketOf(w, 'dddddddd');
    const sockB = socketOf(w, 'eeeeeeee');
    // 4300 now runs something else (another start time and command); 4301 has the right command but another start time.
    writeProc(w.procDir, { pid: 4300, startTime: '9999', argv: ['/usr/bin/sleep', '1000'] });
    writeRecord(w, 'dddddddd', { phase: 'spawned', pid: 4300, startTime: '5200' });
    writeProc(w.procDir, { pid: 4301, startTime: '9998', argv: ['/usr/bin/vim', 'notes'] });
    writeRecord(w, 'eeeeeeee', { phase: 'spawned', pid: 4301, startTime: '5201' });
    const kill = recordingKill();
    const result = await reap(w, { killImpl: kill });
    assert.deepEqual(kill.calls, [], 'nothing was signalled');
    assert.deepEqual(result.reaped.sort(), ['dddddddd', 'eeeeeeee'], 'both records named no live runner and were cleared');
    assert.ok(w.logs.some((line) => /pid 4300 .* not that runner any more/.test(line)));
    assert.equal(sockA === sockB, false);
});

test('the identity is read again right before each signal: a runner replaced between TERM and KILL is not killed', async (t) => {
    const w = world(t);
    const sock = socketOf(w, 'ffffffff');
    writeProc(w.procDir, { pid: 4400, startTime: '5300', argv: runnerArgv(sock) });
    writeRecord(w, 'ffffffff', { phase: 'spawned', pid: 4400, startTime: '5300' });
    // The runner exits on TERM and the pid is taken by an unrelated process before anything else is sent.
    const kill = recordingKill({ dies: { SIGTERM: (pid) => { removeProc(w.procDir, -pid); writeProc(w.procDir, { pid: 4400, startTime: '7777', argv: ['/bin/bash'] }); } } });
    await reap(w, { killImpl: kill });
    assert.deepEqual(kill.calls, [[-4400, 'SIGTERM']]);
});

test('a crash between the spawn and the second write leaves an intent record, and the child is found by its socket path', async (t) => {
    const w = world(t);
    const sock = socketOf(w, '12345678');
    writeRecord(w, '12345678', { phase: 'intent' });
    writeProc(w.procDir, { pid: 4500, startTime: '5400', argv: runnerArgv(sock) });
    writeProc(w.procDir, { pid: 4501, startTime: '5401', argv: ['/usr/bin/tail', '-f', sock] });
    const kill = recordingKill({ dies: { SIGTERM: (pid) => removeProc(w.procDir, -pid) } });
    const result = await reap(w, { killImpl: kill });
    assert.deepEqual(kill.calls, [[-4500, 'SIGTERM']], 'the runner, not the program that merely names its socket');
    assert.deepEqual(result.reaped, ['12345678']);
});

test('the members of a runner\'s group are ended one by one after the leader, each only if it is still the process that was seen', async (t) => {
    const w = world(t);
    const sock = socketOf(w, '0a0a0a0a');
    writeProc(w.procDir, { pid: 4600, startTime: '5500', argv: runnerArgv(sock) });
    writeProc(w.procDir, { pid: 4601, startTime: '5501', pgrp: 4600, argv: ['helper'] });
    writeProc(w.procDir, { pid: 4602, startTime: '5502', pgrp: 4600, argv: ['helper2'] });
    writeProc(w.procDir, { pid: 4603, startTime: '5503', pgrp: 4603, argv: ['unrelated'] });
    writeRecord(w, '0a0a0a0a', { phase: 'spawned', pid: 4600, startTime: '5500' });
    const kill = recordingKill({ dies: {
        SIGTERM: (pid) => { removeProc(w.procDir, -pid); writeProc(w.procDir, { pid: 4602, startTime: '8888', pgrp: 4600, argv: ['replacement'] }); },
        SIGKILL: (pid) => removeProc(w.procDir, pid),
    } });
    await reap(w, { killImpl: kill });
    assert.deepEqual(kill.calls, [[-4600, 'SIGTERM'], [4601, 'SIGKILL']], 'the replaced member is a different process and is left alone');
    assert.equal(fs.existsSync(path.join(w.procDir, '4603')), true);
});

test('a runner that is not its group\'s leader is signalled by pid, not by group', async (t) => {
    const w = world(t);
    const sock = socketOf(w, '0b0b0b0b');
    writeProc(w.procDir, { pid: 4700, startTime: '5600', pgrp: 1234, argv: runnerArgv(sock) });
    writeRecord(w, '0b0b0b0b', { phase: 'spawned', pid: 4700, startTime: '5600' });
    const kill = recordingKill({ dies: { SIGTERM: (pid) => removeProc(w.procDir, pid) } });
    await reap(w, { killImpl: kill });
    assert.deepEqual(kill.calls, [[4700, 'SIGTERM']]);
});

test('malformed and partial records fail safe: nothing is signalled, the reaper does not throw, the record is set aside, the log says why', async (t) => {
    const w = world(t);
    const sock = socketOf(w, '0c0c0c0c');
    // A live runner is there for every one of them to tempt the reaper.
    writeProc(w.procDir, { pid: 4800, startTime: '5700', argv: runnerArgv(sock) });
    const good = { schema: LAUNCH_SCHEMA, startId: '0c0c0c0c', socketPath: sock, phase: 'spawned', pid: 4800, startTime: '5700' };
    const bad = {
        '0c0c0c01': '{"schema":1,"startId":"0c0c0c01","socke',
        '0c0c0c02': 'not json at all',
        '0c0c0c03': JSON.stringify({ ...good, startId: '0c0c0c03', socketPath: socketOf(w, '0c0c0c03'), schema: 2 }),
        '0c0c0c04': JSON.stringify({ ...good, startId: '0c0c0c04', socketPath: socketOf(w, '0c0c0c04'), pid: 1 }),
        '0c0c0c05': JSON.stringify({ ...good, startId: '0c0c0c05', socketPath: socketOf(w, '0c0c0c05'), pid: -4800 }),
        '0c0c0c06': JSON.stringify({ ...good, startId: '0c0c0c06', socketPath: socketOf(w, '0c0c0c06'), pid: '4800' }),
        '0c0c0c07': JSON.stringify({ ...good, startId: '0c0c0c07', socketPath: 'relative/0c0c0c07.sock' }),
        '0c0c0c08': JSON.stringify({ ...good, startId: '0c0c0c08', socketPath: socketOf(w, 'ffffffff') }),
        '0c0c0c09': JSON.stringify({ ...good, startId: 'ZZZZZZZZ' }),
        '0c0c0c0a': JSON.stringify({ ...good, startId: '0c0c0c0a', socketPath: socketOf(w, '0c0c0c0a'), phase: 'running' }),
        '0c0c0c0b': JSON.stringify({ ...good, startId: '0c0c0c0b', socketPath: socketOf(w, '0c0c0c0b'), startTime: 'abc' }),
        '0c0c0c0d': JSON.stringify({ schema: 1, startId: '0c0c0c0d', socketPath: socketOf(w, '0c0c0c0d'), phase: 'intent', pid: 4800 }),
        '0c0c0c0e': '[]',
        '0c0c0c0f': '',
    };
    for (const [id, text] of Object.entries(bad)) fs.writeFileSync(path.join(w.launchDir, `${id}.json`), text);
    // A temporary file of a write that crashed is not a record.
    fs.writeFileSync(path.join(w.launchDir, '.0c0c0c10.json.99.abcd.tmp'), '{"half');
    const kill = recordingKill();
    const result = await reap(w, { killImpl: kill });
    assert.deepEqual(kill.calls, []);
    assert.equal(result.skipped.length, Object.keys(bad).length);
    for (const id of Object.keys(bad)) {
        assert.equal(fs.existsSync(path.join(w.launchDir, `${id}.json`)), false, `${id} was set aside`);
        assert.equal(fs.existsSync(path.join(w.launchDir, `${id}.json.malformed`)), true, `${id} was kept as .malformed`);
    }
    assert.equal(w.logs.filter((line) => /is malformed/.test(line)).length, Object.keys(bad).length);
    assert.equal(fs.existsSync(path.join(w.launchDir, '.0c0c0c10.json.99.abcd.tmp')), true);
    assert.equal(fs.existsSync(path.join(w.procDir, '4800')), true);
});

test('a tampered record cannot make the reaper delete a file that is not one of this host\'s runner sockets', async (t) => {
    const w = world(t);
    const elsewhere = path.join(w.root, 'elsewhere');
    fs.mkdirSync(elsewhere);
    await realSocketFile(t, path.join(elsewhere, '0d0d0d0d.sock'));
    fs.writeFileSync(path.join(w.socketDir, '0d0d0d0e.sock'), 'a plain file named like a socket');
    writeRecord(w, '0d0d0d0d', { phase: 'intent', socketPath: path.join(elsewhere, '0d0d0d0d.sock') });
    writeRecord(w, '0d0d0d0e', { phase: 'intent' });
    const kill = recordingKill();
    await reap(w, { killImpl: kill });
    assert.equal(fs.existsSync(path.join(elsewhere, '0d0d0d0d.sock')), true, 'a socket outside the host\'s directory stays');
    assert.equal(fs.existsSync(path.join(w.socketDir, '0d0d0d0e.sock')), true, 'a regular file stays');
    assert.equal(removeSocketFile(path.join(w.socketDir, 'not-a-start-id.sock'), w.socketDir), false);
    assert.deepEqual(kill.calls, []);
});

test('where the platform has no /proc the reaper does nothing and says so, and leaves every record', async (t) => {
    const w = world(t);
    writeRecord(w, '0e0e0e0e', { phase: 'intent' });
    const kill = recordingKill();
    const result = await reap(w, { killImpl: kill, platformName: 'darwin' });
    assert.deepEqual(kill.calls, []);
    assert.equal(result.reaped.length, 0);
    assert.equal(fs.existsSync(path.join(w.launchDir, '0e0e0e0e.json')), true);
    assert.ok(w.logs.some((line) => /not available on darwin/.test(line)));
});

test('record validation accepts exactly the two phases it writes', () => {
    const base = { schema: LAUNCH_SCHEMA, startId: 'abcdef01', socketPath: '/r/abcdef01.sock' };
    assert.deepEqual(validateRecord({ ...base, phase: 'intent' }), { startId: 'abcdef01', socketPath: '/r/abcdef01.sock', phase: 'intent', pid: null, startTime: null });
    assert.deepEqual(validateRecord({ ...base, phase: 'spawned', pid: 42, startTime: null }).startTime, null);
    assert.throws(() => validateRecord({ ...base, phase: 'spawned', pid: 42 }), /startTime/);
});

test('a record is written durably: temporary file, fsync, rename, fsync of the directory, and never read half-written', (t) => {
    const w = world(t);
    const operations = [];
    const spy = new Proxy(fs, {
        get(target, name) {
            const value = target[name];
            if (!['openSync', 'writeFileSync', 'fsyncSync', 'closeSync', 'renameSync'].includes(name)) return value;
            return (...args) => { operations.push(name === 'openSync' ? `open ${typeof args[0] === 'string' && args[0].endsWith('.tmp') ? 'tmp' : 'dir'}` : name === 'renameSync' ? 'rename' : name.replace('Sync', '')); return value.apply(target, args); };
        },
    });
    const recorder = createLaunchRecorder({ dir: w.launchDir, procDir: w.procDir, fsApi: spy, platformName: 'linux' });
    recorder.intent({ startId: '0f0f0f0f', socketPath: socketOf(w, '0f0f0f0f') });
    assert.deepEqual(operations, ['open tmp', 'writeFile', 'fsync', 'close', 'rename', 'open dir', 'fsync', 'close']);
    // A crash between the write and the rename leaves the old record whole and a temporary file nobody reads.
    const file = path.join(w.launchDir, '0f0f0f0f.json');
    const before = fs.readFileSync(file, 'utf8');
    const crashing = new Proxy(fs, { get: (target, name) => (name === 'renameSync' ? () => { throw new Error('crash'); } : target[name]) });
    assert.throws(() => writeDurable(file, '{"schema":1,"half', { fsApi: crashing }), /crash/);
    assert.equal(fs.readFileSync(file, 'utf8'), before);
    assert.deepEqual(fs.readdirSync(w.launchDir).filter((name) => name.endsWith('.tmp')), [], 'the temporary file of a failed rename is removed');
    assert.equal(readRecords(w.launchDir).length, 1);
});

test('the recorder keeps the pid and start time it reads from /proc, and null where there is no /proc', (t) => {
    const w = world(t);
    writeProc(w.procDir, { pid: 4900, startTime: '6000', argv: ['x'] });
    const recorder = createLaunchRecorder({ dir: w.launchDir, procDir: w.procDir, platformName: 'linux' });
    recorder.spawned({ startId: '10101010', socketPath: socketOf(w, '10101010'), pid: 4900 });
    assert.deepEqual(readRecords(w.launchDir)[0].record, { startId: '10101010', socketPath: socketOf(w, '10101010'), phase: 'spawned', pid: 4900, startTime: '6000' });
    const mac = createLaunchRecorder({ dir: w.launchDir, procDir: w.procDir, platformName: 'darwin' });
    mac.spawned({ startId: '10101011', socketPath: socketOf(w, '10101011'), pid: 4900 });
    assert.equal(readRecords(w.launchDir).find((entry) => entry.record.startId === '10101011').record.startTime, null);
    assert.throws(() => recorder.intent({ startId: '../../x', socketPath: '/x' }), /Invalid start id/);
});
