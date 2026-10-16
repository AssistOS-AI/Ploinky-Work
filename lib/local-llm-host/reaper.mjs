// The reaper: finds the runners an earlier pworker of this home left behind and ends them. It runs only in a process that
// holds the home's ownership (host.acquire), so it can never end the live owner's runner.
//
// For every launch record (launch.mjs) it looks for the runner by identity, never by pid alone:
//   - the recorded pid, when its process table entry (Linux: `/proc/<pid>/stat`) still has the recorded start time and its
//     command line (`/proc/<pid>/cmdline`) has `--host <recorded socket path>` as two arguments;
//   - and any other process whose cmdline names that socket path as its `--host` (a crash between the spawn and the
//     second write leaves only the intent record, and a record without a usable start time cannot say more).
// The identity is read again right before every signal, so a pid the system gave to something else in the meantime is
// never signalled. A group leader gets SIGTERM for its group, then SIGKILL for the group if it stays; the members of the
// group that were seen before the first signal are ended one by one after the leader, each re-checked by start time.
// A malformed record fails safe: nothing is signalled, the file is renamed `<name>.malformed`, and the log says why.
// A process the table cannot read (macOS: `ps` failed; Linux never raises this) is unknown, not gone: the record and the socket
// are kept, no further signal is sent for that record, and the log says so. Only a process the table reports as absent is gone.
// The process table is one small interface (`listPids()`, `readStat(pid)`, `readCmdline(pid)`, and for a table that can list
// everything in one read, `listAll()`): `/proc` on Linux (procfs.mjs), `ps` on macOS (darwinProcs.mjs, where the command is one
// string and the match is on its text). Everything above is shared. On another platform nothing is reaped and the log says so.

import fs from 'node:fs';
import path from 'node:path';

import { ProcessUnreadable, createDarwinProcs } from './darwinProcs.mjs';
import { createProcfsProcs, identifyIn, listensOn } from './procfs.mjs';
import { readRecords, removeSocketFile } from './launch.mjs';

const sleepFor = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Ends the orphaned runners recorded under `launchDir`. Returns `{ reaped, skipped, failed }`: the start ids of runners that
 * were ended or found gone, the records that were left alone (with a reason), and the runners that survived SIGKILL.
 * `killImpl` sends the signals (`process.kill` by default; a test passes a recording fake), `sleep` waits. `procs` is the process
 * table (the platform's own by default: `procDir` on Linux, `ps` on macOS).
 */
export async function reapOrphans({
    launchDir, socketDir, procDir = '/proc', platformName = process.platform, fsApi = fs, killImpl = process.kill,
    sleep = sleepFor, log = () => {}, graceMs = 10_000, killWaitMs = 5_000, pollMs = 50, selfPid = process.pid, procs = null,
} = {}) {
    const result = { reaped: [], skipped: [], failed: [] };
    if (platformName !== 'linux' && platformName !== 'darwin') {
        result.skipped.push({ reason: `orphan recovery is not available on ${platformName}` });
        log(`local-llm: orphan recovery is not available on ${platformName}; runners an earlier pworker left behind are not reaped`);
        return result;
    }
    const table = procs ?? (platformName === 'linux' ? createProcfsProcs(procDir, { fsApi }) : createDarwinProcs());
    for (const entry of readRecords(launchDir, { fsApi })) {
        if (entry.error) {
            const quarantined = `${entry.file}.malformed`;
            try { fsApi.renameSync(entry.file, quarantined); } catch { /* the file stays; the log line is what matters */ }
            result.skipped.push({ file: entry.file, reason: `malformed: ${entry.error}` });
            log(`local-llm: launch record ${entry.file} is malformed (${entry.error}); no process was signalled and the record was set aside as ${path.basename(quarantined)}`);
            continue;
        }
        const { record } = entry;
        try {
            const candidates = find(record);
            const survivors = [];
            const outcomes = [];
            for (const candidate of candidates) {
                const outcome = await end(candidate, record);
                outcomes.push({ pid: candidate.pid, signals: outcome.signals });
                if (!outcome.gone) survivors.push(candidate.pid);
            }
            if (survivors.length) {
                result.failed.push({ startId: record.startId, pids: survivors });
                log(`local-llm: runner ${record.startId} (pid ${survivors.join(', ')}) survived SIGKILL; its launch record is kept`);
                continue;
            }
            removeSocketFile(record.socketPath, socketDir, { fsApi });
            try { fsApi.unlinkSync(entry.file); } catch (error) { if (error?.code !== 'ENOENT') log(`local-llm: launch record ${entry.file} could not be removed (${error.code ?? error.message})`); }
            result.reaped.push(record.startId);
            if (!candidates.length) log(`local-llm: launch record ${record.startId} named no live runner; removed`);
            // Whether a signal was actually sent is what the line says: a runner that was gone before the signal was not ended by this reaper.
            for (const outcome of outcomes) {
                log(outcome.signals.length
                    ? `local-llm: sent ${outcome.signals.join(' and ')} to the orphaned runner ${record.startId} (pid ${outcome.pid}); it is gone`
                    : `local-llm: the orphaned runner ${record.startId} (pid ${outcome.pid}) was already gone; no signal was sent`);
            }
        } catch (error) {
            if (!(error instanceof ProcessUnreadable)) throw error;
            result.skipped.push({ file: entry.file, reason: `unknown: ${error.message}` });
            log(`local-llm: runner ${record.startId} could not be looked at (${error.message}); its launch record and socket are kept and no further signal is sent for it`);
        }
    }
    return result;

    // Every process of the table. A table that lists everything in one read (macOS) gives each one's command and stat with it; the
    // Linux one gives the pids, and the command or stat is read per pid where it is needed.
    function everything() {
        return typeof table.listAll === 'function' ? table.listAll() : table.listPids().map((pid) => ({ pid }));
    }

    // The processes that are this record's runner right now.
    function find(record) {
        const found = new Map();
        if (record.phase === 'spawned') {
            const identity = identifyIn(table, record.pid, record.socketPath, record.startTime);
            if (identity) found.set(identity.pid, identity);
            else log(`local-llm: pid ${record.pid} of launch record ${record.startId} is not that runner any more (gone, or the pid was reused); it is not signalled`);
        }
        // The listing only picks the candidates: each one is identified again with a read of its own, as before every signal.
        for (const item of everything()) {
            const { pid } = item;
            if (pid === selfPid || found.has(pid)) continue;
            if (!listensOn(item.command !== undefined ? item.command : table.readCmdline(pid), record.socketPath)) continue;
            const identity = identifyIn(table, pid, record.socketPath, null);
            if (identity) found.set(pid, identity);
        }
        return [...found.values()];
    }

    // The group members of a leader, with their start times, before any signal is sent.
    function members(leader) {
        const list = [];
        for (const item of everything()) {
            const { pid } = item;
            if (pid === leader.pid || pid === selfPid) continue;
            const stat = item.stat !== undefined ? item.stat : table.readStat(pid);
            if (stat && stat.pgrp === leader.pid && stat.state !== 'Z' && stat.state !== 'X') list.push({ pid, startTime: stat.startTime, pgrp: stat.pgrp });
        }
        return list;
    }

    function same(candidate, record) {
        return identifyIn(table, candidate.pid, record.socketPath, candidate.startTime) !== null;
    }

    function send(pid, signal, sent) {
        try {
            killImpl(pid, signal);
            if (sent && !sent.includes(signal)) sent.push(signal);
            return true;
        } catch (error) {
            if (error?.code !== 'ESRCH') log(`local-llm: signal ${signal} to ${pid} failed (${error?.code ?? error?.message})`);
            return false;
        }
    }

    async function waitGone(candidate, record, ms) {
        const deadline = Date.now() + ms;
        while (same(candidate, record)) {
            if (Date.now() >= deadline) return false;
            await sleep(pollMs);
        }
        return true;
    }

    // One runner: TERM, wait, re-check, KILL, wait. The identity is read again before each signal. Returns `{ gone, signals }`:
    // `signals` lists the signals that were actually sent (none when the runner was gone before the first).
    async function end(candidate, record) {
        const leader = candidate.pgrp === candidate.pid;
        const group = leader ? members(candidate) : [];
        const target = leader ? -candidate.pid : candidate.pid;
        const signals = [];
        if (!same(candidate, record)) return { gone: true, signals };
        send(target, 'SIGTERM', signals);
        if (!(await waitGone(candidate, record, graceMs))) {
            if (!same(candidate, record)) return { gone: true, signals };
            log(`local-llm: runner ${record.startId} (pid ${candidate.pid}) did not exit within ${graceMs} ms; sending SIGKILL`);
            send(target, 'SIGKILL', signals);
            if (!(await waitGone(candidate, record, killWaitMs))) return { gone: false, signals };
        }
        // What the leader started and left behind: each member is the same process it was when seen, in the same group.
        for (const member of group) {
            const stat = table.readStat(member.pid);
            if (stat && stat.startTime === member.startTime && stat.pgrp === member.pgrp && stat.state !== 'Z' && stat.state !== 'X') send(member.pid, 'SIGKILL', signals);
        }
        return { gone: true, signals };
    }
}
