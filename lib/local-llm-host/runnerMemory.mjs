// The resident memory (RSS) of the runners this host started, summed, for the `ram` cap's guard backstop (caps.mjs, DS008 "Caps").
//
//   createRunnerMemoryReader({ launchDir, platformName, procDir, fsApi, exec, procs }) -> () => bytes | null
//
// Which pids: the launch records of the home (launch.mjs) in the phase `spawned`, each identified the way the reaper identifies one
// (procfs.mjs `identifyIn`): the recorded start time must be the process's own, and its command line must name the record's socket as
// its `--host`. A reused pid, a zombie and a process that is gone are not counted. Launch records were chosen over the starter's child
// handles because the limits are built when the host is created and the starter only when the controller is, so the handles would need a
// registry shared between the two, and a record carries what the identity check needs.
//
// Where the figure comes from: on Linux the `VmRSS` line of `<procDir>/<pid>/status` (kB, which are KiB); on macOS the
// `rss` column of one `/bin/ps -ww -o pid=,pgid=,state=,lstart=,rss=,command= -p <pids>` per reading (two spawns fewer per runner than
// reading the table pid by pid), through the `exec(file, args, options) -> text` seam darwinProcs.mjs uses (rss is in KiB). The same
// rows identify the runners, so no other `ps` runs. On Apple
// Silicon the RSS of a Metal process may leave out the memory the GPU driver wires for it (to be measured live, DS008).
//
// The result is a whole number of bytes, 0 when no runner is alive, and null when it could not be read: a record directory that cannot
// be listed, a `ps` that failed, a status or a `ps` row that is not in the expected form. It never throws.

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

import { parsePsRows } from './darwinProcs.mjs';
import { readRecords } from './launch.mjs';
import { createProcfsProcs, identifyIn } from './procfs.mjs';

const PS = '/bin/ps';
const FIELDS = 'pid=,pgid=,state=,lstart=,rss=,command=';
const TOOL_TIMEOUT_MS = 10_000;
const MAX_OUTPUT_BYTES = 1024 * 1024;
const TOOL_ENV = Object.freeze({ PATH: '/usr/bin:/bin', LC_ALL: 'C', LANG: 'C', TZ: 'UTC' });
const VM_RSS = /^VmRSS:[ \t]+(\d+)[ \t]+kB[ \t]*$/m;
const KIB = 1024;

/** The bytes of a `status` text: the `VmRSS` line in KiB, 0 when the process has none (a kernel thread, a zombie), null when the line is not a number of kB. */
export function parseVmRss(text) {
    const match = VM_RSS.exec(text);
    if (!match) return /^VmRSS:/m.test(text) ? null : 0;
    const bytes = Number(match[1]) * KIB;
    return Number.isSafeInteger(bytes) ? bytes : null;
}

export function createRunnerMemoryReader({
    launchDir, platformName = process.platform, procDir = '/proc', fsApi = fs, exec = execFileSync, procs = null,
} = {}) {
    const linuxTable = procs ?? (platformName === 'linux' ? createProcfsProcs(procDir, { fsApi }) : null);

    // The spawned records of this home: { pid, socketPath, startTime }. Throws when the records cannot be read.
    function spawnedRecords() {
        const records = [];
        for (const entry of readRecords(launchDir, { fsApi })) {
            if (entry.error || entry.record.phase !== 'spawned') continue;
            records.push(entry.record);
        }
        return records;
    }

    function linuxBytes(records) {
        let total = 0;
        for (const { pid, socketPath, startTime } of records) {
            if (!identifyIn(linuxTable, pid, socketPath, startTime)) continue;
            let text;
            try {
                text = fsApi.readFileSync(path.join(procDir, String(pid), 'status'), 'utf8');
            } catch (error) {
                if (error?.code === 'ENOENT' || error?.code === 'ESRCH') continue; // gone since it was identified
                return null;
            }
            const bytes = parseVmRss(text);
            if (bytes === null) return null;
            total += bytes;
        }
        return Number.isSafeInteger(total) ? total : null;
    }

    // One `ps` for all the recorded pids: the same row gives the identity (start time, state, command) and the rss.
    function darwinBytes(records) {
        const pids = [...new Set(records.map((record) => record.pid).filter((pid) => Number.isSafeInteger(pid) && pid > 0))];
        if (pids.length === 0) return 0;
        let text;
        try {
            text = String(exec(PS, ['-ww', '-o', FIELDS, '-p', pids.join(',')], { encoding: 'utf8', timeout: TOOL_TIMEOUT_MS, env: TOOL_ENV, maxBuffer: MAX_OUTPUT_BYTES }));
        } catch (error) {
            // `ps -p` exits 1 when a pid is not there, and still prints the rows of the others: those rows are the answer, and no row at all means all are gone.
            const stdout = error?.status === 1 && !error.code && !error.signal ? error.stdout : null;
            if (typeof stdout !== 'string' && !Buffer.isBuffer(stdout)) return null;
            text = String(stdout);
        }
        const parsed = parsePsRows(text, { rss: true });
        if (!parsed) return null;
        const rows = new Map();
        for (const row of parsed) {
            if (!pids.includes(row.pid)) return null; // a row that is not one asked for is not `ps` output of this call
            rows.set(row.pid, row);
        }
        // The rows as a process table, so the identity is the reaper's own (procfs.mjs `identifyIn`).
        const table = { readStat: (pid) => rows.get(pid)?.stat ?? null, readCmdline: (pid) => rows.get(pid)?.command ?? null };
        let total = 0;
        for (const { pid, socketPath, startTime } of records) {
            if (identifyIn(table, pid, socketPath, startTime)) total += rows.get(pid).rssKib * KIB;
        }
        return Number.isSafeInteger(total) ? total : null;
    }

    return function readRunnerMemory() {
        try {
            if (typeof launchDir !== 'string') return null;
            if (platformName === 'darwin' && !procs) return darwinBytes(spawnedRecords());
            if (!linuxTable) return null;
            return linuxBytes(spawnedRecords());
        } catch {
            // An unreadable record directory is no reading; so is anything unexpected.
            return null;
        }
    };
}
