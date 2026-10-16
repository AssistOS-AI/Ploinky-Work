// What `ps` says about processes on macOS, in the shape the reaper reads a process table (procfs.mjs has the Linux one):
//
//   listPids()        every pid in the table
//   readStat(pid)     { state, pgrp, startTime } or null: `state` is the first letter of ps's state ('Z' is a zombie),
//                     `startTime` is `lstart` as UTC epoch seconds in decimal digits (what a launch record keeps)
//   readCmdline(pid)  the command line as one string, or null (ps joins the arguments with spaces, so the reaper matches text)
//   listAll()         [{ pid, stat, command }] for every process, from one `ps` run
//
// Every read runs `/bin/ps -ww` (no truncation of the command) with `TZ=UTC LC_ALL=C LANG=C PATH=/usr/bin:/bin`, through an
// `exec(file, args, options) -> text` function that tests replace. A single-process read that cannot be understood is null (a
// process nobody can identify is never signalled); a listing that cannot be understood throws, so the reaper stops before
// it removes a record on the strength of "no such runner".

import { execFileSync } from 'node:child_process';

import { LSTART_PATTERN, lstartFromMatch } from './darwin.mjs';

const PS = '/bin/ps';
const FIELDS = 'pid=,pgid=,state=,lstart=,command=';
const TOOL_TIMEOUT_MS = 10_000;
const MAX_OUTPUT_BYTES = 256 * 1024 * 1024;
const TOOL_ENV = Object.freeze({ PATH: '/usr/bin:/bin', LC_ALL: 'C', LANG: 'C', TZ: 'UTC' });
// pid, process group, state, the start (`Fri Oct 16 03:42:14 2026`), then the command to the end of the line.
const ROW = new RegExp(`^\\s*(\\d+)\\s+(\\d+)\\s+([A-Za-z]\\S*)\\s+${LSTART_PATTERN}\\s+(\\S.*?)\\s*$`);

/**
 * The rows of `ps -ww -o pid=,pgid=,state=,lstart=,command=` text: { pid, stat: { state, pgrp, startTime }, command }. A line
 * that is not a row continues the command of the row before it (a command line may hold a newline); one before the first
 * row is not `ps` output at all and the text is refused (null). A row whose start is not a real date is refused too.
 */
export function parsePsRows(text) {
    const rows = [];
    for (const line of String(text ?? '').split('\n')) {
        if (line.trim() === '') continue;
        const match = ROW.exec(line);
        if (!match) {
            if (!rows.length) return null;
            rows.at(-1).command += `\n${line}`;
            continue;
        }
        const start = lstartFromMatch(match.slice(4, 10));
        const [pid, pgrp] = [Number(match[1]), Number(match[2])];
        if (!start || !Number.isSafeInteger(pid) || pid <= 0 || !Number.isSafeInteger(pgrp)) return null;
        rows.push({ pid, stat: { state: match[3][0].toUpperCase(), pgrp, startTime: String(Math.floor(start.startMs / 1000)) }, command: match[10] });
    }
    return rows;
}

export function createDarwinProcs({ exec = execFileSync } = {}) {
    // The output of `ps`, or null when it cannot run. A pid that does not exist makes `ps -p` exit 1 with nothing printed.
    function run(args) {
        try {
            return String(exec(PS, args, { encoding: 'utf8', timeout: TOOL_TIMEOUT_MS, env: TOOL_ENV, maxBuffer: MAX_OUTPUT_BYTES }));
        } catch (error) {
            return typeof error?.stdout === 'string' ? error.stdout : null;
        }
    }

    function row(pid) {
        if (!Number.isSafeInteger(pid) || pid <= 0) return null;
        const rows = parsePsRows(run(['-ww', '-o', FIELDS, '-p', String(pid)]));
        return rows && rows.length === 1 && rows[0].pid === pid ? rows[0] : null;
    }

    function listAll() {
        const rows = parsePsRows(run(['-ww', '-A', '-o', FIELDS]));
        if (!rows || rows.length === 0) throw new Error('`ps` did not list the processes in a form that can be read');
        return rows;
    }

    return Object.freeze({
        listAll,
        listPids: () => listAll().map((entry) => entry.pid).sort((a, b) => a - b),
        readStat: (pid) => row(pid)?.stat ?? null,
        readCmdline: (pid) => row(pid)?.command ?? null,
    });
}
