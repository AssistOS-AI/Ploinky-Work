// What `ps` says about processes on macOS, in the shape the reaper reads a process table (procfs.mjs has the Linux one):
//
//   listPids()        every pid in the table
//   readStat(pid)     { state, pgrp, startTime } or null: `state` is the first letter of ps's state ('Z' is a zombie),
//                     `startTime` is `lstart` as UTC epoch seconds in decimal digits (what a launch record keeps)
//   readCmdline(pid)  the command line as one string, or null (ps joins the arguments with spaces, so the reaper matches text)
//   listAll()         [{ pid, stat, command }] for every process, from one `ps` run
//
// Every read runs `/bin/ps -ww` (no truncation of the command) with `TZ=UTC LC_ALL=C LANG=C PATH=/usr/bin:/bin`, through an
// `exec(file, args, options) -> text` function that tests replace.
//
// Null means one thing: `ps -p` exited with status 1 and printed nothing, which is how it says there is no such process. Every
// other failure (a timeout, a signal, a spawn error, too much output, another status, partial or unreadable output) is not "gone":
// a single read throws `ProcessUnreadable` and a listing throws, so the reaper keeps the launch record and the socket, signals
// nothing more, and says so, instead of treating a live runner as gone.

import { execFileSync } from 'node:child_process';

import { LSTART_PATTERN, lstartFromMatch } from './darwin.mjs';

const PS = '/bin/ps';
const FIELDS = 'pid=,pgid=,state=,lstart=,command=';
const TOOL_TIMEOUT_MS = 10_000;
const MAX_OUTPUT_BYTES = 256 * 1024 * 1024;
const TOOL_ENV = Object.freeze({ PATH: '/usr/bin:/bin', LC_ALL: 'C', LANG: 'C', TZ: 'UTC' });
// pid, process group, state, the start (`Fri Oct 16 03:42:14 2026`), then the command to the end of the line.
// A process that is exiting has the state `?NEs` and the command `(node)`: no state letter, the E (exiting) flag, read as 'X' (dead).
// Any other state without a letter is not understood, so the output is unreadable rather than a guess.
const ROW_HEAD = `^\\s*(\\d+)\\s+(\\d+)\\s+(\\S+)\\s+${LSTART_PATTERN}`;
const ROW = new RegExp(`${ROW_HEAD}\\s+(\\S.*?)\\s*$`);
// The same row with the resident size (`rss=`, KiB) between the start and the command: `-o pid=,pgid=,state=,lstart=,rss=,command=`.
const ROW_RSS = new RegExp(`${ROW_HEAD}\\s+(\\d+)\\s+(\\S.*?)\\s*$`);

/**
 * The rows of `ps -ww -o pid=,pgid=,state=,lstart=,command=` text: { pid, stat: { state, pgrp, startTime }, command }. A line
 * that is not a row continues the command of the row before it (a command line may hold a newline); one before the first
 * row is not `ps` output at all and the text is refused (null). A row whose start is not a real date is refused too.
 * With `{ rss: true }` the text has the `rss=` column before the command and each row also has `rssKib` (a whole number).
 */
export function parsePsRows(text, { rss = false } = {}) {
    const pattern = rss ? ROW_RSS : ROW;
    const rows = [];
    for (const line of String(text ?? '').split('\n')) {
        if (line.trim() === '') continue;
        const match = pattern.exec(line);
        if (!match) {
            if (!rows.length) return null;
            rows.at(-1).command += `\n${line}`;
            continue;
        }
        const start = lstartFromMatch(match.slice(4, 10));
        const [pid, pgrp] = [Number(match[1]), Number(match[2])];
        if (!start || !Number.isSafeInteger(pid) || pid <= 0 || !Number.isSafeInteger(pgrp)) return null;
        const state = /^[A-Za-z]/.test(match[3]) ? match[3][0].toUpperCase() : match[3].includes('E') ? 'X' : null;
        if (state === null) return null;
        const entry = { pid, stat: { state, pgrp, startTime: String(Math.floor(start.startMs / 1000)) }, command: match[rss ? 11 : 10] };
        if (rss) entry.rssKib = Number(match[10]);
        rows.push(entry);
    }
    return rows;
}

/** A process that `ps` could not be read for: not the same as a process that is gone. */
export class ProcessUnreadable extends Error {
    constructor(message) {
        super(message);
        this.name = 'ProcessUnreadable';
    }
}

// How `ps -p` says "no such process": exit status 1 with nothing on stdout, and no error code, no signal.
function noSuchProcess(error) {
    const stdout = error?.stdout;
    return error?.status === 1 && !error.code && !error.signal && (typeof stdout === 'string' ? stdout === '' : Buffer.isBuffer(stdout) && stdout.length === 0);
}

export function createDarwinProcs({ exec = execFileSync } = {}) {
    // The output of `ps`: { text } when it ran and exited 0, { gone: true } for the one answer that means "no such process", and
    // { problem } for every other outcome, partial output of a failed run included (it is never used).
    function run(args) {
        try {
            return { text: String(exec(PS, args, { encoding: 'utf8', timeout: TOOL_TIMEOUT_MS, env: TOOL_ENV, maxBuffer: MAX_OUTPUT_BYTES })) };
        } catch (error) {
            if (noSuchProcess(error)) return { gone: true };
            const why = error?.signal ? `signal ${error.signal}` : error?.code ? String(error.code) : error?.status !== undefined ? `exit status ${error.status}` : String(error?.message ?? error);
            return { problem: `\`ps\` failed (${why})` };
        }
    }

    function row(pid) {
        if (!Number.isSafeInteger(pid) || pid <= 0) return null;
        const answer = run(['-ww', '-o', FIELDS, '-p', String(pid)]);
        if (answer.gone) return null;
        if (answer.problem) throw new ProcessUnreadable(`process ${pid} could not be read: ${answer.problem}`);
        const rows = parsePsRows(answer.text);
        if (!rows || rows.length !== 1 || rows[0].pid !== pid) {
            throw new ProcessUnreadable(`process ${pid} could not be read: \`ps\` answered with text that is not its row (${JSON.stringify(answer.text.slice(0, 160))})`);
        }
        return rows[0];
    }

    function listAll() {
        const answer = run(['-ww', '-A', '-o', FIELDS]);
        if (answer.problem) throw new Error(`\`ps\` did not list the processes: ${answer.problem}`);
        const rows = answer.gone ? null : parsePsRows(answer.text);
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
