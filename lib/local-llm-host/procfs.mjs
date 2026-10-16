// What the Linux process file system says about one process, read the way the reaper needs it: the start time and the
// process group from `stat`, the argument vector from `cmdline`. Nothing here parses `ps` text, and every reader takes the
// directory (`/proc` by default) so a test can point it at a tree it built. `createProcfsProcs` is the process table the
// reaper reads on Linux (reaper.mjs); darwinProcs.mjs is the macOS one.

import fs from 'node:fs';
import path from 'node:path';

/**
 * The fields of `<procDir>/<pid>/stat` the reaper uses, or null when the process is gone or the file is not readable
 * as a stat line. `comm` is the second field in parentheses and may itself hold spaces and parentheses, so everything is
 * counted from the last ')'. Field 22 (the start time, in clock ticks since boot) is index 19 after it.
 */
export function readStat(procDir, pid, { fsApi = fs } = {}) {
    let text;
    try {
        text = fsApi.readFileSync(path.join(procDir, String(pid), 'stat'), 'utf8');
    } catch {
        return null;
    }
    const close = text.lastIndexOf(')');
    if (close < 0) return null;
    const rest = text.slice(close + 1).trim().split(/\s+/);
    // state ppid pgrp session ... starttime: field 3 is rest[0], so field 22 is rest[19].
    if (rest.length < 20) return null;
    const [state, , pgrp] = rest;
    const startTime = rest[19];
    if (!/^\d+$/.test(startTime) || !/^\d+$/.test(pgrp) || !/^[A-Za-z]$/.test(state)) return null;
    return { state, pgrp: Number(pgrp), startTime };
}

/** The argument vector of a process (its `cmdline` split at NUL), or null when it is gone or has none. */
export function readCmdline(procDir, pid, { fsApi = fs } = {}) {
    let text;
    try {
        text = fsApi.readFileSync(path.join(procDir, String(pid), 'cmdline'), 'utf8');
    } catch {
        return null;
    }
    if (!text) return null;
    const parts = text.split('\0');
    if (parts.at(-1) === '') parts.pop();
    return parts;
}

/**
 * Whether a command line names `socketPath` as the whole value of its `--host`. A vector (Linux: `cmdline` split at NUL) must
 * hold it as the argument that follows `--host`. A string (macOS: `ps` prints one command, the arguments joined by spaces)
 * must hold ` --host <socketPath>` followed by a space or the end of the string, so `<socketPath>2` is not a match. The socket
 * path is `<8 hex>.sock` in a directory without ':' or ',', and the whole path is in the needle, so a home with spaces still matches.
 */
export function listensOn(argv, socketPath) {
    if (typeof argv === 'string') {
        const needle = ` --host ${socketPath}`;
        for (let at = argv.indexOf(needle); at >= 0; at = argv.indexOf(needle, at + 1)) {
            const end = at + needle.length;
            if (end === argv.length || argv[end] === ' ') return true;
        }
        return false;
    }
    if (!Array.isArray(argv)) return false;
    return argv.some((value, index) => index > 0 && value === socketPath && argv[index - 1] === '--host');
}

/** The pids in `procDir`: its all-digit entries. */
export function listPids(procDir, { fsApi = fs } = {}) {
    try {
        return fsApi.readdirSync(procDir).filter((name) => /^\d+$/.test(name)).map(Number).sort((a, b) => a - b);
    } catch {
        return [];
    }
}

/**
 * A process table in the shape the reaper reads, over a (fake or real) `/proc`: `listPids()`, `readStat(pid)` and
 * `readCmdline(pid)`. The macOS one is darwinProcs.mjs.
 */
export function createProcfsProcs(procDir, { fsApi = fs } = {}) {
    return Object.freeze({
        listPids: () => listPids(procDir, { fsApi }),
        readStat: (pid) => readStat(procDir, pid, { fsApi }),
        readCmdline: (pid) => readCmdline(procDir, pid, { fsApi }),
    });
}

/**
 * The identity of `pid` in the process table `procs` as a process that listens on `socketPath`: its start time and group, or null
 * when the process is gone, a zombie, or does not have that socket as its `--host`. `expectedStart` null accepts any start time.
 */
export function identifyIn(procs, pid, socketPath, expectedStart = null) {
    const stat = procs.readStat(pid);
    if (!stat || stat.state === 'Z' || stat.state === 'X') return null;
    if (expectedStart !== null && stat.startTime !== expectedStart) return null;
    if (!listensOn(procs.readCmdline(pid), socketPath)) return null;
    return { pid, startTime: stat.startTime, pgrp: stat.pgrp };
}

/** `identifyIn` over the `/proc` at `procDir`. */
export function identify(procDir, pid, socketPath, expectedStart = null, { fsApi = fs } = {}) {
    return identifyIn(createProcfsProcs(procDir, { fsApi }), pid, socketPath, expectedStart);
}
