// What the Linux process file system says about one process, read the way the reaper needs it: the start time and the
// process group from `stat`, the argument vector from `cmdline`. Nothing here parses `ps` text, and every reader takes the
// directory (`/proc` by default) so a test can point it at a tree it built.

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

/** Whether an argument vector names `socketPath` as the whole value of its `--host`. */
export function listensOn(argv, socketPath) {
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
 * The identity of `pid` as a process that listens on `socketPath`: its start time and group, or null when the process is
 * gone, a zombie, or does not have that socket as its `--host`. `startTime` null in `expectedStart` accepts any start time.
 */
export function identify(procDir, pid, socketPath, expectedStart = null, { fsApi = fs } = {}) {
    const stat = readStat(procDir, pid, { fsApi });
    if (!stat || stat.state === 'Z' || stat.state === 'X') return null;
    if (expectedStart !== null && stat.startTime !== expectedStart) return null;
    if (!listensOn(readCmdline(procDir, pid, { fsApi }), socketPath)) return null;
    return { pid, startTime: stat.startTime, pgrp: stat.pgrp };
}
