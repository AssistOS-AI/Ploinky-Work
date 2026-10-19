// A fake /proc: a directory tree with the files the reaper reads (stat, cmdline), so the identity and signal logic runs on any
// operating system, and the fake kill that records every signal instead of sending one.
import fs from 'node:fs';
import path from 'node:path';

/** The contents of `<pid>/stat`: `comm` may hold spaces and parentheses, and the start time is field 22. */
export function statLine({ pid, comm = 'llama-server', state = 'S', ppid = 1, pgrp = pid, startTime = '123456' }) {
    const after = [state, ppid, pgrp, pgrp, 0, -1, 4194560, 100, 0, 0, 0, 5, 1, 0, 0, 20, 0, 4, 0, startTime, 1000, 100, 18446744073709551615];
    return `${pid} (${comm}) ${after.join(' ')}\n`;
}

/** The contents of `<pid>/status`: a few of its lines, with `VmRSS` in kB when `rssKb` is given (a kernel thread or a zombie has none). */
export function statusText({ pid, comm = 'llama-server', state = 'S', rssKb }) {
    return [`Name:\t${comm}`, `State:\t${state} (sleeping)`, `Pid:\t${pid}`, 'VmPeak:\t 9999999 kB', ...(rssKb === undefined ? [] : [`VmRSS:\t${String(rssKb).padStart(8, ' ')} kB`]), 'Threads:\t4', ''].join('\n');
}

export function writeStatus(procDir, spec) {
    fs.writeFileSync(path.join(procDir, String(spec.pid), 'status'), typeof spec.text === 'string' ? spec.text : statusText(spec));
}

export function writeProc(procDir, spec) {
    const dir = path.join(procDir, String(spec.pid));
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'stat'), statLine(spec));
    fs.writeFileSync(path.join(dir, 'cmdline'), `${(spec.argv ?? []).join('\0')}\0`);
    if (spec.rssKb !== undefined || typeof spec.statusText === 'string') writeStatus(procDir, { ...spec, ...(typeof spec.statusText === 'string' ? { text: spec.statusText } : {}) });
}

export function removeProc(procDir, pid) {
    fs.rmSync(path.join(procDir, String(pid)), { recursive: true, force: true });
}

/** The argv of a llama-server runner on `socketPath`. */
export const runnerArgv = (socketPath, model = '/m/q.gguf') => ['/run/llama.cpp/b11295/llama-server', '-m', model, '--host', socketPath, '--api-key', 'k'];

/**
 * A kill that records its calls and, per signal, can remove the fake process(es): `dies` maps a signal to a function(pid)
 * that applies its effect to the fake tree.
 */
export function recordingKill({ dies = {} } = {}) {
    const calls = [];
    const kill = (pid, signal) => {
        calls.push([pid, signal]);
        dies[signal]?.(pid);
    };
    kill.calls = calls;
    return kill;
}
