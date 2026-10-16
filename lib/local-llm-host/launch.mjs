// The launch record of one runner start: written durably before the runner can load a model, so a crash at any point leaves
// either no child or a child the reaper can find. One file per start, `<launchDir>/<startId>.json`:
//
//   { schema: 1, startId, socketPath, phase: 'intent' }                          before the spawn
//   { schema: 1, startId, socketPath, phase: 'spawned', pid, startTime }         right after it
//
// `startTime` is field 22 of `/proc/<pid>/stat` (null where the platform has none). Every write goes through a temporary
// file in the same directory: write, fsync, rename over the record, fsync of the directory. A reader therefore sees the
// old record or the new one, never a torn one.

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

import { readStat } from './procfs.mjs';

export const LAUNCH_SCHEMA = 1;
export const START_ID_RE = /^[0-9a-f]{8}$/;
const SOCKET_NAME_RE = /^[0-9a-f]{8}\.sock$/;

/** `text` written to `file` so that a crash leaves the previous content or all of the new. */
export function writeDurable(file, text, { fsApi = fs } = {}) {
    const dir = path.dirname(file);
    const temporary = path.join(dir, `.${path.basename(file)}.${process.pid}.${crypto.randomBytes(4).toString('hex')}.tmp`);
    const fd = fsApi.openSync(temporary, 'wx', 0o600);
    try {
        fsApi.writeFileSync(fd, text);
        fsApi.fsyncSync(fd);
    } finally {
        fsApi.closeSync(fd);
    }
    try {
        fsApi.renameSync(temporary, file);
    } catch (error) {
        try { fsApi.unlinkSync(temporary); } catch {}
        throw error;
    }
    const dirFd = fsApi.openSync(dir, 'r');
    try {
        fsApi.fsyncSync(dirFd);
    } finally {
        fsApi.closeSync(dirFd);
    }
}

/**
 * The record in `value`, checked: a plain object of this schema whose start id is 8 hex characters, whose socket path is
 * absolute and named `<startId>.sock`, and whose `spawned` phase carries a pid above 1 and a start time. Throws an Error
 * that says what is wrong; the reaper turns that into a clear log line and leaves the process alone.
 */
export function validateRecord(value) {
    const bad = (message) => { throw new Error(message); };
    if (value === null || typeof value !== 'object' || Array.isArray(value)) bad('it is not an object');
    if (value.schema !== LAUNCH_SCHEMA) bad(`its schema is ${JSON.stringify(value.schema)}, not ${LAUNCH_SCHEMA}`);
    if (typeof value.startId !== 'string' || !START_ID_RE.test(value.startId)) bad('its startId is not 8 hex characters');
    if (typeof value.socketPath !== 'string' || !path.isAbsolute(value.socketPath) || path.normalize(value.socketPath) !== value.socketPath) {
        bad('its socketPath is not an absolute normalized path');
    }
    if (path.basename(value.socketPath) !== `${value.startId}.sock`) bad('its socketPath is not <startId>.sock');
    if (value.phase === 'intent') {
        if (value.pid !== undefined || value.startTime !== undefined) bad('an intent record carries a pid');
        return { startId: value.startId, socketPath: value.socketPath, phase: 'intent', pid: null, startTime: null };
    }
    if (value.phase !== 'spawned') bad(`its phase is ${JSON.stringify(value.phase)}`);
    if (!Number.isSafeInteger(value.pid) || value.pid <= 1) bad('its pid is not a process id');
    if (value.startTime !== null && (typeof value.startTime !== 'string' || !/^\d+$/.test(value.startTime))) bad('its startTime is not a tick count or null');
    return { startId: value.startId, socketPath: value.socketPath, phase: 'spawned', pid: value.pid, startTime: value.startTime };
}

/** Every record file in `dir`: `{ file, record }` for a valid one, `{ file, error }` for one that is not. Temporary files and other names are ignored. */
export function readRecords(dir, { fsApi = fs } = {}) {
    let names;
    try {
        names = fsApi.readdirSync(dir);
    } catch (error) {
        if (error?.code === 'ENOENT') return [];
        throw error;
    }
    const entries = [];
    for (const name of names.sort()) {
        if (!/^[0-9a-f]{8}\.json$/.test(name)) continue;
        const file = path.join(dir, name);
        try {
            const record = validateRecord(JSON.parse(fsApi.readFileSync(file, 'utf8')));
            if (`${record.startId}.json` !== name) throw new Error('its startId is not the name of its file');
            entries.push({ file, record });
        } catch (error) {
            entries.push({ file, error: error instanceof SyntaxError ? `it is not JSON (${error.message})` : error.message });
        }
    }
    return entries;
}

/**
 * The writer of one home's launch records. `intent` runs before the spawn (a failure means no child is started),
 * `spawned` right after it, and `clear` once the runner and its group are gone.
 */
export function createLaunchRecorder({ dir, procDir = '/proc', fsApi = fs, platformName = process.platform }) {
    const fileOf = (startId) => {
        if (!START_ID_RE.test(startId)) throw new Error(`Invalid start id ${String(startId)}`);
        return path.join(dir, `${startId}.json`);
    };
    const write = (startId, body) => {
        fsApi.mkdirSync(dir, { recursive: true, mode: 0o700 });
        writeDurable(fileOf(startId), `${JSON.stringify({ schema: LAUNCH_SCHEMA, startId, ...body })}\n`, { fsApi });
    };
    return Object.freeze({
        dir,
        fileOf,
        intent({ startId, socketPath }) {
            write(startId, { socketPath, phase: 'intent' });
        },
        spawned({ startId, socketPath, pid }) {
            // Where the platform has no /proc the start time stays null and the cmdline match alone identifies the child.
            const stat = platformName === 'linux' ? readStat(procDir, pid, { fsApi }) : null;
            write(startId, { socketPath, phase: 'spawned', pid, startTime: stat?.startTime ?? null });
        },
        /** Removes the record of `startId` and, when it is a socket of this host's own directory, the socket file. */
        clear({ startId, socketPath = null, socketDir = null }) {
            try { fsApi.unlinkSync(fileOf(startId)); } catch (error) { if (error?.code !== 'ENOENT') throw error; }
            if (socketPath && socketDir) removeSocketFile(socketPath, socketDir, { fsApi });
        },
    });
}

/**
 * Removes a runner's socket file when, and only when, it is named `<8 hex>.sock`, sits directly in `socketDir`, and is a
 * socket owned by this user: a tampered record cannot make this delete anything else. Returns whether a file was removed.
 */
export function removeSocketFile(socketPath, socketDir, { fsApi = fs } = {}) {
    if (path.dirname(socketPath) !== socketDir || !SOCKET_NAME_RE.test(path.basename(socketPath))) return false;
    let stat;
    try {
        stat = fsApi.lstatSync(socketPath);
    } catch {
        return false;
    }
    if (!stat.isSocket() || (typeof process.getuid === 'function' && stat.uid !== process.getuid())) return false;
    try {
        fsApi.unlinkSync(socketPath);
        return true;
    } catch {
        return false;
    }
}
