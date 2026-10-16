// The strict library check of a runnable copy of a runner: every ELF file it ships (executables and shared objects alike, found by
// the ELF magic, not by name) is run through `ldd` with no library path but the system's, and a library that is not found fails the
// check, with one exception: `libcuda.so.1`, which the NVIDIA driver supplies and a machine without a GPU does not have.
//
// This is stricter than the engine's install check (lib/local-llm/tools/runnerInstallCheck.mjs), which counts a library as provided
// when any file of the copy carries its name, whether or not the loader would find it (a broken `$ORIGIN` run path still passes there),
// and which runs `ldd` only on shared objects and on the one launched file. The workflow and the live smoke script use this check
// beside it.

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

export const DRIVER_LIBRARY = 'libcuda.so.1';
const MAGIC = Buffer.from([0x7f, 0x45, 0x4c, 0x46]);

/** The regular files under `dir` that start with the ELF magic, sorted, as absolute paths. Links are not followed. */
export function findElfFiles(dir, { fsApi = fs } = {}) {
    const found = [];
    const walk = (current) => {
        for (const entry of fsApi.readdirSync(current, { withFileTypes: true })) {
            const full = path.join(current, entry.name);
            if (entry.isDirectory()) walk(full);
            else if (entry.isFile() && hasElfMagic(full, fsApi)) found.push(full);
        }
    };
    walk(dir);
    return found.sort();
}

function hasElfMagic(file, fsApi) {
    let fd;
    try {
        fd = fsApi.openSync(file, 'r');
        const head = Buffer.alloc(MAGIC.length);
        return fsApi.readSync(fd, head, 0, MAGIC.length, 0) === MAGIC.length && head.equals(MAGIC);
    } catch {
        return false;
    } finally {
        if (fd !== undefined) { try { fsApi.closeSync(fd); } catch { /* nothing to do */ } }
    }
}

/** The libraries `ldd` reports as "not found" for `file`, run with the system path only (no LD_LIBRARY_PATH), or `{ error }` when ldd could not run. */
export function lddMissing(file, { spawn = spawnSync } = {}) {
    const result = spawn('ldd', [file], { encoding: 'utf8', env: { PATH: '/usr/bin:/bin' }, timeout: 60_000 });
    if (result.error) return { error: result.error.code || result.error.message };
    const output = `${result.stdout || ''}${result.stderr || ''}`;
    if (/not a dynamic executable/.test(output)) return { missing: [] };
    if (result.status !== 0 && !/=>/.test(output)) return { error: `ldd exited ${result.status ?? result.signal}: ${output.trim().slice(0, 200)}` };
    return { missing: output.split('\n').filter((line) => line.includes('not found')).map((line) => line.trim().split(/\s+/)[0]) };
}

/**
 * Checks every ELF file under `dir`. Resolves `{ ok, files, checked, driverOnly, unresolved, errors }`: `checked` is how many ELF files
 * there were, `driverOnly` the files whose only missing library is the driver's, `unresolved` maps a file (relative to `dir`) to the
 * libraries nothing supplies. No ELF file at all fails (an empty or wrongly stripped copy has nothing to prove).
 */
export function strictLibraryCheck(dir, { ldd = lddMissing, fsApi = fs } = {}) {
    const files = findElfFiles(dir, { fsApi });
    const unresolved = {};
    const errors = {};
    const driverOnly = [];
    for (const file of files) {
        const relative = path.relative(dir, file);
        const answer = ldd(file);
        if (answer.error) { errors[relative] = answer.error; continue; }
        const others = answer.missing.filter((library) => library !== DRIVER_LIBRARY);
        if (others.length) unresolved[relative] = others;
        else if (answer.missing.length) driverOnly.push(relative);
    }
    const problems = [];
    if (files.length === 0) problems.push('the copy holds no ELF file');
    if (Object.keys(unresolved).length) problems.push(`${Object.keys(unresolved).length} ELF files need libraries that are not found: ${Object.entries(unresolved).map(([f, l]) => `${f} (${l.join(', ')})`).join('; ')}`);
    if (Object.keys(errors).length) problems.push(`ldd could not check ${Object.keys(errors).length} ELF files: ${Object.entries(errors).map(([f, e]) => `${f} (${e})`).join('; ')}`);
    return { ok: problems.length === 0, checked: files.length, driverOnly: driverOnly.sort(), unresolved, errors, problems };
}
