// The strict library check of a runnable copy of a runner on macOS, the counterpart of elf.mjs: every Mach-O file the copy ships
// (executables and dylibs alike, found by the Mach-O magic, not by name; links are not followed) is read with `otool -L` (its
// dependencies) and `otool -l` (its LC_RPATH entries), and a dependency passes only when
//   (a) it lives under /usr/lib/ or /System/Library/ (since Big Sur those are in the dyld shared cache, not on disk, so they are
//       provided by the system and cannot be looked up), or
//   (b) it is @rpath/..., @loader_path/... or @executable_path/... and resolves, through the file's own rpaths (@loader_path is the
//       file's directory, @executable_path the launched program's directory), to an existing file inside the copy. Relative links
//       of the copy are followed, but a path that leaves the copy does not count.
// Anything else (/opt/homebrew/..., /usr/local/..., an @rpath nothing in the copy answers) fails, naming the file and the library.
// A dylib's own install name (its LC_ID_DYLIB, the first line of `otool -L`) is not a dependency. A fat file is checked for the slices
// otool prints (by default the host's own, after a banner naming every architecture), not for every architecture in it.

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

export const OTOOL = '/usr/bin/otool';
const SYSTEM_PREFIXES = ['/usr/lib/', '/System/Library/'];
// feedfacf / cffaedfe: 64-bit Mach-O, big / little endian; cafebabe / cafebabf (64-bit offsets) / bebafeca / bfbafeca: a fat file, big / little endian.
const THIN_MAGIC = new Set(['feedfacf', 'cffaedfe']);
const FAT_MAGIC = new Set(['cafebabe', 'cafebabf', 'bebafeca', 'bfbafeca']);
// A Java class file also starts with cafebabe, followed by its version (at least 45); a fat file is followed by its architecture count.
const MAX_FAT_ARCHITECTURES = 30;

/** The regular files under `dir` that start with a Mach-O magic, sorted, as absolute paths. Links are not followed. */
export function findMachoFiles(dir, { fsApi = fs } = {}) {
    const found = [];
    const walk = (current) => {
        for (const entry of fsApi.readdirSync(current, { withFileTypes: true })) {
            const full = path.join(current, entry.name);
            if (entry.isDirectory()) walk(full);
            else if (entry.isFile() && hasMachoMagic(full, fsApi)) found.push(full);
        }
    };
    walk(dir);
    return found.sort();
}

function hasMachoMagic(file, fsApi) {
    let fd;
    try {
        fd = fsApi.openSync(file, 'r');
        const head = Buffer.alloc(8);
        if (fsApi.readSync(fd, head, 0, 8, 0) !== 8) return false;
        const magic = head.subarray(0, 4).toString('hex');
        if (THIN_MAGIC.has(magic)) return true;
        if (!FAT_MAGIC.has(magic)) return false;
        const count = magic.startsWith('cafebab') ? head.readUInt32BE(4) : head.readUInt32LE(4);
        return count >= 1 && count <= MAX_FAT_ARCHITECTURES;
    } catch {
        return false;
    } finally {
        if (fd !== undefined) { try { fsApi.closeSync(fd); } catch { /* nothing to do */ } }
    }
}

// otool prints "<file>:" for a thin file and "<file> (architecture <arch>):" before each slice of a fat one.
const HEADER_RE = /^\S.*?(?: \(architecture (\S+)\))?:$/;

/**
 * `otool -L` output as slices `[{ arch, libraries, unreadable }]` (arch is null for a thin file); the libraries are in the order
 * printed. An indented line that is not a dependency is kept in `unreadable`, never dropped: the caller fails on it.
 */
export function parseOtoolL(output) {
    const slices = [];
    for (const line of String(output).split('\n')) {
        const header = HEADER_RE.exec(line);
        if (header) { slices.push({ arch: header[1] ?? null, libraries: [], unreadable: [] }); continue; }
        const library = /^\s+(\S.*?) \(compatibility version [^)]*\)\s*$/.exec(line);
        if (!slices.length) continue;
        if (library) slices.at(-1).libraries.push(library[1]);
        else if (/^\s+\S/.test(line)) slices.at(-1).unreadable.push(line.trim());
    }
    return slices;
}

/** `otool -l` output as slices `[{ arch, rpaths, installName }]`: the LC_RPATH paths in order, and the LC_ID_DYLIB name (null for a program). */
export function parseOtoolLoadCommands(output) {
    const slices = [];
    let command = null;
    for (const line of String(output).split('\n')) {
        const header = HEADER_RE.exec(line);
        if (header) { slices.push({ arch: header[1] ?? null, rpaths: [], installName: null }); command = null; continue; }
        if (!slices.length) continue;
        const cmd = /^\s+cmd (\S+)\s*$/.exec(line);
        if (cmd) { command = cmd[1]; continue; }
        if (command === 'LC_RPATH') {
            const rpath = /^\s+path (\S.*?) \(offset \d+\)\s*$/.exec(line);
            if (rpath) { slices.at(-1).rpaths.push(rpath[1]); command = null; }
        } else if (command === 'LC_ID_DYLIB') {
            const name = /^\s+name (\S.*?) \(offset \d+\)\s*$/.exec(line);
            if (name) { slices.at(-1).installName = name[1]; command = null; }
        }
    }
    return slices;
}

/** Runs `/usr/bin/otool <flag> <file>`; resolves `{ output }` or `{ error }`. */
export function runOtool(flag, file, { spawn = spawnSync } = {}) {
    const result = spawn(OTOOL, [flag, file], { encoding: 'utf8', env: { PATH: '/usr/bin:/bin' }, timeout: 60_000, maxBuffer: 64 * 1024 * 1024 });
    if (result.error) return { error: result.error.code || result.error.message };
    if (result.status !== 0) return { error: `otool ${flag} exited ${result.status ?? result.signal}: ${`${result.stderr || ''}`.trim().slice(0, 200)}` };
    return { output: result.stdout || '' };
}

const isInside = (root, file) => file === root || file.startsWith(root.endsWith(path.sep) ? root : root + path.sep);

/**
 * The file in the copy that `library` names for `file`, or null. `root` is the copy's real path, `executableDir` the directory of the
 * launched program. A candidate counts when it is an existing regular file whose real path (relative links followed) is inside `root`.
 */
export function resolveDependency(library, { file, rpaths, root, executableDir, fsApi = fs }) {
    const expand = (name) => {
        if (name.startsWith('@loader_path')) return path.join(path.dirname(file), name.slice('@loader_path'.length));
        if (name.startsWith('@executable_path')) return path.join(executableDir, name.slice('@executable_path'.length));
        return name;
    };
    let candidates = [];
    if (library.startsWith('@rpath/')) candidates = rpaths.map((rpath) => path.join(expand(rpath), library.slice('@rpath/'.length)));
    else if (/^@(loader_path|executable_path)(\/|$)/.test(library)) candidates = [expand(library)];
    for (const candidate of candidates) {
        if (!path.isAbsolute(candidate)) continue;
        try {
            const real = fsApi.realpathSync(candidate);
            if (isInside(root, real) && fsApi.statSync(real).isFile()) return real;
        } catch { /* not there: try the next rpath */ }
    }
    return null;
}

const isSystemLibrary = (library) => SYSTEM_PREFIXES.some((prefix) => library.startsWith(prefix));

/**
 * The dependencies of one Mach-O file that nothing provides, from the output of `otool -L` and `otool -l`. Resolves `{ unresolved }`
 * (`"arch: library"` for a fat file, the library alone for a thin one) or `{ error }` when the output has no slice, a line that is not a dependency, or a slice without a library.
 */
export function unresolvedDependencies({ file, listing, loadCommands, root, executableDir, fsApi = fs }) {
    const libraries = parseOtoolL(listing);
    const commands = parseOtoolLoadCommands(loadCommands);
    if (libraries.length === 0) return { error: 'otool -L printed nothing readable' };
    const unresolved = [];
    for (const [index, slice] of libraries.entries()) {
        // A line of the listing that is not a dependency, or a slice with none (every dynamically linked file lists libSystem at least),
        // means the output was not understood: that is an error for the file, not a pass.
        if (slice.unreadable.length) return { error: `otool -L printed a line that is not a dependency${slice.arch ? ` (${slice.arch})` : ''}: ${slice.unreadable[0].slice(0, 120)}` };
        if (slice.libraries.length === 0) return { error: `otool -L listed no library${slice.arch ? ` for ${slice.arch}` : ''}` };
        const { rpaths, installName } = commands[index] ?? { rpaths: [], installName: null };
        let skippedInstallName = false;
        for (const library of slice.libraries) {
            // The first line of `otool -L` for a dylib is its own install name.
            if (!skippedInstallName && installName !== null && library === installName) { skippedInstallName = true; continue; }
            if (isSystemLibrary(library)) continue;
            if (resolveDependency(library, { file, rpaths, root, executableDir, fsApi })) continue;
            unresolved.push(slice.arch ? `${slice.arch}: ${library}` : library);
        }
    }
    return { unresolved };
}

/**
 * Checks every Mach-O file under `dir`. `executable` is the launched program, relative to `dir` (@executable_path is its directory;
 * default `dir`). `otool(flag, file)` is the seam for tests and resolves `{ output }` or `{ error }`. Resolves
 * `{ ok, checked, unresolved, errors, problems }` like elf.mjs's check: `unresolved` maps a file (relative to `dir`) to the libraries
 * nothing supplies. No Mach-O file at all fails (an empty or wrongly stripped copy has nothing to prove).
 */
export function strictLibraryCheck(dir, { otool = runOtool, fsApi = fs, executable = null } = {}) {
    const files = findMachoFiles(dir, { fsApi });
    const root = fsApi.realpathSync(dir);
    const executableDir = executable ? path.dirname(path.join(root, executable)) : root;
    const unresolved = {};
    const errors = {};
    for (const file of files) {
        const relative = path.relative(dir, file);
        const real = path.join(root, relative);
        const listing = otool('-L', file);
        const loadCommands = listing.error ? listing : otool('-l', file);
        if (listing.error || loadCommands.error) { errors[relative] = listing.error || loadCommands.error; continue; }
        const answer = unresolvedDependencies({ file: real, listing: listing.output, loadCommands: loadCommands.output, root, executableDir, fsApi });
        if (answer.error) errors[relative] = answer.error;
        else if (answer.unresolved.length) unresolved[relative] = answer.unresolved;
    }
    const problems = [];
    if (files.length === 0) problems.push('the copy holds no Mach-O file');
    if (Object.keys(unresolved).length) problems.push(`${Object.keys(unresolved).length} Mach-O files need libraries that are not found: ${Object.entries(unresolved).map(([f, l]) => `${f} (${l.join(', ')})`).join('; ')}`);
    if (Object.keys(errors).length) problems.push(`otool could not check ${Object.keys(errors).length} Mach-O files: ${Object.entries(errors).map(([f, e]) => `${f} (${e})`).join('; ')}`);
    return { ok: problems.length === 0, checked: files.length, driverOnly: [], unresolved, errors, problems };
}

/** The `ldd` replacement for the engine's install check on macOS: it has none, and the strict check above is the library proof. */
export const noLdd = () => [];
