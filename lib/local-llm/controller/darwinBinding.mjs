// The download binder of a platform that cannot ask the kernel where an open
// descriptor points (macOS has no /proc/self/fd, DS002). It is weaker than the
// Linux one and says so: after a no-follow open it compares what the handle is
// with what the path is now, by device and inode, at every level.
//
// `confirmByStat` runs after every open. The handle must be a regular file;
// the anchored root must still be the directory it was anchored as; every
// directory between the root and the file must be a real directory (not a
// link); and the final path must be a regular file, not a link, with the
// handle's own device and inode. A swap that happened before the check makes
// one of those differ and the file is not bound.
//
// The gap that remains: a same-user process that swaps a directory and puts
// the original back between this check and the next use of the descriptor is
// not seen. `checkCreateChain` narrows what that process can do before a file
// is created: every directory from the root down to the parent must be owned
// by the user and writable by nobody else, so only the user's own processes
// are in that position. Untrusted shared roots are off on such a platform.

import path from 'node:path';

import { DownloadError } from './downloadError.mjs';

function unsafePath(message) {
    return new DownloadError('UNSAFE_PATH', message);
}

// An error that says the path is not there (any more): the opened file is not what the path names.
const GONE_CODES = ['ENOENT', 'ENOTDIR'];

async function lstatOrNull(fsApi, target) {
    try {
        return await fsApi.promises.lstat(target);
    } catch (error) {
        if (GONE_CODES.includes(error.code)) return null;
        throw unsafePath(`Cannot confirm which file was opened (${error.code || 'error'})`);
    }
}

// The path of `expected` below the anchored real root, as its directory names and the file name; null when it is not below it.
function partsBelow(anchor, expected) {
    const relative = path.relative(anchor.real, expected);
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) return null;
    return relative.split(path.sep);
}

/**
 * Whether the opened `handle` is the regular file at `expected`, a path below
 * `anchor.real`: the stat binder described above. Returns false when any
 * identity differs or a level is missing or is a link; throws UNSAFE_PATH when
 * a level cannot be read at all.
 */
export async function confirmByStat(fsApi, handle, expected, anchor) {
    if (!anchor) throw unsafePath('Cannot confirm which file was opened (no anchored root)');
    const opened = await handle.stat();
    if (!opened.isFile()) return false;
    const parts = partsBelow(anchor, expected);
    if (!parts) return false;
    const root = await lstatOrNull(fsApi, anchor.real);
    if (!root || !root.isDirectory() || root.dev !== anchor.dev || root.ino !== anchor.ino) return false;
    let current = anchor.real;
    for (const part of parts.slice(0, -1)) {
        current = path.join(current, part);
        const level = await lstatOrNull(fsApi, current);
        if (!level || !level.isDirectory()) return false;
    }
    const final = await lstatOrNull(fsApi, expected);
    return Boolean(final) && final.isFile() && final.dev === opened.dev && final.ino === opened.ino;
}

/**
 * Before a file is created or a partial file is opened for appending (which
 * creates it): every directory from `privateRoot` (when it holds the anchored
 * root) or the anchored root down to the file's parent must be a real
 * directory owned by `uid`, with no group or other write bit. Refuses with
 * UNSAFE_PATH otherwise, before any file exists. A level that is missing is
 * left to the open, which fails on it by itself.
 */
export async function checkCreateChain(fsApi, anchor, file, { uid, privateRoot = null } = {}) {
    if (!Number.isInteger(uid)) throw unsafePath('Cannot check who owns the model store: the user id is unknown');
    const relative = path.relative(anchor.path, path.resolve(file));
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
        throw unsafePath('A model file path escapes its directory');
    }
    const chain = [];
    if (privateRoot !== null) {
        let root = null;
        try {
            root = await fsApi.promises.realpath(privateRoot);
        } catch {
            // A private root that is not there holds nothing: the anchored chain alone is checked.
        }
        const below = root === null ? null : path.relative(root, anchor.real);
        if (below !== null && !below.startsWith('..') && !path.isAbsolute(below)) {
            let current = root;
            chain.push(current);
            for (const part of below ? below.split(path.sep) : []) chain.push(current = path.join(current, part));
        }
    }
    if (chain.at(-1) !== anchor.real) chain.push(anchor.real);
    let current = anchor.real;
    for (const part of relative.split(path.sep).slice(0, -1)) chain.push(current = path.join(current, part));
    for (const directory of chain) {
        const stats = await lstatOrNull(fsApi, directory);
        if (!stats) return;
        if (!stats.isDirectory()) throw unsafePath(`${directory} is a symbolic link or not a directory; the model store cannot be used through it`);
        if (stats.uid !== uid) throw unsafePath(`${directory} is not owned by the user running this program`);
        if ((stats.mode & 0o022) !== 0) throw unsafePath(`${directory} can be written by its group or by others; remove that permission (chmod go-w)`);
    }
}
