// The operating-system strategies of macOS (`darwinPlatform`) and the choice
// between the platforms the engine knows (`platformFor`). A platform it does
// not know is refused, never guessed. See `linuxPlatform` in host.mjs for what
// each member is; the members below differ as follows.
//
//   openedPathOf, openatPath   macOS cannot say where a descriptor points, so the first rejects
//                              and the second throws; the binder uses `confirmOpened` instead
//   confirmOpened(fsApi, handle, expected, anchor)   the stat binder (DS002)
//   beforeCreate(fsApi, anchor, file)   refuses a store whose directories another user could change
//   untrustedRoots: false      workspace-shared roots are never walked
//   setOomScoreAdj             nothing to write
//   tarFlags                   `-m`: bsdtar rejects `--touch`
//   performanceCores()         the high-performance cores, or null (the runner then counts them itself)
//   socketPathMax: 103         sizeof(sockaddr_un.sun_path) - 1 on macOS (Linux: 107)
//   pressureScale: 'level'     the memory pressure it reads is the kernel's level (1, 2, 4), so the cpu profile is guarded
//                              like the metal one (an Intel Mac); a platform without the member reads a PSI share
//
// The members `confirmOpened`, `beforeCreate`, `untrustedRoots` and
// `performanceCores` are optional strategies: the download binder and the
// runners use them only when a platform has them, so a platform written for
// the Linux members alone keeps working.

import { execFileSync } from 'node:child_process';
import path from 'node:path';

import { LocalLlmError } from './errors.mjs';
import { checkCreateChain, confirmByStat } from './controller/darwinBinding.mjs';
import { linuxPlatform } from './host.mjs';

// The platform shape carries /proc and /dev/dri directories; macOS has neither, and these never exist, so a reader
// that tried them would find nothing and fail closed.
const NO_PROC_DIR = '/nonexistent/proc';
const NO_DRI_DIR = '/nonexistent/dri';
const SYSCTL = '/usr/sbin/sysctl';
// sizeof(sockaddr_un.sun_path) is 104 on macOS, one byte of it the terminator.
const SOCKET_PATH_MAX = 103;

function unsupported(what) {
    return new LocalLlmError('unsupported_platform', `${what} is not available on macOS: the download binder compares file identities instead (DS002).`);
}

// A positive integer printed by `sysctl -n <name>`, or null when the key is missing or the output is not one.
function sysctlCount(exec, name) {
    try {
        const text = String(exec(SYSCTL, ['-n', name], { encoding: 'utf8', timeout: 2000, env: { PATH: '/usr/bin:/bin', LC_ALL: 'C' } })).trim();
        const value = /^\d+$/.test(text) ? Number(text) : NaN;
        return Number.isSafeInteger(value) && value > 0 ? value : null;
    } catch {
        return null;
    }
}

/**
 * The macOS strategies. `privateRoot` is the directory the program keeps its
 * state in (null: none): the store is refused unless it and every directory
 * from there down are the user's own and not writable by others. `uid` and
 * `exec` are test seams (the user id, and a function that runs a program and
 * returns its output).
 */
export function darwinPlatform({ privateRoot = null, uid = process.getuid?.(), exec = execFileSync } = {}) {
    if (privateRoot !== null && (typeof privateRoot !== 'string' || !path.posix.isAbsolute(privateRoot))) {
        throw new LocalLlmError('invalid_host', `Invalid host platform.privateRoot: ${String(privateRoot)} is not an absolute path`, { field: 'platform.privateRoot' });
    }
    return Object.freeze({
        name: 'darwin',
        procDir: NO_PROC_DIR,
        devDriDir: NO_DRI_DIR,
        socketPathMax: SOCKET_PATH_MAX,
        pressureScale: 'level',
        openedPathOf: async () => { throw unsupported('Reading where an open descriptor points'); },
        openatPath: () => { throw unsupported('Opening a name through a directory descriptor'); },
        unreadableReason: 'macOS has no /proc/self/fd',
        setOomScoreAdj: () => {},
        tarFlags: (into, strip) => ['-C', into, `--strip-components=${strip}`, '--no-same-owner', '-m'],
        confirmOpened: (fsApi, handle, expected, anchor) => confirmByStat(fsApi, handle, expected, anchor),
        beforeCreate: (fsApi, anchor, file) => checkCreateChain(fsApi, anchor, file, { uid, privateRoot }),
        untrustedRoots: false,
        performanceCores: () => sysctlCount(exec, 'hw.perflevel0.physicalcpu'),
    });
}

/**
 * The platform for a Node `process.platform` name: `linux` and `darwin`; any
 * other name throws `unsupported_platform` (the engine fails closed rather
 * than run a binder it has not been written for). `options` go to the
 * platform's constructor.
 */
export function platformFor(name = process.platform, options = {}) {
    if (name === 'linux') return linuxPlatform(options);
    if (name === 'darwin') return darwinPlatform(options);
    throw new LocalLlmError('unsupported_platform', `The local-llm engine has no download binder for the platform '${String(name)}'; it supports linux and darwin.`, { platform: String(name) });
}
