// The macOS readers of a standalone host (DS005, DS007): the hardware snapshot the engine admits against, the memory
// and pressure readers its guard samples, and the process identity a restart compares. Everything is read with the
// system tools by absolute path (`sysctl`, `vm_stat`, `ps`), through an `exec(file, args, options) -> text` function
// that tests replace. A figure that cannot be parsed is null, never a default, so admission refuses.
//
// This directory imports nothing from lib/local-llm and nothing in lib/local-llm imports it: the engine takes these
// readers as seams (`readHardwareSnapshot`, `readMemory`, `readPressure`), and the directory can move to a host as it is.

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';

const MIB = 1024 * 1024;
const SYSCTL = '/usr/sbin/sysctl';
const VM_STAT = '/usr/bin/vm_stat';
const PS = '/bin/ps';
const TOOL_TIMEOUT_MS = 3000;
// The tools' output is parsed as English text in one time zone, whatever the user's locale is.
const TOOL_ENV = Object.freeze({ PATH: '/usr/bin:/bin:/usr/sbin:/sbin', LC_ALL: 'C', LANG: 'C', TZ: 'UTC' });
// kern.memorystatus_vm_pressure_level: normal, warning, critical.
const PRESSURE_LEVELS = Object.freeze([1, 2, 4]);
const MONTHS = Object.freeze(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']);

// The output of a tool, or null when it cannot run. A tool that exits non-zero still prints what it knows (`sysctl`
// does for the keys it has), and the parsers below take only the lines they can read.
function run(exec, file, args) {
    try {
        return String(exec(file, args, { encoding: 'utf8', timeout: TOOL_TIMEOUT_MS, env: TOOL_ENV }));
    } catch (error) {
        return typeof error?.stdout === 'string' ? error.stdout : Buffer.isBuffer(error?.stdout) ? error.stdout.toString('utf8') : null;
    }
}

function safeInteger(text) {
    return /^\d+$/.test(String(text)) && Number.isSafeInteger(Number(text)) ? Number(text) : null;
}

/** `name: value` lines (the output of `sysctl a b c`) as an object of strings; a line that is not one is skipped. */
export function parseSysctl(text) {
    const values = Object.create(null);
    for (const line of String(text ?? '').split('\n')) {
        const match = /^([A-Za-z0-9_.]+):\s*(.*?)\s*$/.exec(line);
        if (match) values[match[1]] = match[2];
    }
    return values;
}

/**
 * The page counts and page size of `vm_stat`. Returns null unless the header names a page size and all four counts the
 * available-memory estimate needs are present: free, inactive, speculative and purgeable.
 */
export function parseVmStat(text) {
    const source = String(text ?? '');
    const header = /\(page size of (\d+) bytes\)/.exec(source);
    const pageSize = header ? safeInteger(header[1]) : null;
    if (pageSize === null || pageSize < 1024 || pageSize > 1024 * 1024 || (pageSize & (pageSize - 1)) !== 0) return null;
    const count = (label) => {
        const match = new RegExp(`^Pages ${label}:\\s+(\\d+)\\.\\s*$`, 'm').exec(source);
        return match ? safeInteger(match[1]) : null;
    };
    const free = count('free');
    const inactive = count('inactive');
    const speculative = count('speculative');
    const purgeable = count('purgeable');
    if ([free, inactive, speculative, purgeable].includes(null)) return null;
    return { pageSize, freePages: free, inactivePages: inactive, speculativePages: speculative, purgeablePages: purgeable };
}

/** The free bytes of `vm.swapusage` (`total = 14336.00M  used = 13503.50M  free = 832.50M  (encrypted)`), or null. */
export function parseSwapFree(text) {
    const match = /\bfree = (\d+(?:\.\d+)?)([KMG])\b/.exec(String(text ?? ''));
    if (!match) return null;
    const unit = { K: 1024, M: MIB, G: 1024 * MIB }[match[2]];
    const bytes = Math.round(Number(match[1]) * unit);
    return Number.isSafeInteger(bytes) ? bytes : null;
}

/**
 * The first Metal device of `llama-server --list-devices` (`  MTL0: Apple M1 (12124 MiB, 12123 MiB free)`):
 * { name, totalBytes, freeBytes }, or null when there is no such line.
 */
export function parseMetalDevices(text) {
    const match = /^\s*MTL0:\s*(.+?)\s*\((\d+) MiB,\s*(\d+) MiB free\)\s*$/m.exec(String(text ?? ''));
    if (!match) return null;
    const total = safeInteger(match[2]);
    const free = safeInteger(match[3]);
    if (total === null || free === null || total <= 0 || free > total) return null;
    return { name: match[1], totalBytes: total * MIB, freeBytes: free * MIB };
}

/**
 * The reclaimable memory of the page counts and the kernel's own percentage, as the smaller of the two (the design's
 * provisional estimator): min(total x level / 100, (free + inactive + speculative + purgeable) x page size).
 * Null when either is unknown.
 */
function availableFrom(totalBytes, level, vm) {
    if (totalBytes === null || level === null || vm === null) return null;
    const reclaimable = (vm.freePages + vm.inactivePages + vm.speculativePages + vm.purgeablePages) * vm.pageSize;
    return Number.isSafeInteger(reclaimable) ? Math.min(Math.floor(totalBytes * level / 100), reclaimable) : null;
}

function levelOf(values) {
    const level = safeInteger(values['kern.memorystatus_level']);
    return level !== null && level <= 100 ? level : null;
}

function pressureOf(values) {
    const level = safeInteger(values['kern.memorystatus_vm_pressure_level']);
    return PRESSURE_LEVELS.includes(level) ? level : null;
}

/**
 * The memory figures the engine sizes against and the guard samples. Never throws: every figure that cannot be read is
 * null. `pressureLevel` is 1 (normal), 2 (warning) or 4 (critical).
 */
export function readDarwinMemory({ exec = execFileSync } = {}) {
    const values = parseSysctl(run(exec, SYSCTL, ['hw.memsize', 'kern.memorystatus_level', 'kern.memorystatus_vm_pressure_level', 'vm.swapusage']));
    const vm = parseVmStat(run(exec, VM_STAT, []));
    const totalBytes = safeInteger(values['hw.memsize']);
    return {
        totalBytes,
        availableBytes: availableFrom(totalBytes, levelOf(values), vm),
        freeBytes: vm ? vm.freePages * vm.pageSize : null,
        swapFreeBytes: parseSwapFree(values['vm.swapusage']),
        cachedBytes: null,
        pressureLevel: pressureOf(values),
    };
}

/** The kernel's memory pressure level (1, 2 or 4), or null. Never throws. */
export function readDarwinPressure({ exec = execFileSync } = {}) {
    return pressureOf(parseSysctl(run(exec, SYSCTL, ['kern.memorystatus_vm_pressure_level'])));
}

// One optional sysctl key as a positive integer, or null (an Intel Mac has no performance-level keys).
function optionalCount(exec, name) {
    const value = safeInteger(parseSysctl(run(exec, SYSCTL, [name]))[name]);
    return value !== null && value > 0 ? value : null;
}

async function readDisk(dataDir, statfs) {
    try {
        const stats = await statfs(dataDir);
        return { freeBytes: Number(stats.bavail) * Number(stats.bsize), totalBytes: Number(stats.blocks) * Number(stats.bsize) };
    } catch (error) {
        return { freeBytes: null, totalBytes: null, error: error.message };
    }
}

function stopped() {
    return Object.assign(new Error('Stopped'), { name: 'AbortError', code: 'ABORT_ERR' });
}

/**
 * The hardware snapshot of an Apple Silicon Mac, in the shape of the engine's own:
 *
 *   { at, gpu: { available, backend: 'metal', name, memoryModel: 'unified', deviceCapBytes | null },
 *     memory, disk, cpus, cores }
 *
 * The GPU's limit comes from `probeDevice({ signal })`, which runs `<runner>/llama-server --list-devices` and returns
 * its text, or null while the runner is not installed (the limit is then null and admission refuses). The answer is
 * kept in `deviceCache` (an object the caller keeps) and read again only when `iogpu.wired_limit_mb` changes.
 * `cores` is the performance cores, or the physical cores where the Mac has no such level.
 */
export async function readDarwinSnapshot({ exec = execFileSync, statfs = (target) => fs.promises.statfs(target), dataDir, probeDevice = null,
    signal, deviceCache = null, cpus = os.availableParallelism?.() ?? os.cpus().length, now = () => new Date() } = {}) {
    if (signal?.aborted) throw stopped();
    const pendingDisk = readDisk(dataDir, statfs);
    const wired = parseSysctl(run(exec, SYSCTL, ['iogpu.wired_limit_mb']))['iogpu.wired_limit_mb'] ?? null;
    let device = deviceCache && deviceCache.device && deviceCache.wired === wired ? deviceCache.device : undefined;
    if (device === undefined && probeDevice) {
        let text = null;
        try {
            text = await probeDevice({ signal });
        } catch {
            if (signal?.aborted) throw stopped();
        }
        // A runner that is not installed (no text) is not remembered: the next snapshot asks again.
        if (typeof text === 'string') {
            device = { found: parseMetalDevices(text) };
            if (deviceCache) Object.assign(deviceCache, { wired, device });
        }
    }
    const metal = device?.found ?? null;
    const gpu = device && !metal
        ? { available: false, backend: 'metal', name: null, memoryModel: 'unified', deviceCapBytes: null, reason: 'llama-server reported no Metal device.' }
        : { available: true, backend: 'metal', name: metal?.name ?? null, memoryModel: 'unified', deviceCapBytes: metal?.totalBytes ?? null };
    const disk = await pendingDisk;
    if (signal?.aborted) throw stopped();
    return {
        at: now().toISOString(),
        gpu,
        memory: readDarwinMemory({ exec }),
        disk,
        cpus,
        cores: optionalCount(exec, 'hw.perflevel0.physicalcpu') ?? optionalCount(exec, 'hw.physicalcpu'),
    };
}

/** A snapshot reader for the engine's `readHardwareSnapshot` seam: it keeps the device answer between snapshots. */
export function createDarwinSnapshotReader(options = {}) {
    const deviceCache = {};
    return ({ signal } = {}) => readDarwinSnapshot({ ...options, signal, deviceCache });
}

/**
 * The start time and command line of a `ps -o lstart=,command=` line, or null for anything else. `lstart` is
 * `Fri Oct 16 03:42:14 2026`; it is read as UTC because the tool is run with `TZ=UTC LC_ALL=C` (N2).
 */
export function parsePsIdentity(text) {
    const match = /^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) ([A-Z][a-z]{2})\s+(\d{1,2}) (\d\d):(\d\d):(\d\d) (\d{4})\s+(\S.*?)\s*$/m.exec(String(text ?? ''));
    if (!match) return null;
    const month = MONTHS.indexOf(match[1]);
    const [day, hour, minute, second, year] = [match[2], match[3], match[4], match[5], match[6]].map(Number);
    const startMs = Date.UTC(year, month, day, hour, minute, second);
    const date = new Date(startMs);
    if (month < 0 || date.getUTCFullYear() !== year || date.getUTCMonth() !== month || date.getUTCDate() !== day
        || date.getUTCHours() !== hour || date.getUTCMinutes() !== minute || date.getUTCSeconds() !== second) return null;
    return { startMs, startedAt: date.toISOString(), command: match[7] };
}

/**
 * Who a process id is now: { pid, startMs, startedAt, command }, or null when there is no such process or the output
 * is not understood. Runs `ps -o lstart=,command= -p <pid>` with `TZ=UTC LC_ALL=C`, so the answer does not depend on the locale.
 */
export function readProcessIdentity(pid, { exec = execFileSync } = {}) {
    if (!Number.isSafeInteger(pid) || pid <= 0) return null;
    const identity = parsePsIdentity(run(exec, PS, ['-o', 'lstart=,command=', '-p', String(pid)]));
    return identity ? { pid, ...identity } : null;
}
