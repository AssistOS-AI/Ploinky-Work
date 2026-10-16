// The macOS readers (lib/local-llm-darwin): the snapshot, memory and pressure figures and the process identity, from
// output captured on a 16 GiB Apple M1 (16 KiB pages, llama.cpp b11295) and from fakes. Nothing here runs a system
// tool, so the file runs on Linux and on macOS alike. Expected numbers are written out, not computed by the parser's
// own formula.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
    createDarwinSnapshotReader,
    parseMetalDevices,
    parsePsIdentity,
    parseSwapFree,
    parseSysctl,
    parseVmStat,
    readDarwinMemory,
    readDarwinPressure,
    readDarwinSnapshot,
    readProcessIdentity,
} from '../../lib/local-llm-darwin/darwin.mjs';

const fixture = (name) => fs.readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8');
const VM_STAT = fixture('vm_stat.txt');
const VM_STAT_4K = fixture('vm_stat-4k.txt');
// What `sysctl hw.memsize kern.memorystatus_level ... vm.swapusage` prints on that Mac.
const SYSCTL = `${fixture('sysctl.txt')}${fixture('swapusage.txt')}`;

// The captured figures, written out.
const TOTAL = 17_179_869_184;
const RECLAIMABLE = 2_889_121_792; // (6235 + 165709 + 4376 + 18) pages x 16384
const MEMORYSTATUS_34 = 5_841_155_522; // floor(17179869184 x 34 / 100)
const FREE = 102_154_240; // 6235 pages x 16384
const SWAP_FREE = 872_939_520; // 832.50 MiB
const MTL0_TOTAL = 12_712_935_424; // 12124 MiB

// A stand-in for execFileSync: `sysctl` answers the keys of `sysctlText` and fails (non-zero exit, partial output) for
// the others, as macOS does for an unknown key; `vm_stat` and `ps` answer the given text, or throw when it is null.
function fakeExec({ sysctlText = SYSCTL, vmStat = VM_STAT, ps = null, sysctlFails = false } = {}) {
    const calls = [];
    const exec = (file, args, options) => {
        calls.push({ file, args: [...args], options });
        if (file === '/usr/sbin/sysctl') {
            if (sysctlFails) throw new Error('no sysctl');
            const known = new Map(sysctlText.split('\n').filter(Boolean).map((line) => [line.slice(0, line.indexOf(':')), line]));
            const lines = args.filter((key) => known.has(key)).map((key) => known.get(key));
            if (lines.length < args.length) throw Object.assign(new Error('unknown oid'), { stdout: `${lines.join('\n')}\n` });
            return `${lines.join('\n')}\n`;
        }
        if (file === '/usr/bin/vm_stat') {
            if (vmStat === null) throw new Error('no vm_stat');
            return vmStat;
        }
        if (file === '/bin/ps') {
            if (ps === null) throw Object.assign(new Error('no such process'), { status: 1, stdout: '' });
            return ps;
        }
        throw new Error(`unexpected tool ${file}`);
    };
    return { exec, calls };
}

const withKeys = (replacements) => SYSCTL.split('\n').map((line) => {
    const key = line.slice(0, line.indexOf(':'));
    return Object.hasOwn(replacements, key) ? (replacements[key] === null ? '' : `${key}: ${replacements[key]}`) : line;
}).join('\n');

test('vm_stat: the 16 KiB capture parses to the captured page counts, and the 4 KiB one to the same bytes', () => {
    assert.deepEqual(parseVmStat(VM_STAT), { pageSize: 16384, freePages: 6235, inactivePages: 165709, speculativePages: 4376, purgeablePages: 18 });
    const small = parseVmStat(VM_STAT_4K);
    assert.deepEqual(small, { pageSize: 4096, freePages: 24940, inactivePages: 662836, speculativePages: 17504, purgeablePages: 72 });
    const bytes = (vm) => (vm.freePages + vm.inactivePages + vm.speculativePages + vm.purgeablePages) * vm.pageSize;
    assert.equal(bytes(parseVmStat(VM_STAT)), RECLAIMABLE);
    assert.equal(bytes(small), RECLAIMABLE, 'a page size of 4096 gives the same bytes from four times the pages');
});

test('vm_stat: a header without a sane page size, or a missing count, is not parsed', () => {
    for (const text of ['', 'garbage', null, undefined, VM_STAT.replace('page size of 16384', 'page size of 0'),
        VM_STAT.replace('page size of 16384', 'page size of 3000'), VM_STAT.replace(/\(page size of \d+ bytes\)/, ''),
        VM_STAT.replace(/^Pages purgeable:.*$/m, ''), VM_STAT.replace(/^Pages free:.*$/m, 'Pages free: many.'),
        VM_STAT.replace(/^Pages inactive:.*$/m, 'Pages inactive: -5.')]) {
        assert.equal(parseVmStat(text), null, String(text).slice(0, 60));
    }
});

test('sysctl, swap and Metal device captures parse to the captured figures', () => {
    const values = parseSysctl(SYSCTL);
    assert.equal(values['hw.memsize'], '17179869184');
    assert.equal(values['hw.pagesize'], '16384');
    assert.equal(values['kern.memorystatus_level'], '34');
    assert.equal(values['kern.memorystatus_vm_pressure_level'], '2');
    assert.equal(values['hw.perflevel0.physicalcpu'], '4');
    assert.equal(values['iogpu.wired_limit_mb'], '0');
    assert.equal(parseSwapFree(values['vm.swapusage']), SWAP_FREE);
    assert.equal(parseSwapFree('total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)'), 0);
    assert.equal(parseSwapFree('free = 2.00G'), 2 * 1024 ** 3);
    assert.equal(parseSwapFree('nothing'), null);
    assert.deepEqual(parseMetalDevices(fixture('list-devices.txt')), { name: 'Apple M1', totalBytes: MTL0_TOTAL, freeBytes: 12123 * 1024 * 1024 });
    for (const text of ['', 'Available devices:\n  BLAS: Accelerate (0 MiB, 0 MiB free)\n', 'MTL0: Apple M1 (many MiB, 1 MiB free)', 'MTL0: Apple M1 (10 MiB, 11 MiB free)', null]) {
        assert.equal(parseMetalDevices(text), null, String(text));
    }
});

test('readDarwinMemory: the captured Mac gives the captured figures, and the tools run by absolute path with a fixed locale', () => {
    const { exec, calls } = fakeExec();
    assert.deepEqual(readDarwinMemory({ exec }), {
        totalBytes: TOTAL, availableBytes: RECLAIMABLE, freeBytes: FREE, swapFreeBytes: SWAP_FREE, cachedBytes: null, pressureLevel: 2,
    });
    assert.deepEqual(calls.map((call) => [call.file, call.args]), [
        ['/usr/sbin/sysctl', ['hw.memsize', 'kern.memorystatus_level', 'kern.memorystatus_vm_pressure_level', 'vm.swapusage']],
        ['/usr/bin/vm_stat', []],
    ]);
    for (const call of calls) {
        assert.equal(call.options.env.LC_ALL, 'C');
        assert.equal(call.options.env.TZ, 'UTC');
        assert.equal(Number.isFinite(call.options.timeout), true);
    }
});

test('readDarwinMemory: 4 KiB pages give the same bytes, and the smaller of the two estimators wins', () => {
    assert.equal(readDarwinMemory({ exec: fakeExec({ vmStat: VM_STAT_4K }).exec }).availableBytes, RECLAIMABLE);
    // The kernel's own percentage is the smaller one here: 34 % of the memory is 5.8 GB, and reclaimable pages are 2.9 GB, so the pages win ...
    assert.equal(MEMORYSTATUS_34 > RECLAIMABLE, true);
    // ... and at 10 % the percentage (1,717,986,918) is below the pages.
    assert.equal(readDarwinMemory({ exec: fakeExec({ sysctlText: withKeys({ 'kern.memorystatus_level': '10' }) }).exec }).availableBytes, 1_717_986_918);
    assert.equal(readDarwinMemory({ exec: fakeExec({ sysctlText: withKeys({ 'kern.memorystatus_level': '100' }) }).exec }).availableBytes, RECLAIMABLE);
});

test('readDarwinMemory: anything unreadable gives null figures and never throws', () => {
    const none = { totalBytes: null, availableBytes: null, freeBytes: null, swapFreeBytes: null, cachedBytes: null, pressureLevel: null };
    assert.deepEqual(readDarwinMemory({ exec: fakeExec({ sysctlFails: true, vmStat: null }).exec }), none);
    assert.deepEqual(readDarwinMemory({ exec: () => { throw new Error('boom'); } }), none);
    assert.deepEqual(readDarwinMemory({ exec: () => 'garbage' }), none);
    // Garbage vm_stat: no available memory (admission refuses), the rest still read.
    const garbage = readDarwinMemory({ exec: fakeExec({ vmStat: 'Mach Virtual Memory Statistics: ???' }).exec });
    assert.equal(garbage.availableBytes, null);
    assert.equal(garbage.freeBytes, null);
    assert.equal(garbage.totalBytes, TOTAL);
    assert.equal(garbage.pressureLevel, 2);
    // A missing or malformed kernel percentage, memory size or pressure level.
    assert.equal(readDarwinMemory({ exec: fakeExec({ sysctlText: withKeys({ 'kern.memorystatus_level': null }) }).exec }).availableBytes, null);
    assert.equal(readDarwinMemory({ exec: fakeExec({ sysctlText: withKeys({ 'kern.memorystatus_level': '101' }) }).exec }).availableBytes, null);
    assert.equal(readDarwinMemory({ exec: fakeExec({ sysctlText: withKeys({ 'hw.memsize': 'lots' }) }).exec }).availableBytes, null);
    assert.equal(readDarwinMemory({ exec: fakeExec({ sysctlText: withKeys({ 'kern.memorystatus_vm_pressure_level': '3' }) }).exec }).pressureLevel, null);
    assert.equal(readDarwinMemory({ exec: fakeExec({ sysctlText: withKeys({ 'vm.swapusage': 'unknown' }) }).exec }).swapFreeBytes, null);
});

test('readDarwinPressure: 1, 2 and 4 are levels, every other answer is null', () => {
    for (const level of [1, 2, 4]) {
        assert.equal(readDarwinPressure({ exec: fakeExec({ sysctlText: withKeys({ 'kern.memorystatus_vm_pressure_level': String(level) }) }).exec }), level);
    }
    for (const level of ['0', '3', '5', '', 'x', '-1', '2.5', null]) {
        assert.equal(readDarwinPressure({ exec: fakeExec({ sysctlText: withKeys({ 'kern.memorystatus_vm_pressure_level': level }) }).exec }), null, String(level));
    }
    assert.equal(readDarwinPressure({ exec: fakeExec({ sysctlFails: true }).exec }), null);
    const { exec, calls } = fakeExec();
    assert.equal(readDarwinPressure({ exec }), 2);
    assert.deepEqual(calls.map((call) => [call.file, call.args]), [['/usr/sbin/sysctl', ['kern.memorystatus_vm_pressure_level']]]);
});

const statfs = async () => ({ bsize: 4096, blocks: 1_000_000, bavail: 250_000 });

test('readDarwinSnapshot: the captured Mac gives the engine\'s snapshot shape, with no Linux-only keys', async () => {
    const probes = [];
    const snapshot = await readDarwinSnapshot({
        exec: fakeExec().exec, statfs, dataDir: '/home/x/data', cpus: 8, now: () => new Date('2026-10-16T05:00:00Z'),
        probeDevice: async () => { probes.push(1); return fixture('list-devices.txt'); },
    });
    assert.deepEqual(snapshot, {
        at: '2026-10-16T05:00:00.000Z',
        gpu: { available: true, backend: 'metal', name: 'Apple M1', memoryModel: 'unified', deviceCapBytes: MTL0_TOTAL },
        memory: { totalBytes: TOTAL, availableBytes: RECLAIMABLE, freeBytes: FREE, swapFreeBytes: SWAP_FREE, cachedBytes: null, pressureLevel: 2 },
        disk: { freeBytes: 250_000 * 4096, totalBytes: 1_000_000 * 4096 },
        cpus: 8,
        cores: 4,
    });
    assert.equal(Object.hasOwn(snapshot, 'cgroupMemory'), false);
    assert.equal(probes.length, 1);
});

test('readDarwinSnapshot: a Mac without performance levels uses its physical cores, and an unreadable disk is a null figure', async () => {
    const intel = withKeys({ 'hw.perflevel0.physicalcpu': null });
    const snapshot = await readDarwinSnapshot({ exec: fakeExec({ sysctlText: intel }).exec, statfs: async () => { throw new Error('EIO'); }, dataDir: '/x', cpus: 8 });
    assert.equal(snapshot.cores, 8);
    assert.deepEqual(snapshot.disk, { freeBytes: null, totalBytes: null, error: 'EIO' });
    // Without a probe (the runner is not installed) the device limit is null and admission refuses; nothing is invented.
    assert.deepEqual(snapshot.gpu, { available: true, backend: 'metal', name: null, memoryModel: 'unified', deviceCapBytes: null });
    const neither = await readDarwinSnapshot({ exec: fakeExec({ sysctlText: withKeys({ 'hw.perflevel0.physicalcpu': null, 'hw.physicalcpu': null }) }).exec, statfs, dataDir: '/x', cpus: 2 });
    assert.equal(neither.cores, null);
});

test('readDarwinSnapshot: a device list without Metal makes the GPU unavailable with a reason', async () => {
    const snapshot = await readDarwinSnapshot({ exec: fakeExec().exec, statfs, dataDir: '/x', cpus: 8, probeDevice: async () => 'Available devices:\n  BLAS: Accelerate (0 MiB, 0 MiB free)\n' });
    assert.equal(snapshot.gpu.available, false);
    assert.equal(snapshot.gpu.deviceCapBytes, null);
    assert.match(snapshot.gpu.reason, /no Metal device/);
});

test('the snapshot reader probes the device once, again when iogpu.wired_limit_mb changes, and keeps asking while no runner is installed', async () => {
    let wired = '0';
    const exec = (file, args, options) => fakeExec({ sysctlText: withKeys({ 'iogpu.wired_limit_mb': wired }) }).exec(file, args, options);
    let probes = 0;
    const read = createDarwinSnapshotReader({ exec, statfs, dataDir: '/x', cpus: 8, probeDevice: async () => { probes += 1; return fixture('list-devices.txt'); } });
    await read();
    await read();
    assert.equal(probes, 1);
    wired = '10240';
    assert.equal((await read()).gpu.deviceCapBytes, MTL0_TOTAL);
    assert.equal(probes, 2);
    await read();
    assert.equal(probes, 2);
    let asked = 0;
    const waiting = createDarwinSnapshotReader({ exec: fakeExec().exec, statfs, dataDir: '/x', cpus: 8, probeDevice: async () => { asked += 1; return null; } });
    await waiting();
    await waiting();
    assert.equal(asked, 2, 'a missing runner is not remembered');
    const failing = createDarwinSnapshotReader({ exec: fakeExec().exec, statfs, dataDir: '/x', cpus: 8, probeDevice: async () => { throw new Error('spawn failed'); } });
    assert.equal((await failing()).gpu.deviceCapBytes, null);
});

test('readDarwinSnapshot: a stop before or during the probe rejects with an abort', async () => {
    const stop = new AbortController();
    stop.abort();
    await assert.rejects(readDarwinSnapshot({ exec: fakeExec().exec, statfs, dataDir: '/x', signal: stop.signal }), { name: 'AbortError' });
    const during = new AbortController();
    await assert.rejects(readDarwinSnapshot({ exec: fakeExec().exec, statfs, dataDir: '/x', signal: during.signal,
        probeDevice: async () => { during.abort(); throw new Error('killed'); } }), { name: 'AbortError' });
});

test('ps identity: the captured line gives the start time as UTC and the command', () => {
    const identity = parsePsIdentity(fixture('ps.txt'));
    assert.equal(identity.startedAt, '2026-10-16T03:42:14.000Z');
    assert.equal(identity.startMs, Date.UTC(2026, 9, 16, 3, 42, 14));
    assert.match(identity.command, /^bash \/Users\/example\/work\/capture-fixtures\.sh$/);
    // A day below ten is space padded, and the command keeps its own spaces.
    assert.deepEqual(parsePsIdentity('Sat Oct  3 00:00:05 2026     /usr/bin/python3 -m  http.server 8000\n'),
        { startMs: Date.UTC(2026, 9, 3, 0, 0, 5), startedAt: '2026-10-03T00:00:05.000Z', command: '/usr/bin/python3 -m  http.server 8000' });
    for (const text of ['', null, 'garbage', 'Fri Oct 16 03:42:14 2026', 'Fri Feb 31 03:42:14 2026     x', 'Fri Foo 16 03:42:14 2026     x', 'Fri Oct 16 25:42:14 2026     x']) {
        assert.equal(parsePsIdentity(text), null, String(text));
    }
});

test('readProcessIdentity runs ps by absolute path with TZ=UTC and LC_ALL=C every time, and refuses a pid that is not a positive integer', () => {
    const { exec, calls } = fakeExec({ ps: fixture('ps.txt') });
    assert.deepEqual(readProcessIdentity(4242, { exec }), {
        pid: 4242, startMs: Date.UTC(2026, 9, 16, 3, 42, 14), startedAt: '2026-10-16T03:42:14.000Z',
        command: 'bash /Users/example/work/capture-fixtures.sh',
    });
    assert.equal(calls[0].file, '/bin/ps');
    assert.deepEqual(calls[0].args, ['-o', 'lstart=,command=', '-p', '4242']);
    assert.equal(calls[0].options.env.TZ, 'UTC');
    assert.equal(calls[0].options.env.LC_ALL, 'C');
    const before = calls.length;
    for (const pid of [0, -1, 1.5, NaN, '12', '12; rm -rf /', null, undefined, 2 ** 60]) assert.equal(readProcessIdentity(pid, { exec }), null, String(pid));
    assert.equal(calls.length, before, 'no tool ran for a pid that is not valid');
    // A process that is gone (ps exits 1 with no output), and a tool that cannot run.
    assert.equal(readProcessIdentity(4242, { exec: fakeExec().exec }), null);
    assert.equal(readProcessIdentity(4242, { exec: () => { throw new Error('ENOENT'); } }), null);
});
