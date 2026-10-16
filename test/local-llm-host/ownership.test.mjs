// Ownership of a home (plan: Ownership and recovery (1)): an exclusive loopback bind, held by exactly one process at every
// instant, released by the kernel when its holder dies, never reclaimed by anyone. The contenders are separate real processes
// (fixtures/contender.mjs); the same observer judges the real lock and a check-then-write variant that must be caught.
import '../home.mjs';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { LocalLlmError } from '../../lib/local-llm/errors.mjs';
import { OWNER_PORT_FIRST, OWNER_PORT_LAST, acquireOwnership, diagnoseHolder, ownerPortFor } from '../../lib/local-llm-host/ownership.mjs';

const CONTENDER = fileURLToPath(new URL('./fixtures/contender.mjs', import.meta.url));
const LIB = fileURLToPath(new URL('../../lib/local-llm-host', import.meta.url));

function scratchHome(t) {
    const dir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'own-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

function startContender(t, options) {
    const child = spawn(process.execPath, [CONTENDER, JSON.stringify(options)], { stdio: ['pipe', 'pipe', 'inherit'] });
    const events = [];
    const waiters = [];
    let buffer = '';
    child.stdout.on('data', (chunk) => {
        buffer += chunk;
        for (let end = buffer.indexOf('\n'); end >= 0; end = buffer.indexOf('\n')) {
            const event = JSON.parse(buffer.slice(0, end));
            buffer = buffer.slice(end + 1);
            events.push(event);
            for (const waiter of [...waiters]) waiter();
        }
    });
    const exited = new Promise((resolve) => child.once('exit', (code, signal) => resolve({ code, signal })));
    t.after(() => { try { child.kill('SIGKILL'); } catch {} });
    return {
        child, events, exited,
        pid: child.pid,
        go: () => child.stdin.write('go\n'),
        async event(name, timeoutMs = 10_000) {
            const deadline = Date.now() + timeoutMs;
            for (;;) {
                const found = events.find((event) => event.ev === name);
                if (found) return found;
                if (Date.now() > deadline) assert.fail(`no ${name} event from contender ${child.pid}`);
                await new Promise((resolve) => { const timer = setTimeout(resolve, 20); waiters.push(() => { clearTimeout(timer); resolve(); }); });
            }
        },
    };
}

/** The held intervals of a set of contenders, from their events: [{ pid, from, to }] in ns. */
function intervals(contenders) {
    const held = [];
    for (const { events } of contenders) {
        const acquired = events.find((event) => event.ev === 'acquired');
        const released = events.find((event) => event.ev === 'released');
        if (acquired) held.push({ pid: acquired.pid, from: BigInt(acquired.t), to: released ? BigInt(released.t) : null });
    }
    return held;
}

/** The most intervals that overlap at any instant: the number of owners at the worst observation point. */
function maxOwners(held) {
    const edges = held.flatMap(({ from, to }) => [[from, 1], [to ?? from + 10n ** 12n, -1]]);
    edges.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : a[1] - b[1]));
    let now = 0;
    let most = 0;
    for (const [, step] of edges) { now += step; most = Math.max(most, now); }
    return most;
}

const freePort = () => new Promise((resolve) => {
    const server = net.createServer();
    server.listen(0, '127.0.0.1', () => { const { port } = server.address(); server.close(() => resolve(port)); });
});

test('the owner port is a pure function of the home inside a range below both ephemeral ranges', () => {
    const ports = new Set();
    for (let index = 0; index < 500; index += 1) {
        const port = ownerPortFor(`/home/user${index}/.pworker`);
        assert.ok(port >= OWNER_PORT_FIRST && port <= OWNER_PORT_LAST);
        ports.add(port);
        assert.equal(ownerPortFor(`/home/user${index}/.pworker`), port);
    }
    assert.ok(OWNER_PORT_LAST < 32768, 'below the Linux ephemeral range (32768-60999) and the macOS one (49152-65535)');
    for (const reserved of [18080, 19611, 19613, 19614]) assert.ok(reserved < OWNER_PORT_FIRST || reserved > OWNER_PORT_LAST);
    assert.ok(ports.size > 400, 'homes spread over the range');
});

test('a second acquire of the same home is refused with the pid of the holder, and the port is free again after release', async () => {
    const realHome = `/pworker/home-${process.pid}-a`;
    const port = await freePort();
    const first = await acquireOwnership({ realHome, port });
    assert.equal(first.port, port);
    await assert.rejects(acquireOwnership({ realHome, port }), (error) => {
        assert.ok(error instanceof LocalLlmError);
        assert.equal(error.code, 'owner_held');
        assert.equal(error.details.pid, process.pid);
        assert.equal(error.details.kind, 'same-home');
        return true;
    });
    // The failed attempt did not take, move or free anything: the first holder still holds.
    assert.deepEqual((await diagnoseHolder({ port, realHome })), { kind: 'same-home', pid: process.pid });
    await first.release();
    // The diagnostic connections left closed sockets behind (TIME_WAIT on the owner side); the next bind is not blocked by them.
    const second = await acquireOwnership({ realHome, port });
    await second.release();
});

test('a pworker that owns another home on the same port is named as a collision, not as the owner of this home', async () => {
    const port = await freePort();
    const other = await acquireOwnership({ realHome: '/pworker/other-home', port });
    try {
        await assert.rejects(acquireOwnership({ realHome: '/pworker/this-home', port }), (error) => {
            assert.equal(error.code, 'owner_port_collision');
            assert.equal(error.details.home, '/pworker/other-home');
            assert.match(error.message, /another PWORKER_HOME/);
            return true;
        });
    } finally { await other.release(); }
});

test('a port held by a program that is not pworker is a clear refusal, whether it stays silent, talks or closes at once', async () => {
    for (const behaviour of ['silent', 'banner', 'close']) {
        const port = await freePort();
        const sockets = new Set();
        const server = net.createServer((socket) => {
            sockets.add(socket);
            socket.on('error', () => {});
            if (behaviour === 'banner') socket.write('SSH-2.0-OpenSSH_9.6\r\n');
            if (behaviour === 'close') socket.destroy();
        });
        await new Promise((resolve) => server.listen(port, '127.0.0.1', resolve));
        try {
            await assert.rejects(acquireOwnership({ realHome: '/pworker/unrelated', port }), (error) => {
                assert.equal(error.code, 'owner_port_taken', behaviour);
                assert.match(error.message, new RegExp(`Port ${port} on 127.0.0.1`));
                assert.match(error.message, /not pworker/);
                return true;
            });
        } finally { for (const socket of sockets) socket.destroy(); await new Promise((resolve) => server.close(resolve)); }
    }
});

test('a stale owner.json is informational: it is neither read nor moved, and the next owner writes over it', async (t) => {
    const home = scratchHome(t);
    const ownerFile = path.join(home, 'owner.json');
    fs.writeFileSync(ownerFile, JSON.stringify({ pid: 2 ** 22 + 1, port: 1, home: 'gone' }));
    const inode = fs.statSync(ownerFile).ino;
    const port = await freePort();
    const lock = await acquireOwnership({ realHome: home, port, ownerFile });
    try {
        const record = JSON.parse(fs.readFileSync(ownerFile, 'utf8'));
        assert.equal(record.pid, process.pid);
        assert.equal(record.port, port);
        assert.equal(fs.statSync(ownerFile).ino, inode, 'written in place, not replaced by a rename');
        // Someone else asking now is refused although the record names a dead pid: the record decides nothing, the bind does.
        fs.writeFileSync(ownerFile, JSON.stringify({ pid: 2 ** 22 + 1, port: 1, home: 'gone' }));
        await assert.rejects(acquireOwnership({ realHome: home, port, ownerFile }), { code: 'owner_held' });
        assert.equal(JSON.parse(fs.readFileSync(ownerFile, 'utf8')).pid, 2 ** 22 + 1, 'the refused contender did not touch the record');
    } finally { await lock.release(); }
});

test('four real processes race for one home: exactly one owner at every observation point, and each gets its turn', async (t) => {
    const home = scratchHome(t);
    const racers = [0, 1, 2, 3].map(() => startContender(t, { home, mode: 'bind', holdMs: 150, retryMs: 20_000, waitForGo: true }));
    await Promise.all(racers.map((racer) => racer.event('ready')));
    // Polling during the race: who answers on the port right now. A pid seen must belong to a contender that holds then.
    const realHome = fs.realpathSync(home);
    const port = ownerPortFor(realHome);
    const samples = [];
    let racing = true;
    const poller = (async () => {
        while (racing) {
            const from = process.hrtime.bigint();
            const seen = await diagnoseHolder({ port, realHome, timeoutMs: 500 });
            samples.push({ from, to: process.hrtime.bigint(), pid: seen.kind === 'same-home' ? seen.pid : null });
            await new Promise((resolve) => setTimeout(resolve, 3));
        }
    })();
    for (const racer of racers) racer.go();
    const exits = await Promise.all(racers.map((racer) => racer.exited));
    racing = false;
    await poller;
    assert.deepEqual(exits.map((exit) => exit.code), [0, 0, 0, 0]);
    const held = intervals(racers);
    assert.equal(held.length, 4, 'every contender eventually owned the home once');
    assert.equal(maxOwners(held), 1, 'never two owners at once');
    assert.equal(racers.flatMap((racer) => racer.events).filter((event) => event.ev === 'refused').length, 0);
    const slack = 60n * 1_000_000n;
    const seenPids = new Set();
    for (const sample of samples.filter((s) => s.pid !== null)) {
        seenPids.add(sample.pid);
        const owner = held.find((interval) => interval.pid === sample.pid);
        assert.ok(owner, `pid ${sample.pid} answered on the port but never reported holding it`);
        // The answer came at some instant between the start and the end of the sample: that span must meet the owner's held interval.
        assert.ok(sample.from <= owner.to + slack && sample.to >= owner.from - slack, `pid ${sample.pid} answered outside its held interval`);
    }
    assert.ok(seenPids.size >= 2, `the poller saw the handover (${seenPids.size} owners seen)`);
});

test('negative control: a check-then-write lock under the same harness is caught as two owners', async (t) => {
    const home = scratchHome(t);
    const racers = [0, 1].map(() => startContender(t, { home, mode: 'check-then-write', gapMs: 150, holdMs: 400, retryMs: 0, waitForGo: true }));
    await Promise.all(racers.map((racer) => racer.event('ready')));
    for (const racer of racers) racer.go();
    await Promise.all(racers.map((racer) => racer.exited));
    const held = intervals(racers);
    assert.equal(held.length, 2, 'both passed the check');
    assert.equal(maxOwners(held), 2, 'the observer sees them overlap');
});

test('nothing reclaims: a contender that is refused leaves the live owner running and owning', async (t) => {
    const home = scratchHome(t);
    const owner = startContender(t, { home, mode: 'bind', holdMs: -1, retryMs: 0 });
    await owner.event('acquired');
    const late = startContender(t, { home, mode: 'bind', holdMs: 0, retryMs: 0 });
    await late.event('refused');
    assert.deepEqual(await late.exited, { code: 0, signal: null });
    const realHome = fs.realpathSync(home);
    assert.deepEqual(await diagnoseHolder({ port: ownerPortFor(realHome), realHome }), { kind: 'same-home', pid: owner.pid });
    assert.doesNotThrow(() => process.kill(owner.pid, 0), 'the owner was not signalled');
    owner.child.kill('SIGKILL');
    await owner.exited;
});

test('after SIGKILL of the owner, with a runner-like child still alive, the next contender acquires at once with one attempt', async (t) => {
    const home = scratchHome(t);
    const owner = startContender(t, { home, mode: 'bind', holdMs: -1, retryMs: 0, child: true });
    await owner.event('acquired');
    const { child: childPid } = await owner.event('child');
    t.after(() => { try { process.kill(childPid, 'SIGKILL'); } catch {} });
    const realHome = fs.realpathSync(home);
    // A diagnostic connection first: its closed socket must not block the next owner either.
    assert.equal((await diagnoseHolder({ port: ownerPortFor(realHome), realHome })).pid, owner.pid);
    owner.child.kill('SIGKILL');
    assert.equal((await owner.exited).signal, 'SIGKILL');
    assert.doesNotThrow(() => process.kill(childPid, 0), 'the child outlived its owner');
    // One attempt, no retry: the kernel released the port with the process, although the child lives on. If the listener had
    // leaked into the child (no close-on-exec), the child would still hold the port and this would be refused.
    const next = startContender(t, { home, mode: 'bind', holdMs: 0, retryMs: 0 });
    await next.event('acquired');
    assert.deepEqual(await next.exited, { code: 0, signal: null });
});

test('negative control: a child that inherits the listener keeps the home owned after SIGKILL, and the same harness sees it', async (t) => {
    const home = scratchHome(t);
    const owner = startContender(t, { home, mode: 'leak', holdMs: -1, retryMs: 0 });
    await owner.event('acquired');
    const { child: childPid } = await owner.event('child');
    t.after(() => { try { process.kill(childPid, 'SIGKILL'); } catch {} });
    owner.child.kill('SIGKILL');
    await owner.exited;
    const next = startContender(t, { home, mode: 'bind', holdMs: 0, retryMs: 0 });
    await next.event('refused');
});

test('no code path moves, renames or removes an owner record (source scan, with a planted negative control)', () => {
    const calls = /\b(unlink|unlinkSync|rm|rmSync|rename|renameSync|rmdir|rmdirSync|truncate|truncateSync|copyFile|copyFileSync|link|linkSync|symlink|symlinkSync)\s*\(/g;
    // The text of every call to one of those functions, to its balanced closing parenthesis.
    function callsIn(source) {
        const found = [];
        for (const match of source.matchAll(calls)) {
            let depth = 0;
            let end = match.index + match[0].length - 1;
            for (; end < source.length; end += 1) {
                if (source[end] === '(') depth += 1;
                if (source[end] === ')' && --depth === 0) break;
            }
            found.push(source.slice(match.index, end + 1));
        }
        return found;
    }
    const code = (text) => text.split('\n').filter((line) => !/^\s*(\/\/|\*|\/\*)/.test(line)).join('\n');
    const offenders = (source) => callsIn(code(source)).filter((call) => /owner/i.test(call));
    // The scan sees what it is meant to see.
    assert.equal(offenders('fs.unlinkSync(ownerFile);').length, 1);
    assert.equal(offenders('fsApi.renameSync(path.join(dir, "owner.json"), other)').length, 1);
    assert.equal(offenders('fs.rmSync(layout.ownerFile, { force: true })').length, 1);
    assert.equal(offenders('fs.unlinkSync(launchFile)').length, 0);
    const files = fs.readdirSync(LIB).filter((name) => name.endsWith('.mjs'));
    assert.ok(files.includes('ownership.mjs') && files.includes('host.mjs') && files.includes('reaper.mjs'));
    for (const name of files) {
        const source = fs.readFileSync(path.join(LIB, name), 'utf8');
        assert.deepEqual(offenders(source), [], `${name} removes or moves an owner record`);
        // The owner record is named in two modules only, and ownership.mjs only writes it.
        if (!['host.mjs', 'ownership.mjs'].includes(name)) assert.equal(/owner\.json|ownerFile/.test(code(source)), false, `${name} names the owner record`);
    }
    const own = code(fs.readFileSync(path.join(LIB, 'ownership.mjs'), 'utf8'));
    const uses = own.split('\n').filter((line) => /ownerFile/.test(line));
    for (const line of uses) assert.match(line, /ownerFile = null|if \(ownerFile\)|path\.dirname\(ownerFile\)|writeFileSync\(ownerFile|ownerFile\} could not be written|`local-llm: \$\{ownerFile\}/, line);
});

test('at run time an acquire, a refusal, a release and a second acquire touch the owner record only by writing it', async (t) => {
    const home = scratchHome(t);
    const ownerFile = path.join(home, 'state', 'owner.json');
    const touched = [];
    const spy = new Proxy(fs, {
        get(target, name) {
            const value = target[name];
            if (typeof value !== 'function' || !/unlink|rm|rename|truncate|copyFile|link|symlink/i.test(String(name))) return value;
            return (...args) => { touched.push([String(name), ...args.filter((arg) => typeof arg === 'string')]); return value.apply(target, args); };
        },
    });
    const port = await freePort();
    const first = await acquireOwnership({ realHome: home, port, ownerFile, fsApi: spy });
    await assert.rejects(acquireOwnership({ realHome: home, port, ownerFile, fsApi: spy }), { code: 'owner_held' });
    await first.release();
    const second = await acquireOwnership({ realHome: home, port, ownerFile, fsApi: spy });
    await second.release();
    assert.deepEqual(touched.filter((entry) => entry.some((arg) => /owner/.test(arg))), []);
    assert.ok(fs.existsSync(ownerFile), 'the record is still there after the owners left');
});
