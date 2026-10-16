// One contender for the ownership of a home, run as its own process by the ownership tests. It prints one JSON line per event
// ({ ev, pid, t }, t = the monotonic clock in ns, which every process of the machine shares):
//   acquired   after the lock is taken, released   just BEFORE it is given up (so the held interval is never longer than the real one),
//   refused    the attempt window ended without the lock, child   the pid of a runner-like child it started while holding.
// argv[2] is a JSON object:
//   home        the pworker home (its realpath names the lock)
//   mode        'bind'  the real implementation (acquireOwnership)
//               'check-then-write'  a deliberately broken lock (look for a file, wait, write it): the negative control
//               'leak'  the real lock, but its child inherits the listening socket: the negative control of the child test
//   holdMs      how long to hold; -1 holds until killed
//   retryMs     how long to keep trying (0: one attempt)
//   startAtMs   wall-clock time at which every contender starts (a barrier)
//   gapMs       the check-then-write variant's window between the check and the write
//   port        owner port override (bind mode)
//   child       true: start a long-lived child process while holding
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';

import net from 'node:net';

import { acquireOwnership, ownerPortFor } from '../../../lib/local-llm-host/ownership.mjs';

const options = JSON.parse(process.argv[2]);
const now = () => process.hrtime.bigint().toString();
const emit = (ev, extra = {}) => process.stdout.write(`${JSON.stringify({ ev, pid: process.pid, t: now(), ...extra })}\n`);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

while (options.startAtMs && Date.now() < options.startAtMs) await sleep(1);

const realHome = fs.realpathSync(options.home);
const deadline = Date.now() + (options.retryMs ?? 0);
let lock = null;
const lockFile = path.join(realHome, 'check.lock');

async function attempt() {
    if (options.mode === 'check-then-write') {
        if (fs.existsSync(lockFile)) return null;
        await sleep(options.gapMs ?? 50);
        fs.writeFileSync(lockFile, String(process.pid));
        return { release: async () => { fs.unlinkSync(lockFile); } };
    }
    if (options.mode === 'leak') {
        const server = net.createServer();
        await new Promise((resolve, reject) => { server.once('error', reject); server.listen(options.port ?? ownerPortFor(realHome), '127.0.0.1', resolve); });
        const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { detached: true, stdio: ['ignore', 'ignore', 'ignore', server._handle.fd] });
        child.unref();
        emit('child', { child: child.pid });
        return { port: server.address().port, release: () => new Promise((resolve) => server.close(resolve)) };
    }
    try {
        return await acquireOwnership({ realHome, port: options.port ?? ownerPortFor(realHome), ownerFile: path.join(realHome, 'owner.json') });
    } catch (error) {
        if (['owner_held', 'owner_port_collision', 'owner_port_taken'].includes(error.code)) return null;
        throw error;
    }
}

for (;;) {
    lock = await attempt();
    if (lock || Date.now() >= deadline) break;
    await sleep(2);
}
if (!lock) {
    emit('refused');
    process.exit(0);
}
emit('acquired', { port: lock.port ?? null });
if (options.child && options.mode !== 'leak') {
    const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { detached: true, stdio: 'ignore' });
    child.unref();
    emit('child', { child: child.pid });
}
if (options.holdMs >= 0) {
    await sleep(options.holdMs);
    emit('released');
    await lock.release();
    process.exit(0);
}
// Held until killed.
setInterval(() => {}, 1000);
