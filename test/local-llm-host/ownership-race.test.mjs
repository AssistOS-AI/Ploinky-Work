// A predecessor that releases the home between the contender's failed bind and its handshake connect: the connect finds nobody,
// which is not "a program that is not pworker". The contender binds again and becomes the owner (it must not stay without the engine
// for its whole lifetime because of a few milliseconds).
import '../home.mjs';
import assert from 'node:assert/strict';
import net from 'node:net';
import test from 'node:test';

import { acquireOwnership, diagnoseHolder } from '../../lib/local-llm-host/ownership.mjs';
import { freePort } from './fixtures/engine.mjs';

const REAL_HOME = '/pworker/race-home';

test('a handshake that finds the predecessor gone is retried: the contender acquires after the release', async () => {
    const port = await freePort();
    const predecessor = await acquireOwnership({ realHome: REAL_HOME, port, pid: 11111 });
    let connects = 0;
    // The connect waits until the predecessor has finished releasing, as when its drain ends at that instant.
    const racing = { ...net, connect: (options) => { connects += 1; const socket = new net.Socket(); predecessor.release().then(() => socket.connect(options)); return socket; } };
    const successor = await acquireOwnership({ realHome: REAL_HOME, port, pid: 22222, netImpl: racing, goneDelayMs: 5 });
    assert.equal(successor.port, port);
    assert.ok(connects >= 1, 'the failed bind was diagnosed first');
    // The new owner really holds the port: a third contender is told so.
    await assert.rejects(acquireOwnership({ realHome: REAL_HOME, port, pid: 33333 }), { code: 'owner_held' });
    await successor.release();
});

test('diagnoseHolder reports a refused connection and an empty close as gone, and a silent or foreign program as unrelated', async () => {
    const port = await freePort();
    assert.deepEqual(await diagnoseHolder({ port, realHome: REAL_HOME }), { kind: 'gone' }, 'nothing listens');
    const empty = net.createServer((socket) => socket.end());
    await new Promise((resolve) => empty.listen(port, '127.0.0.1', resolve));
    assert.deepEqual(await diagnoseHolder({ port, realHome: REAL_HOME }), { kind: 'gone' }, 'closed without a byte');
    await new Promise((resolve) => empty.close(resolve));
    const foreign = net.createServer((socket) => socket.end('HTTP/1.1 200 OK\r\n\r\n'));
    await new Promise((resolve) => foreign.listen(port, '127.0.0.1', resolve));
    assert.deepEqual(await diagnoseHolder({ port, realHome: REAL_HOME }), { kind: 'unrelated' });
    await new Promise((resolve) => foreign.close(resolve));
    const silent = net.createServer((socket) => socket.on('error', () => {}));
    await new Promise((resolve) => silent.listen(port, '127.0.0.1', resolve));
    assert.deepEqual(await diagnoseHolder({ port, realHome: REAL_HOME, timeoutMs: 100 }), { kind: 'unrelated' });
    silent.closeAllConnections?.();
    await new Promise((resolve) => silent.close(resolve));
});

test('a program that accepts and closes without a word is retried a bounded number of times and then refused as not pworker', async () => {
    const port = await freePort();
    const empty = net.createServer((socket) => socket.end());
    await new Promise((resolve) => empty.listen(port, '127.0.0.1', resolve));
    try {
        await assert.rejects(acquireOwnership({ realHome: REAL_HOME, port, goneRetries: 3, goneDelayMs: 1 }), { code: 'owner_port_taken' });
    } finally {
        await new Promise((resolve) => empty.close(resolve));
    }
});
