// The wildcard probe of the owner bind on macOS (decision taken from a two-process probe on a Mac: with a foreign holder on
// 0.0.0.0:P or [::]:P the loopback bind still succeeds, but a bind of 0.0.0.0:P then fails EADDRINUSE; with no holder it
// succeeds). Here a fake `net` plays the kernel, so the file runs on Linux and macOS alike and makes no real bind on the
// wildcard address; the real two-process probe is a script outside the test globs.
import '../home.mjs';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import test from 'node:test';

import { acquireOwnership } from '../../lib/local-llm-host/ownership.mjs';

const REAL_HOME = '/pworker/darwin-home';
const PORT = 21234;

/**
 * A `net` whose servers bind in a table of { host -> holder }: a host in `busy` makes listen fail with EADDRINUSE, a host in
 * `errors` fails with that error code, any other host binds. Every bind, close and connection handler is recorded.
 */
function fakeNet({ busy = [], errors = {} } = {}) {
    const bound = new Set();
    const events = [];
    const servers = [];
    const createServer = (handler) => {
        const server = new EventEmitter();
        let host = null;
        Object.assign(server, {
            handler,
            listen({ port, host: wanted }, callback) {
                events.push(`listen ${wanted}:${port}`);
                setImmediate(() => {
                    if (busy.includes(wanted) || bound.has(wanted)) return server.emit('error', Object.assign(new Error('listen EADDRINUSE'), { code: 'EADDRINUSE' }));
                    if (errors[wanted]) return server.emit('error', Object.assign(new Error('listen failed'), { code: errors[wanted] }));
                    host = wanted;
                    bound.add(wanted);
                    return callback();
                });
                return server;
            },
            close(callback) {
                events.push(`close ${host}`);
                if (host !== null) bound.delete(host);
                host = null;
                setImmediate(() => callback?.());
            },
            unref() { return server; },
            closeAllConnections() {},
        });
        servers.push(server);
        return server;
    };
    // The handshake after a failed loopback bind: the holder answers with a line that is not pworker's.
    const connect = () => {
        const socket = new EventEmitter();
        Object.assign(socket, { setTimeout() {}, destroy() {} });
        setImmediate(() => socket.emit('data', Buffer.from('HTTP/1.1 400 Bad Request\n')));
        return socket;
    };
    return { createServer, connect, bound, events, servers };
}

const acquire = (net, extra = {}) => acquireOwnership({ realHome: REAL_HOME, port: PORT, netImpl: net, goneRetries: 0, goneDelayMs: 1, ...extra });

test('macOS: a foreign holder of the wildcard address is refused with owner_port_taken, and nothing is left bound', async () => {
    const net = fakeNet({ busy: ['0.0.0.0'] });
    await assert.rejects(acquire(net, { platformName: 'darwin' }), (error) => {
        assert.equal(error.code, 'owner_port_taken');
        assert.match(error.message, /on a wildcard address/);
        assert.match(error.message, new RegExp(String(PORT)));
        assert.match(error.message, /not pworker/);
        return true;
    });
    assert.deepEqual(net.events, [`listen 127.0.0.1:${PORT}`, `listen 0.0.0.0:${PORT}`, 'close 127.0.0.1'], 'the loopback listener is released at once');
    assert.equal(net.bound.size, 0);
});

test('macOS: with no foreign holder the probe binds the wildcard address, drops connections, closes it before ownership resolves, and keeps the loopback listener', async () => {
    const net = fakeNet();
    const ownership = await acquire(net, { platformName: 'darwin' });
    assert.equal(ownership.port, PORT);
    assert.deepEqual(net.events, [`listen 127.0.0.1:${PORT}`, `listen 0.0.0.0:${PORT}`, 'close 0.0.0.0']);
    assert.deepEqual([...net.bound], ['127.0.0.1'], 'only the loopback listener is kept; no wildcard listener outlives the probe');
    // The probe server drops every connection at once.
    const probe = net.servers[1];
    const socket = { destroyed: false, destroy() { this.destroyed = true; } };
    probe.handler(socket);
    assert.equal(socket.destroyed, true);
    await ownership.release();
    assert.equal(net.bound.size, 0);
});

test('macOS: any other error of the probe bind releases the loopback listener and is thrown as it is', async () => {
    const net = fakeNet({ errors: { '0.0.0.0': 'EACCES' } });
    await assert.rejects(acquire(net, { platformName: 'darwin' }), { code: 'EACCES' });
    assert.equal(net.bound.size, 0);
    assert.deepEqual(net.events.slice(-1), ['close 127.0.0.1']);
});

test('macOS: a loopback holder is still refused by the loopback bind, and the wildcard address is not probed', async () => {
    const net = fakeNet({ busy: ['127.0.0.1'] });
    await assert.rejects(acquire(net, { platformName: 'darwin' }), { code: 'owner_port_taken' });
    assert.ok(net.events.every((event) => !event.includes('0.0.0.0')), net.events.join(', '));
});

test('Linux makes no wildcard probe: the kernel refuses the loopback bind next to a wildcard holder by itself', async () => {
    const net = fakeNet();
    const ownership = await acquire(net, { platformName: 'linux' });
    assert.deepEqual(net.events, [`listen 127.0.0.1:${PORT}`]);
    await ownership.release();
    // A platform that names none is the process's own: on this machine either no probe (Linux) or the probe (macOS), never both.
    const own = fakeNet();
    const defaulted = await acquire(own);
    assert.equal(own.events.some((event) => event.includes('0.0.0.0')), process.platform === 'darwin');
    await defaulted.release();
});

test('the probe is skipped for a host other than loopback and never runs without a successful loopback bind', async () => {
    const net = fakeNet();
    const ownership = await acquire(net, { platformName: 'darwin', host: '127.0.0.2' });
    assert.ok(net.events.every((event) => !event.includes('0.0.0.0')));
    await ownership.release();
});
