// Who may run the local-llm engine of one pworker home: the process that holds an exclusive loopback TCP listener on the
// home's own port. The kernel grants a bind to one process at a time and takes it back when the holder exits, however it
// exits (a SIGKILL included), so there is never a stale lock to reclaim and nothing here reclaims anything.
//
//   - The port comes from the realpath of the home (ownerPortFor), in a range below both the Linux (32768-60999) and the
//     macOS (49152-65535) ephemeral ports, so no outgoing connection can hold it by chance.
//   - Ownership is the successful bind, with no `reusePort` and no `exclusive: false`, and no other port is tried. EADDRINUSE
//     means someone else holds it: the controller is not created.
//   - The listener answers every connection with one line naming the home and the pid, then closes it. That handshake is for
//     the error message only (`diagnoseHolder`): it tells a pworker of this home from a pworker of another home that hashed
//     to the same port from a program that is not pworker. It never grants ownership.
//   - The listener is created by libuv with close-on-exec, so a runner the owner starts does not inherit it: when the owner
//     is killed the port is free at once, even while the runner it started is still alive.
//   - On macOS a program listening on `0.0.0.0:<port>` or `[::]:<port>` does not stop the loopback bind (Linux refuses it), so
//     after the bind succeeds the owner probes the wildcard address with a listener that drops every connection and is closed at
//     once. EADDRINUSE there means an unrelated holder: the loopback listener is released and the home is refused. A pworker never
//     binds a wildcard address, so nothing of its own can make the probe fail, and a wildcard listener is never kept.
//   - `owner.json` (the pid, the port and the time) is written after the bind for people to read. Nothing reads it to decide
//     anything, and no code here moves, renames or removes it: the next owner writes over it in place.

import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';

import { LocalLlmError } from '../local-llm/errors.mjs';

/** The owner ports: below Linux's (32768) and macOS's (49152) ephemeral ranges, and away from the proxy's 18080 and the 196xx ports. */
export const OWNER_PORT_FIRST = 20000;
export const OWNER_PORT_LAST = 32767;
const HANDSHAKE_TAG = 'pworker-local-llm-owner';
const HANDSHAKE_TIMEOUT_MS = 1500;
const HANDSHAKE_MAX_BYTES = 512;

/** The owner port of a home, from its realpath (callers pass the real path). Deterministic, in [OWNER_PORT_FIRST, OWNER_PORT_LAST]. */
export function ownerPortFor(realHome) {
    const digest = crypto.createHash('sha256').update(`pworker local-llm owner ${realHome}`).digest();
    return OWNER_PORT_FIRST + (digest.readUInt32BE(0) % (OWNER_PORT_LAST - OWNER_PORT_FIRST + 1));
}

/**
 * Connects to `port` and reads the handshake line. Resolves one of
 *   { kind: 'same-home', pid }       a pworker that owns this home
 *   { kind: 'other-home', pid, home }  a pworker that owns another home whose port is the same
 *   { kind: 'unrelated' }            anything else (no answer, another protocol, nothing listening any more)
 *   { kind: 'gone' }                 no answer at all: the connection was refused or closed empty (the holder just left)
 * It only diagnoses; it never decides who owns the home.
 */
export function diagnoseHolder({ port, realHome, host = '127.0.0.1', timeoutMs = HANDSHAKE_TIMEOUT_MS, netImpl = net }) {
    return new Promise((resolve) => {
        let received = '';
        let settled = false;
        const socket = netImpl.connect({ port, host });
        const finish = (diagnosis) => {
            if (settled) return;
            settled = true;
            socket.destroy();
            resolve(diagnosis);
        };
        const decide = () => {
            try {
                const hello = JSON.parse(received.split('\n')[0]);
                if (hello?.tag === HANDSHAKE_TAG && Number.isSafeInteger(hello.pid) && typeof hello.home === 'string') {
                    return hello.home === realHome ? { kind: 'same-home', pid: hello.pid } : { kind: 'other-home', pid: hello.pid, home: hello.home };
                }
            } catch { /* not ours */ }
            // Nobody answered at all (the connection was refused, or closed without a byte): the holder left between the failed bind and
            // this connect, which is not a program that is not pworker. The caller binds again.
            return received === '' ? { kind: 'gone' } : { kind: 'unrelated' };
        };
        socket.setTimeout(timeoutMs, () => finish({ kind: 'unrelated' }));
        socket.on('data', (chunk) => {
            received += chunk.toString('utf8');
            if (received.includes('\n') || received.length > HANDSHAKE_MAX_BYTES) finish(decide());
        });
        socket.on('end', () => finish(decide()));
        socket.on('error', (error) => finish(error?.code === 'ECONNREFUSED' ? { kind: 'gone' } : { kind: 'unrelated' }));
        socket.on('close', () => finish(decide()));
    });
}

function wildcardRefusal(port, realHome) {
    return new LocalLlmError('owner_port_taken', `Port ${port}, which pworker uses to own ${realHome}, is held by a program that is not pworker on a wildcard address `
        + '(0.0.0.0 or ::; pworker binds 127.0.0.1 only). Stop that program, or use another PWORKER_HOME.', { kind: 'wildcard', port });
}

function refusal(kind, port, realHome, diagnosis) {
    if (kind === 'same-home') {
        return new LocalLlmError('owner_held', `Another pworker (pid ${diagnosis.pid}) already runs the local models of ${realHome}; only one process may.`, { kind, port, pid: diagnosis.pid });
    }
    if (kind === 'other-home') {
        return new LocalLlmError('owner_port_collision', `Port ${port}, which owns ${realHome}, is held by the pworker (pid ${diagnosis.pid}) of ${diagnosis.home}. `
            + 'Two homes hash to the same owner port; use another PWORKER_HOME for one of them.', { kind, port, pid: diagnosis.pid, home: diagnosis.home });
    }
    return new LocalLlmError('owner_port_taken', `Port ${port} on 127.0.0.1, which pworker uses to own ${realHome}, is held by a program that is not pworker. `
        + 'Stop that program, or use another PWORKER_HOME.', { kind, port });
}

/**
 * Whether `0.0.0.0:<port>` is held by someone else: a probe bind whose listener drops every connection and is closed before this
 * resolves, so no wildcard listener outlives the call. EADDRINUSE is true, a successful bind false, any other error is thrown.
 */
async function wildcardBusy({ port, netImpl }) {
    const probe = netImpl.createServer((socket) => socket.destroy());
    try {
        await new Promise((resolve, reject) => {
            probe.once('error', reject);
            probe.listen({ port, host: '0.0.0.0' }, () => { probe.off('error', reject); resolve(); });
        });
    } catch (error) {
        if (error?.code === 'EADDRINUSE') return true;
        throw error;
    }
    await new Promise((resolve) => { probe.close(() => resolve()); probe.closeAllConnections?.(); });
    return false;
}

/**
 * Takes ownership of the home `realHome` (its realpath): an exclusive bind of `port` on loopback. Resolves
 * `{ port, release() }`; rejects with `owner_held`, `owner_port_collision` or `owner_port_taken` when the bind fails with
 * EADDRINUSE, and with the bind's own error otherwise. `release()` closes the listener and resolves once it is closed.
 * `ownerFile`, when given, receives an informational record after the bind (a failure to write it is logged, nothing more).
 */
export async function acquireOwnership({
    realHome, port = ownerPortFor(realHome), host = '127.0.0.1', ownerFile = null, pid = process.pid,
    netImpl = net, fsApi = fs, log = () => {}, now = () => new Date(), goneRetries = 40, goneDelayMs = 25, platformName = process.platform,
}) {
    const sockets = new Set();
    const server = netImpl.createServer((socket) => {
        sockets.add(socket);
        socket.on('close', () => sockets.delete(socket));
        socket.on('error', () => {});
        socket.end(`${JSON.stringify({ tag: HANDSHAKE_TAG, pid, home: realHome })}\n`);
    });
    // A bind that fails because the holder is leaving at that very moment is tried again: a refused or empty handshake means the
    // holder is gone (the port may be free by now), not that an unrelated program holds it. Nothing is claimed except by a successful bind.
    for (let attempt = 0; ; attempt++) {
        try {
            await new Promise((resolve, reject) => {
                server.once('error', reject);
                // No reusePort and no exclusive:false: the bind is all or nothing.
                server.listen({ port, host }, () => { server.off('error', reject); resolve(); });
            });
            break;
        } catch (error) {
            if (error?.code !== 'EADDRINUSE') throw error;
            const diagnosis = await diagnoseHolder({ port, realHome, host, netImpl });
            if (diagnosis.kind === 'gone' && attempt < goneRetries) {
                await new Promise((resolve) => setTimeout(resolve, goneDelayMs));
                continue;
            }
            throw refusal(diagnosis.kind === 'gone' ? 'unrelated' : diagnosis.kind, port, realHome, diagnosis);
        }
    }
    // macOS grants the loopback bind next to a wildcard holder of the same port: look for one before the ownership is claimed.
    if (platformName === 'darwin' && host === '127.0.0.1') {
        const closeOwn = () => new Promise((resolve) => { for (const socket of sockets) socket.destroy(); server.close(() => resolve()); });
        let busy;
        try {
            busy = await wildcardBusy({ port, netImpl });
        } catch (error) {
            await closeOwn();
            throw error;
        }
        if (busy) {
            await closeOwn();
            throw wildcardRefusal(port, realHome);
        }
    }
    // The listener is not a reason to keep the process alive; the server it belongs to is.
    server.unref();
    server.on('error', (error) => log(`local-llm: the owner listener on ${port} reported ${error?.code ?? error?.message}`));
    if (ownerFile) {
        try {
            fsApi.mkdirSync(path.dirname(ownerFile), { recursive: true, mode: 0o700 });
            // Written in place over whatever an earlier owner left: a record is never moved or removed.
            fsApi.writeFileSync(ownerFile, `${JSON.stringify({ pid, port, home: realHome, startedAt: now().toISOString() })}\n`, { mode: 0o600 });
        } catch (error) {
            log(`local-llm: ${ownerFile} could not be written (${error?.code ?? error?.message}); it is informational only`);
        }
    }
    let released = null;
    return Object.freeze({
        port,
        release() {
            released ??= new Promise((resolve) => {
                for (const socket of sockets) socket.destroy();
                server.close(() => resolve());
            });
            return released;
        },
    });
}
