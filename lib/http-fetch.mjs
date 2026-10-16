// A fetch over node:http and node:https without the fixed 300-second header and body timeouts of Node's built-in fetch: a model that
// thinks for longer than that before its first byte is waited for until the caller's own signal (a duration cap) ends the call.
// It returns a standard Response whose body streams, so SSE answers pass through unchanged. Used by the server for its providers and
// by the library for its server. `init.socketPath` sends the request to a Unix socket instead of the host of the URL (a local model server
// that listens on one); the URL then only names the request path and the Host header.
import http from 'node:http';
import https from 'node:https';
import { Readable } from 'node:stream';

const agents = { 'http:': new http.Agent({ keepAlive: true, maxSockets: 256 }), 'https:': new https.Agent({ keepAlive: true, maxSockets: 256 }) };

export function httpFetch(input, init = {}) {
  const url = new URL(typeof input === 'string' ? input : input.url);
  const lib = url.protocol === 'https:' ? https : http;
  const headers = {};
  for (const [k, v] of new Headers(init.headers ?? {})) headers[k] = v;
  const body = init.body == null ? null : typeof init.body === 'string' || Buffer.isBuffer(init.body) || init.body instanceof Uint8Array ? init.body : String(init.body);
  if (body != null && headers['content-length'] == null) headers['content-length'] = String(Buffer.byteLength(body));
  return new Promise((resolve, reject) => {
    const signal = init.signal;
    if (signal?.aborted) return reject(signal.reason ?? new DOMException('This operation was aborted', 'AbortError'));
    // The shared pool names a connection by host, port and socket path, so a socket never reuses a TCP connection or another socket's.
    const req = lib.request(url, { method: init.method || 'GET', headers, agent: agents[url.protocol], ...(init.socketPath ? { socketPath: init.socketPath } : {}) }, (res) => {
      const h = new Headers();
      for (const [k, v] of Object.entries(res.headers)) if (v != null) for (const x of [].concat(v)) h.append(k, String(x));
      const status = res.statusCode ?? 0;
      const noBody = status === 204 || status === 304 || (init.method || 'GET').toUpperCase() === 'HEAD';
      if (noBody) res.resume();
      // new Response() accepts only 200-599: any other status (a broken or hostile endpoint) rejects instead of throwing uncaught.
      try { resolve(new Response(noBody ? null : Readable.toWeb(res), { status, statusText: res.statusMessage, headers: h })); }
      catch (e) { res.destroy(); reject(Object.assign(new TypeError('fetch failed'), { cause: new Error(`invalid HTTP status ${status}: ${e.message}`) })); }
    });
    const onAbort = () => { const e = signal.reason ?? new DOMException('This operation was aborted', 'AbortError'); req.destroy(e); reject(e); };
    signal?.addEventListener('abort', onAbort, { once: true });
    req.on('error', (e) => { signal?.removeEventListener('abort', onAbort); reject(Object.assign(new TypeError('fetch failed'), { cause: e })); });
    req.on('close', () => signal?.removeEventListener('abort', onAbort));
    if (body != null) req.write(body);
    req.end();
  });
}
