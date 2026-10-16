// A tiny stand-in for a llama-server, run by the real-process launcher test (test/local-launcher.test.mjs): GET /health answers 200,
// GET /shutdown makes it exit, and a lifetime timer makes it exit by itself, so cleaning up never needs a signal.
import http from 'node:http';

const arg = (name, fallback) => { const i = process.argv.indexOf(name); return i < 0 ? fallback : process.argv[i + 1]; };
const port = Number(arg('--port'));
const lifetimeMs = Number(arg('--lifetime-ms', 30000));

http.createServer((req, res) => {
  if (req.url === '/health') { res.writeHead(200, { 'content-type': 'application/json' }); return res.end('{"ok":true}'); }
  if (req.url === '/shutdown') return res.end('bye', () => process.exit(0));
  res.writeHead(404);
  res.end();
}).listen(port, '127.0.0.1');
setTimeout(() => process.exit(0), lifetimeMs);
