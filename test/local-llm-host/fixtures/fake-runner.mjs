// A stand-in for llama-server: `fake-runner.mjs ... --host <socket path> --api-key <key> ...`. It listens on the Unix socket,
// answers /health, answers /v1/models and /v1/chat/completions only with `Authorization: Bearer <key>`, and says what it saw
// at GET /seen. The chat answer echoes the model and the key it was given; with "stream": true it answers three SSE events.
// It exits on SIGTERM (or ignores it with FAKE_RUNNER_IGNORE_TERM=1, for the reaper's SIGKILL path).
import fs from 'node:fs';
import http from 'node:http';

const argv = process.argv.slice(2);
const after = (flag) => argv[argv.indexOf(flag) + 1];
const socketPath = after('--host');
const key = after('--api-key');
const seen = [];

const server = http.createServer((req, res) => {
    const authorized = req.headers.authorization === `Bearer ${key}`;
    let body = '';
    req.on('data', (chunk) => { body += chunk; });
    req.on('end', () => {
        if (req.url === '/health') { res.writeHead(200); res.end('{"status":"ok"}'); return; }
        if (req.url === '/seen') { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify(seen)); return; }
        if (!authorized) { res.writeHead(401); res.end('{"error":"bad key"}'); return; }
        if (req.url === '/v1/models') { res.writeHead(200, { 'content-type': 'application/json' }); res.end('{"data":[{"id":"fake"}]}'); return; }
        if (req.url === '/v1/chat/completions') {
            const request = JSON.parse(body || '{}');
            seen.push({ model: request.model, key: req.headers.authorization.slice(7) });
            if (request.stream) {
                res.writeHead(200, { 'content-type': 'text/event-stream' });
                res.write('data: {"choices":[{"delta":{"content":"one "}}]}\n\n');
                setTimeout(() => res.write('data: {"choices":[{"delta":{"content":"two"}}]}\n\n'), 20);
                setTimeout(() => res.end('data: [DONE]\n\n'), 40);
                return;
            }
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(JSON.stringify({ choices: [{ message: { content: `echo ${request.model}` } }], usage: { prompt_tokens: 1, completion_tokens: 1 } }));
            return;
        }
        res.writeHead(404); res.end('{}');
    });
});

try { fs.unlinkSync(socketPath); } catch {}
server.listen(socketPath, () => { if (process.send) process.send('listening'); });
if (process.env.FAKE_RUNNER_IGNORE_TERM === '1') process.on('SIGTERM', () => {});
else process.on('SIGTERM', () => { server.close(); process.exit(0); });
