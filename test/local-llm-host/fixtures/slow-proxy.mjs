// A stand-in for a pworker proxy that drains slowly: it records itself in the home like `pworker serve`, answers /health, and on
// SIGTERM stops answering at once but exits only after SLOW_EXIT_MS (SLOW_EXIT_MS=never: it ignores SIGTERM).
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';

const home = process.env.PWORKER_HOME;
const server = http.createServer((req, res) => {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ ok: true, uptime_s: 1, tiers: {}, upstreams: {} }));
});
server.listen(0, '127.0.0.1', () => {
    fs.writeFileSync(path.join(home, 'server.json'), `${JSON.stringify({ pid: process.pid, host: '127.0.0.1', port: server.address().port })}\n`);
    console.log('ready');
});
process.on('SIGTERM', () => {
    if (process.env.SLOW_EXIT_MS === 'never') return;
    server.close();
    server.closeAllConnections();
    setTimeout(() => process.exit(0), Number(process.env.SLOW_EXIT_MS));
});
