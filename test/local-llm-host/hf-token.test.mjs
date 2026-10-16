// V19: `pworker models hf-token` with a synthetic token. The file keys/huggingface.env is mode 0600 after the first write and after a
// replacement (also of a file that was 0644), holds one line, and the token is in no output, no log and no other file of the home;
// a negative control shows that the scan finds a planted copy.
import '../home.mjs';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import path from 'node:path';
import { PassThrough } from 'node:stream';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { checkToken, hfTokenFile, readTokenInput, writeHfToken } from '../../lib/pworker/hf-token.mjs';
import { loadLayers } from '../../lib/config.mjs';
import { createLocalLlmHost } from '../../lib/local-llm-host/host.mjs';
import { createPworkerServer } from '../../lib/server.mjs';
import { serverHome } from '../../lib/settings.mjs';
import { freePort, scratch } from './fixtures/engine.mjs';

const BIN = fileURLToPath(new URL('../../bin/pworker.mjs', import.meta.url));
const FIRST = 'hf_SyntheticTokenOne0123456789abcdef';
const SECOND = 'hf_SyntheticTokenTwo9876543210fedcba';

function bin(home, args, input) {
    const { NODE_TEST_CONTEXT: _context, ...parent } = process.env;
    const child = spawn(process.execPath, [BIN, ...args], { env: { ...parent, PWORKER_HOME: home }, stdio: ['pipe', 'pipe', 'pipe'] });
    let stdout = '', stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.stdin.end(input);
    return new Promise((resolve) => child.on('close', (code) => resolve({ code, stdout, stderr })));
}

/** Every file under `dir` whose bytes contain `needle`, as paths relative to `dir`. */
function filesContaining(dir, needle) {
    const hits = [];
    const walk = (current) => {
        for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
            const full = path.join(current, entry.name);
            if (entry.isDirectory()) walk(full);
            else if (entry.isFile() && fs.readFileSync(full).includes(needle)) hits.push(path.relative(dir, full));
        }
    };
    walk(dir);
    return hits;
}

const mode = (file) => (fs.statSync(file).mode & 0o777).toString(8).padStart(4, '0');

test('the token is stored mode 0600 after the first write and after a replacement, as one line that is replaced', async (t) => {
    const home = scratch(t);
    const file = hfTokenFile(home);
    const first = await bin(home, ['models', 'hf-token'], `${FIRST}\n`);
    assert.equal(first.code, 0, first.stderr);
    assert.equal(mode(file), '0600');
    assert.equal(fs.readFileSync(file, 'utf8'), `HF_TOKEN="${FIRST}"\n`);
    assert.equal(mode(path.dirname(file)), '0700', 'the keys folder is private too');
    // A replacement: the old file may have been made group-readable by someone; the new one is not.
    fs.chmodSync(file, 0o644);
    const second = await bin(home, ['models', 'hf-token'], `${SECOND}\n`);
    assert.equal(second.code, 0, second.stderr);
    assert.equal(mode(file), '0600', 'a replaced 0644 file is 0600 again');
    const text = fs.readFileSync(file, 'utf8');
    assert.equal(text, `HF_TOKEN="${SECOND}"\n`);
    assert.equal(text.trim().split('\n').length, 1, 'one line');
    assert.ok(!text.includes(FIRST), 'the old token is gone');
    assert.deepEqual(fs.readdirSync(path.dirname(file)).filter((name) => name.startsWith('.huggingface')), [], 'no temporary file is left');
});

test('the token appears in no output and no file of the home but its own, and a planted copy is found by the same scan', async (t) => {
    const home = scratch(t);
    const results = [await bin(home, ['models', 'hf-token'], `${FIRST}\n`), await bin(home, ['models', 'hf-token'], `${SECOND}\n`)];
    for (const result of results) {
        for (const secret of [FIRST, SECOND]) {
            assert.ok(!result.stdout.includes(secret), 'stdout');
            assert.ok(!result.stderr.includes(secret), 'stderr');
        }
        assert.match(result.stdout, /Hugging Face token stored in .*huggingface\.env \(mode 0600\)/);
    }
    // The host reads the token from the file for the engine and redacts it from its messages.
    const host = createLocalLlmHost({ home, env: { PWORKER_HOME: home }, platformName: 'linux' });
    assert.equal(host.engineEnvironment().HF_TOKEN, SECOND);
    assert.equal(host.redact(`failed with ${SECOND}`), 'failed with ***');
    assert.deepEqual(filesContaining(home, SECOND), [path.join('keys', 'huggingface.env')], 'only the token file holds the token');
    assert.deepEqual(filesContaining(home, FIRST), [], 'the replaced token is nowhere');
    // Negative control: the scan does see a copy in a log.
    fs.mkdirSync(path.join(home, 'logs'), { recursive: true });
    fs.writeFileSync(path.join(home, 'logs', 'pworker.log'), `leaked ${SECOND}\n`);
    assert.deepEqual(filesContaining(home, SECOND).sort(), [path.join('keys', 'huggingface.env'), path.join('logs', 'pworker.log')]);
});

test('a token that is empty or has spaces, quotes or shell characters is refused, without echoing it, and nothing is written', async (t) => {
    const home = scratch(t);
    for (const bad of ['', 'two words', 'quo"te', '$(touch x)', 'a'.repeat(600)]) {
        const result = await bin(home, ['models', 'hf-token'], `${bad}\n`);
        assert.equal(result.code, 1, JSON.stringify(bad));
        assert.match(result.stderr, /Not stored/);
        if (bad.length > 3) assert.ok(!result.stderr.includes(bad) && !result.stdout.includes(bad), 'the refused token is not echoed');
    }
    assert.equal(fs.existsSync(hfTokenFile(home)), false);
    assert.equal(checkToken(FIRST).ok, true);
    assert.throws(() => writeHfToken(home, 'bad token'), /Not stored/);
});

test('on a terminal the token is typed without echo: nothing of it is written to the screen, Backspace edits, Ctrl-C cancels', async () => {
    const make = () => {
        const input = Object.assign(new EventEmitter(), { isTTY: true, raw: false, setRawMode(on) { this.raw = on; }, setEncoding() {}, resume() {}, pause() {} });
        const output = new PassThrough();
        let screen = '';
        output.on('data', (chunk) => { screen += chunk; });
        return { input, output, screen: () => screen };
    };
    const typed = make();
    const pending = readTokenInput({ input: typed.input, output: typed.output });
    assert.equal(typed.input.raw, true, 'raw mode: the terminal does not echo');
    for (const key of ['hf_', 'abcX', '\u007f', 'def', '\r']) typed.input.emit('data', key);
    assert.equal(await pending, 'hf_abcdef');
    assert.equal(typed.input.raw, false, 'raw mode is restored');
    assert.ok(!typed.screen().includes('hf_') && !typed.screen().includes('abc'), 'the token is not on the screen');
    const cancelled = make();
    const second = readTokenInput({ input: cancelled.input, output: cancelled.output });
    cancelled.input.emit('data', 'hf_x');
    cancelled.input.emit('data', '\u0003');
    assert.equal(await second, null);
});

// The token file is read by the host of the server, whose home is config.taskHome, else the Pworker home (serverHome); the command
// must write where the server reads, also when the user's configuration names a data folder elsewhere or a taskHome.
test('the command writes the token where the server reads it: a taskHome moves both, a dataDir moves neither', async (t) => {
    const servers = [];
    try {
        await writtenWhereRead(t, servers);
    } finally {
        for (const server of servers) await server.close();
    }
});

async function writtenWhereRead(t, servers) {
    const configured = async (layout) => {
        const home = scratch(t);
        const other = scratch(t);
        fs.writeFileSync(path.join(home, 'config.json'), JSON.stringify(layout(other)));
        const written = await bin(home, ['models', 'hf-token'], `${FIRST}\n`);
        assert.equal(written.code, 0, written.stderr);
        const env = { PWORKER_HOME: home };
        const { config } = loadLayers({ env, project: false });
        const server = await createPworkerServer({ config, env });
        servers.push(server);
        return { home, other, server };
    };
    // A data folder directly under some other folder (like "~/pw-data" under a home folder) must not make that folder the engine's home.
    const byData = await configured((other) => ({ dataDir: path.join(other, 'data') }));
    assert.equal(byData.server.localLlm.layout.home, byData.home);
    assert.equal(byData.server.localLlm.layout.keysFile, hfTokenFile(byData.home));
    assert.equal(fs.readFileSync(byData.server.localLlm.layout.keysFile, 'utf8'), `HF_TOKEN="${FIRST}"\n`, 'the server reads the file the command wrote');
    assert.equal(byData.server.localLlm.engineEnvironment().HF_TOKEN, FIRST);
    assert.equal(fs.existsSync(hfTokenFile(byData.other)), false, 'nothing is written beside the data folder');
    const byTask = await configured((other) => ({ taskHome: other }));
    assert.equal(byTask.server.localLlm.layout.home, byTask.other);
    assert.equal(byTask.server.localLlm.engineEnvironment().HF_TOKEN, FIRST);
    assert.equal(fs.existsSync(hfTokenFile(byTask.home)), false);
    // Without a configured home both are PWORKER_HOME.
    const plain = await configured(() => ({}));
    assert.equal(plain.server.localLlm.layout.home, plain.home);
    assert.equal(plain.server.localLlm.engineEnvironment().HF_TOKEN, FIRST);
    assert.equal(serverHome({ taskHome: '/x/y', dataDir: '/z/data' }, {}), '/x/y');
    assert.equal(serverHome({ dataDir: '/z/data' }, { PWORKER_HOME: '/h' }), '/h');
    assert.equal(serverHome({ dataDir: '/home/me/pw-data' }, { PWORKER_HOME: '/h' }), '/h', 'a data folder under a home folder does not make it the engine home');
    assert.equal(serverHome({}, { PWORKER_HOME: '/h' }), '/h');
}

test('the host\'s own log lines pass the token filter: a refusal that names the token in a path is logged with ***', async (t) => {
    const home = scratch(t);
    const lines = [];
    // An unusable socket directory is refused with a message that quotes the path; here the path carries the token.
    const host = createLocalLlmHost({
        config: { localLlm: { socketDir: `/tmp/not:ok-${FIRST}` } }, home, env: { PWORKER_HOME: home, HF_TOKEN: FIRST },
        platformName: 'linux', ownerPort: await freePort(), log: (line) => lines.push(line),
    });
    t.after(() => host.drain());
    await assert.rejects(host.acquire(), (error) => error.code === 'invalid_socket_dir');
    assert.ok(lines.length > 0, 'the refusal was logged');
    assert.ok(lines.every((line) => !line.includes(FIRST)), 'no log line holds the token');
    assert.ok(lines.some((line) => line.includes('***')), lines.join('\n'));
});

test('`pworker provider local-llm --endpoint ...` is refused and the engine\'s provider spec is not overwritten', async (t) => {
    const home = scratch(t);
    const result = await bin(home, ['provider', 'local-llm', '--endpoint', 'http://127.0.0.1:9/v1'], '');
    assert.notEqual(result.code, 0);
    assert.match(result.stderr, /local-llm engine of this machine/);
    const config = path.join(home, 'config.json');
    const written = fs.existsSync(config) ? JSON.parse(fs.readFileSync(config, 'utf8')) : {};
    assert.equal(written.providers?.['local-llm'], undefined, 'no user override of the engine provider was saved');
});
