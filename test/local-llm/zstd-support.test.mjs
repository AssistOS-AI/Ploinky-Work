// A .tar.zst runner needs Node's zstd decoder (zlib.createZstdDecompress, from Node 22.15). The package supports
// Node 22.13 and later, so the installer checks at run time and refuses cleanly where it is missing, before it
// downloads or unpacks anything. A zlib stand-in without the function plays the older Node.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import zlib from 'node:zlib';

import { createRunnerInstaller } from '../../lib/local-llm/controller/runnerInstaller.mjs';
import { validateRunnerLock } from '../../lib/local-llm/controller/runnerLock.mjs';
import { LocalLlmError } from '../../lib/local-llm/errors.mjs';

function tempDir(t, name) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), `local-llm-${name}-`));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

const SHA = (buffer) => crypto.createHash('sha256').update(buffer).digest('hex');
const OLD_ZLIB = Object.freeze({ ...zlib, createZstdDecompress: undefined });

function entryOf(name, size) {
    return {
        version: '1.0.0', kind: 'archive', licence: { name: 'MIT', url: 'https://example.org/LICENSE' },
        files: [{ name, url: `https://github.com/example/r/releases/download/1.0.0/${name}`, size, sha256: 'a'.repeat(64), strip: 0 }],
        check: { executable: 'llama-server' },
    };
}

// An installer whose download and tools are spies, over a cache that already holds the file.
function harness(t, { name = 'runner-1.0.0.tar.zst', ...options } = {}) {
    const root = tempDir(t, 'zstd');
    const bytes = Buffer.from('not decoded in these tests');
    const lock = validateRunnerLock({ schema: 'local-llm.runners-lock/v1', runners: { testrunner: { ...entryOf(name, bytes.length), files: [{ ...entryOf(name, bytes.length).files[0], sha256: SHA(bytes) }] } } });
    const calls = { download: 0, run: 0 };
    const cacheRoot = path.join(root, 'data', 'runners');
    const runRoot = path.join(root, 'opt', 'runners');
    const installer = createRunnerInstaller({
        lock, cacheRoot, runRoot,
        download: async () => { calls.download += 1; return { bytesTransferred: 0 }; },
        inspect: async ({ target }) => (fs.existsSync(target) ? { state: 'complete', bytes: bytes.length } : { state: 'absent', bytes: 0 }),
        run: async () => { calls.run += 1; return { code: 0, output: '' }; },
        ...options,
    });
    const entry = installer.entryFor('testrunner');
    const files = path.join(cacheRoot, 'testrunner', '1.0.0', 'files');
    return {
        installer, calls, entry, runDir: path.join(runRoot, 'testrunner', '1.0.0'), stageDir: path.join(runRoot, '.stage-testrunner'),
        seedCache() {
            fs.mkdirSync(files, { recursive: true });
            fs.writeFileSync(path.join(files, name), bytes);
            fs.writeFileSync(path.join(cacheRoot, 'testrunner', '1.0.0', 'installed.json'), JSON.stringify({ id: 'testrunner', version: '1.0.0', digest: entry.digest }));
        },
    };
}

const refusal = (error) => error instanceof LocalLlmError && error.code === 'zstd_unsupported';

test('zstd: where the decoder is missing, a .tar.zst install is refused with a stable code before any download', async (t) => {
    const h = harness(t, { zlib: OLD_ZLIB });
    await assert.rejects(() => h.installer.fetchAll(h.entry), (error) => {
        assert.ok(refusal(error), String(error.code));
        assert.deepEqual(error.details, { runner: 'testrunner', file: 'runner-1.0.0.tar.zst' });
        assert.match(error.message, /^testrunner 1\.0\.0 needs runner-1\.0\.0\.tar\.zst unpacked, and this Node\.js \(v\d+\.\d+\.\d+[^)]*\) cannot decompress zstd; use Node\.js 22\.15 or newer\.$/);
        assert.doesNotMatch(error.message, /ploinky|agent|container/i);
        return true;
    });
    assert.equal(h.calls.download, 0, 'nothing was downloaded');
    assert.equal(h.calls.run, 0);
});

test('zstd: where the decoder is missing, building a cached .tar.zst runner is refused and leaves no copy, marker or staging', async (t) => {
    const h = harness(t, { zlib: OLD_ZLIB });
    h.seedCache();
    await assert.rejects(() => h.installer.ensureRunnable('testrunner'), refusal);
    assert.equal(h.calls.run, 0, 'no tool ran');
    assert.equal(fs.existsSync(path.join(h.runDir, '.ready.json')), false);
    assert.equal(fs.existsSync(h.stageDir), false);
});

test('zstd: the refusal comes from the missing function and not from the Node version, so a Node that has it is not refused', async (t) => {
    assert.equal(typeof zlib.createZstdDecompress, 'function', 'this test needs a Node with zstd');
    const real = harness(t);
    await real.installer.fetchAll(real.entry);
    assert.equal(real.calls.download, 1);
    // A decoder bound by the caller is used whether or not zlib has the function.
    const bound = harness(t, { zlib: OLD_ZLIB, zstdDecoder: () => zlib.createZstdDecompress() });
    await bound.installer.fetchAll(bound.entry);
    assert.equal(bound.calls.download, 1);
});

test('zstd: a runner whose archive is .tar.gz is not affected by the missing decoder', async (t) => {
    const h = harness(t, { name: 'runner-1.0.0.tar.gz', zlib: OLD_ZLIB });
    await h.installer.fetchAll(h.entry);
    assert.equal(h.calls.download, 1);
});
