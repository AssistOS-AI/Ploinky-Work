// The runner locks of the macOS hosts (lib/local-llm-host/locks): each one loads through the engine's validator with the
// launch file required, and pins the llama.cpp b11295 release asset by size and sha256. The numbers are written out: they
// were checked on 2026-10-16 by downloading each asset, hashing it, and reading the GitHub release's own listing.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { loadRunnerLock, loadRunnerLocks, validateRunnerLock } from '../../lib/local-llm/controller/runnerLock.mjs';
import { createRunners } from '../../lib/local-llm/runners/index.mjs';
import { defineHost } from '../../lib/local-llm/host.mjs';

const lockFile = (name) => fileURLToPath(new URL(`../../lib/local-llm-host/locks/${name}`, import.meta.url));
const ARM64 = lockFile('runners.lock.host-darwin-arm64.json');
const AMD64 = lockFile('runners.lock.host-darwin-amd64.json');

const EXPECTED = [
    [ARM64, 'llama.cpp', 'llama-b11295-bin-macos-arm64.tar.gz', 11_817_829, '1a0c5f60a3f2532a6d81c783bb18f11bac2925bc16dd605896b13fbabf4c7a56'],
    [AMD64, 'llama.cpp-cpu', 'llama-b11295-bin-macos-x64.tar.gz', 11_372_934, 'd14a2272a438aab97c0b184f4c371176ece01daf726a2589741d928f2fc0e456'],
];

test('darwin-locks: each lock loads with the launch file required and pins the b11295 asset by size and sha256', () => {
    for (const [file, id, name, size, sha256] of EXPECTED) {
        const lock = loadRunnerLock(file, { requireExecutable: true });
        assert.deepEqual(Object.keys(lock.runners), [id], file);
        const entry = lock.runners[id];
        assert.equal(entry.version, 'b11295');
        assert.equal(entry.kind, 'archive');
        assert.equal(entry.files.length, 1);
        const [asset] = entry.files;
        assert.equal(asset.name, name);
        assert.equal(asset.url, `https://github.com/ggml-org/llama.cpp/releases/download/b11295/${name}`);
        assert.equal(asset.size, size);
        assert.equal(asset.sha256, sha256);
        assert.equal(asset.strip, 1);
        assert.equal(entry.check.executable, 'llama-server');
        assert.equal(entry.totalBytes, size);
        assert.equal(entry.licence.name, 'MIT');
        assert.equal(entry.licence.requiresAcceptance, false);
        assert.match(entry.licence.url, /\/blob\/b11295\/LICENSE$/);
    }
});

test('darwin-locks: the lock is the host lock of its runner: the id has an adapter and the controller offers it for install', () => {
    const runners = createRunners({ host: defineHost({ dataDir: '/x/data', runRoot: '/x/run' }) });
    for (const [file, id] of EXPECTED) {
        assert.ok(Object.hasOwn(runners, id), `${id} has an adapter`);
        const locks = loadRunnerLocks({ image: file, agent: null });
        assert.equal(locks.origin[id], 'image');
    }
    // The Intel Mac's runner is the one whose adapter already pins the same tag.
    assert.equal(runners['llama.cpp-cpu'].pinnedVersion, loadRunnerLock(AMD64).runners['llama.cpp-cpu'].version);
});

test('darwin-locks: a changed digest, size, name, host or tag, or a missing launch file, does not validate', () => {
    const document = JSON.parse(fs.readFileSync(ARM64, 'utf8'));
    const variant = (change) => {
        const copy = structuredClone(document);
        change(copy.runners['llama.cpp']);
        return copy;
    };
    assert.doesNotThrow(() => validateRunnerLock(variant(() => {}), { requireExecutable: true }));
    for (const change of [
        (entry) => { entry.files[0].sha256 = entry.files[0].sha256.slice(1); },
        (entry) => { entry.files[0].size = 0; },
        (entry) => { entry.files[0].name = '../llama.tar.gz'; },
        (entry) => { entry.files[0].url = 'https://example.com/llama-b11295-bin-macos-arm64.tar.gz'; },
        (entry) => { entry.files[0].url = entry.files[0].url.replace('https:', 'http:'); },
        (entry) => { entry.files[0].strip = 2; },
        (entry) => { delete entry.check; },
    ]) {
        assert.throws(() => validateRunnerLock(variant(change), { requireExecutable: true }), { code: 'invalid_runner_lock' }, String(change));
    }
    // The tag appears in the name, the URL and the licence link of each lock, and the three agree.
    for (const [file] of EXPECTED) {
        const text = fs.readFileSync(file, 'utf8');
        assert.deepEqual([...new Set(text.match(/b\d{4,6}/g))], ['b11295'], file);
    }
});
