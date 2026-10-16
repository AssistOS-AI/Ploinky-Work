// R-1: on a platform that refuses a loose model store (macOS, DS002) a deletion is refused too, before anything is removed: the
// directories above the files are the user's own and not writable by others, or the weights stay where they are. The platform is
// injected, so the file runs on Linux and macOS alike; the default platform (Linux) removes as it always did.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { darwinPlatform } from '../../lib/local-llm/darwinPlatform.mjs';
import { artifactPaths, removeArtifact, snapshotPaths } from '../../lib/local-llm/controller/downloader.mjs';
import { createWeightStores } from '../../lib/local-llm/controller/weightStores.mjs';
import { linuxPlatform } from '../../lib/local-llm/host.mjs';

const UID = process.getuid();
const COMMIT = 'a'.repeat(40);
const ARTIFACT = Object.freeze({ source: 'huggingface', repo: 'test/model', file: 'model.gguf', revision: 'main', commit: COMMIT, size: 10, sha256: 'b'.repeat(64) });

function scratch(t) {
    const dir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-removal-')));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

// A store holding one file, with every directory at `mode` (whatever the umask is).
function storeWith(t, mode = 0o755) {
    const root = path.join(scratch(t), 'models');
    const paths = artifactPaths({ root, artifact: ARTIFACT });
    fs.mkdirSync(paths.dir, { recursive: true });
    for (let dir = paths.dir; dir.length >= root.length; dir = path.dirname(dir)) fs.chmodSync(dir, mode);
    fs.writeFileSync(paths.file, '0123456789');
    return { root, paths };
}

const unsafe = { code: 'UNSAFE_PATH' };

test('darwin removal: a store that others can write is refused before anything is removed, whichever directory is loose', async (t) => {
    for (const loose of ['root', 'repo', 'commit']) {
        const { root, paths } = storeWith(t);
        const dir = { root, repo: path.dirname(paths.dir), commit: paths.dir }[loose];
        fs.chmodSync(dir, 0o777);
        await assert.rejects(removeArtifact({ root, artifact: ARTIFACT, platform: darwinPlatform({ privateRoot: root }) }), unsafe, loose);
        assert.equal(fs.readFileSync(paths.file, 'utf8'), '0123456789', `${loose}: the weights are still there`);
    }
});

test('darwin removal: another user\'s directory is refused, a store of the user\'s own is removed with its empty directories, and a missing root removes nothing', async (t) => {
    const { root, paths } = storeWith(t);
    await assert.rejects(removeArtifact({ root, artifact: ARTIFACT, platform: darwinPlatform({ privateRoot: root, uid: UID + 1 }) }), unsafe);
    assert.equal(fs.existsSync(paths.file), true);
    const freed = await removeArtifact({ root, artifact: ARTIFACT, platform: darwinPlatform({ privateRoot: root }) });
    assert.equal(freed, 10);
    assert.equal(fs.existsSync(paths.file), false);
    assert.equal(await removeArtifact({ root: path.join(scratch(t), 'absent'), artifact: ARTIFACT, platform: darwinPlatform() }), 0);
});

test('darwin removal: without a platform, or on Linux, a loose store is removed as it always was', async (t) => {
    for (const platform of [undefined, linuxPlatform()]) {
        const { root, paths } = storeWith(t, 0o777);
        const freed = await removeArtifact({ root, artifact: ARTIFACT, ...(platform ? { platform } : {}) });
        assert.equal(freed, 10);
        assert.equal(fs.existsSync(paths.file), false);
    }
});

test('darwin removal: Delete weights of a Hugging Face snapshot checks the store first, and the whole store is kept when it is loose', async (t) => {
    const dataDir = scratch(t);
    const hfRoot = path.join(dataDir, 'models', 'hf');
    const source = { type: 'hf-snapshot', repo: 'test/model', commit: COMMIT, revision: 'main', files: [{ path: 'model.gguf', size: 10, sha256: 'b'.repeat(64) }] };
    const { dir, stateDir } = snapshotPaths({ root: hfRoot, repo: source.repo, commit: source.commit });
    for (const target of [dir, stateDir]) {
        fs.mkdirSync(target, { recursive: true });
        // Every level, whatever the umask is.
        for (let current = target; current.length > dataDir.length; current = path.dirname(current)) fs.chmodSync(current, 0o755);
    }
    fs.writeFileSync(path.join(dir, 'model.gguf'), '0123456789');
    const stores = (platform) => createWeightStores({ dataDir, env: {}, platform, remove: removeArtifact });
    const looseStore = stores(darwinPlatform({ privateRoot: hfRoot }));
    fs.chmodSync(path.dirname(dir), 0o777);
    await assert.rejects(looseStore['hf-snapshot'].remove(source), unsafe);
    assert.equal(fs.existsSync(path.join(dir, 'model.gguf')), true, 'nothing was removed from a loose store');
    // The same store under the default platform is removed, and a tidy one is removed under the darwin platform.
    fs.chmodSync(path.dirname(dir), 0o755);
    assert.equal(await stores(darwinPlatform({ privateRoot: hfRoot }))['hf-snapshot'].remove(source), 10);
    assert.equal(fs.existsSync(path.join(dir, 'model.gguf')), false);
});
