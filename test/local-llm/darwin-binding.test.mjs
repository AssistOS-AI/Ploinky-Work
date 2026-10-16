// The macOS download binder (DS002): a no-follow open followed by a stat comparison of the handle, the anchored root, every
// directory below it and the final path. The tests run on the real file system with the platform injected, so they run
// on Linux and on macOS alike; nothing here reads /proc.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { darwinPlatform, platformFor } from '../../lib/local-llm/darwinPlatform.mjs';
import { downloadArtifact, downloadFile, downloadSnapshotFile, snapshotPaths, verifyArtifact } from '../../lib/local-llm/controller/downloader.mjs';
import { anchorRoot, anchorStore, createBound, openBound, openPartialBound, walkShared } from '../../lib/local-llm/controller/workspaceReuse.mjs';
import { linuxPlatform } from '../../lib/local-llm/host.mjs';

const UID = process.getuid();

// A real directory with no symbolic link in its path (/var is a link on macOS), removed after the test.
function scratch(t) {
    const dir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-darwin-binding-')));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

// A directory with an explicit mode on every level it creates, whatever the umask is.
function directory(...parts) {
    const dir = path.join(...parts);
    const missing = [];
    for (let current = dir; !fs.existsSync(current); current = path.dirname(current)) missing.unshift(current);
    for (const level of missing) {
        fs.mkdirSync(level);
        fs.chmodSync(level, 0o755);
    }
    return dir;
}

// The real file system, except that `after` runs once, right after the first open returns: the moment a swap is dangerous.
function swappingFs(after) {
    let done = false;
    return { ...fs, promises: { ...fs.promises, open: async (...args) => {
        const handle = await fs.promises.open(...args);
        if (!done) {
            done = true;
            await after();
        }
        return handle;
    } } };
}

async function store(t, { platform = darwinPlatform() } = {}) {
    const root = directory(scratch(t), 'models');
    const sub = directory(root, 'sub');
    return { root, sub, anchor: await anchorStore(fs, root, { platform }), platform };
}

const unsafe = { code: 'UNSAFE_PATH' };

test('darwin-binding: an ordinary store file opens, and the handle reads it', async (t) => {
    const { sub, anchor } = await store(t);
    const file = path.join(sub, 'weights.gguf');
    fs.writeFileSync(file, 'hello');
    const opened = await openBound(fs, anchor, file);
    assert.ok(opened.handle, JSON.stringify(opened));
    assert.equal(opened.stat.size, 5);
    assert.equal((await opened.handle.readFile()).toString(), 'hello');
    await opened.handle.close();
});

test('darwin-binding: a created staging file is bound, and a partial file resumes', async (t) => {
    const { sub, anchor } = await store(t);
    const staging = path.join(sub, 'x.staging');
    const created = await createBound(fs, anchor, staging);
    await created.writeFile('abc');
    await created.close();
    assert.equal(fs.readFileSync(staging, 'utf8'), 'abc');
    // The same path again is O_EXCL: an existing file is never reused as a new one.
    await assert.rejects(createBound(fs, anchor, staging), { code: 'EEXIST' });
    const partial = path.join(sub, 'x.partial');
    fs.writeFileSync(partial, 'start-');
    const resumed = await openPartialBound(fs, anchor, partial);
    await resumed.write('end');
    await resumed.close();
    assert.equal(fs.readFileSync(partial, 'utf8'), 'start-end');
    const fresh = await openPartialBound(fs, anchor, path.join(sub, 'new.partial'));
    await fresh.close();
    assert.equal(fs.existsSync(path.join(sub, 'new.partial')), true);
});

test('darwin-binding: a symbolic link as the last path part is rejected', async (t) => {
    const { root, sub, anchor } = await store(t);
    const target = path.join(path.dirname(root), 'outside.gguf');
    fs.writeFileSync(target, 'secret');
    const link = path.join(sub, 'link.gguf');
    fs.symlinkSync(target, link);
    assert.deepEqual(await openBound(fs, anchor, link), { rejected: 'symlink' });
    // The link is also refused as a place to create or append: O_NOFOLLOW, not a follow.
    await assert.rejects(openPartialBound(fs, anchor, link), { code: 'ELOOP' });
    assert.equal(fs.readFileSync(target, 'utf8'), 'secret');
});

test('darwin-binding: a file reached through a symbolic link to a directory is rejected', async (t) => {
    const { root, sub, anchor } = await store(t);
    fs.writeFileSync(path.join(sub, 'weights.gguf'), 'hello');
    fs.symlinkSync(sub, path.join(root, 'alias'));
    const opened = await openBound(fs, anchor, path.join(root, 'alias', 'weights.gguf'));
    assert.equal(opened.handle, undefined);
    assert.match(opened.rejected, /symbolic link/);
    // The honest path still opens.
    const honest = await openBound(fs, anchor, path.join(sub, 'weights.gguf'));
    assert.ok(honest.handle);
    await honest.handle.close();
});

test('darwin-binding: a directory swapped for another between the open and the check is rejected because the inodes differ', async (t) => {
    const { sub, anchor } = await store(t);
    const file = path.join(sub, 'weights.gguf');
    fs.writeFileSync(file, 'hello');
    // The replacement holds a regular file of the same name, so only the device and inode comparison can tell.
    const swapped = swappingFs(() => {
        fs.renameSync(sub, `${sub}.old`);
        directory(sub);
        fs.writeFileSync(file, 'hello');
    });
    const opened = await openBound(swapped, anchor, file);
    assert.equal(opened.handle, undefined);
    assert.ok(opened.rejected, 'the swapped file is not bound');
    assert.equal(fs.statSync(file).ino !== fs.statSync(path.join(`${sub}.old`, 'weights.gguf')).ino, true, 'the fixture really swapped files');
});

test('darwin-binding: the anchored root swapped for another directory is rejected', async (t) => {
    const { root, sub, anchor } = await store(t);
    const file = path.join(sub, 'weights.gguf');
    fs.writeFileSync(file, 'hello');
    // The new tree links the very file that was opened, so the final inode matches and only the root's identity can tell.
    const swapped = swappingFs(() => {
        fs.renameSync(root, `${root}.old`);
        directory(sub);
        fs.linkSync(path.join(`${root}.old`, 'sub', 'weights.gguf'), file);
    });
    const opened = await openBound(swapped, anchor, file);
    assert.equal(opened.handle, undefined);
    assert.ok(opened.rejected);
    assert.equal(fs.statSync(file).ino, fs.statSync(path.join(`${root}.old`, 'sub', 'weights.gguf')).ino, 'the fixture links the same file');
});

test('darwin-binding: a created file in a directory swapped right after the create is refused and closed', async (t) => {
    const { sub, anchor } = await store(t);
    const staging = path.join(sub, 'x.staging');
    const swapped = swappingFs(() => {
        fs.renameSync(sub, `${sub}.old`);
        directory(sub);
    });
    await assert.rejects(createBound(swapped, anchor, staging), unsafe);
    // The file that was created stayed inside the store's own (renamed) directory.
    assert.equal(fs.existsSync(path.join(`${sub}.old`, 'x.staging')), true);
    assert.equal(fs.existsSync(staging), false);
});

test('darwin-binding: a parent that is a symbolic link out of the store is refused before anything is created', async (t) => {
    const { root, sub, anchor } = await store(t);
    const outside = directory(path.dirname(root), 'outside');
    fs.rmSync(sub, { recursive: true });
    fs.symlinkSync(outside, sub);
    await assert.rejects(createBound(fs, anchor, path.join(sub, 'x.staging')), unsafe);
    await assert.rejects(openPartialBound(fs, anchor, path.join(sub, 'x.partial')), unsafe);
    assert.deepEqual(fs.readdirSync(outside), [], 'nothing was created outside the store');
});

test('darwin-binding: a parent that others can write is refused before anything is created, and mode 755 is accepted', async (t) => {
    const { sub, anchor } = await store(t);
    fs.chmodSync(sub, 0o775);
    await assert.rejects(createBound(fs, anchor, path.join(sub, 'x.staging')), { code: 'UNSAFE_PATH', message: /group or by others/ });
    await assert.rejects(openPartialBound(fs, anchor, path.join(sub, 'x.partial')), { code: 'UNSAFE_PATH', message: /group or by others/ });
    assert.deepEqual(fs.readdirSync(sub), []);
    fs.chmodSync(sub, 0o757);
    await assert.rejects(createBound(fs, anchor, path.join(sub, 'x.staging')), unsafe);
    fs.chmodSync(sub, 0o755);
    await (await createBound(fs, anchor, path.join(sub, 'x.staging'))).close();
});

test('darwin-binding: a store directory owned by another user is refused before anything is created', async (t) => {
    const { sub, anchor } = await store(t, { platform: darwinPlatform({ uid: UID + 1 }) });
    await assert.rejects(createBound(fs, anchor, path.join(sub, 'x.staging')), { code: 'UNSAFE_PATH', message: /not owned by the user/ });
    assert.deepEqual(fs.readdirSync(sub), []);
});

test('darwin-binding: the private root and every directory down from it are checked, nothing above it is', async (t) => {
    const base = scratch(t);
    const home = directory(base, 'home');
    const models = directory(home, 'local-llm', 'models');
    const sub = directory(models, 'sub');
    const platform = darwinPlatform({ privateRoot: home });
    const anchor = await anchorStore(fs, models, { platform });
    await (await createBound(fs, anchor, path.join(sub, 'a.staging'))).close();
    // The directory between the private root and the store.
    fs.chmodSync(path.join(home, 'local-llm'), 0o777);
    await assert.rejects(createBound(fs, anchor, path.join(sub, 'b.staging')), { code: 'UNSAFE_PATH', message: /local-llm can be written/ });
    fs.chmodSync(path.join(home, 'local-llm'), 0o755);
    // The private root itself.
    fs.chmodSync(home, 0o775);
    await assert.rejects(createBound(fs, anchor, path.join(sub, 'b.staging')), { code: 'UNSAFE_PATH', message: /home can be written/ });
    fs.chmodSync(home, 0o755);
    // Above the private root nothing is looked at (a shared temporary directory is writable by others).
    fs.chmodSync(base, 0o777);
    await (await createBound(fs, anchor, path.join(sub, 'b.staging'))).close();
    // Without a private root, only the anchored root and below count.
    const bare = await anchorStore(fs, models, { platform: darwinPlatform() });
    fs.chmodSync(home, 0o775);
    await (await createBound(fs, bare, path.join(sub, 'c.staging'))).close();
    fs.chmodSync(home, 0o755);
});

test('darwin-binding: an untrusted root is never anchored, so a shared root is never walked', async (t) => {
    const shared = directory(scratch(t), 'shared');
    fs.writeFileSync(path.join(shared, 'm.gguf'), 'hello');
    const platform = darwinPlatform();
    assert.equal(platform.untrustedRoots, false);
    assert.equal(await anchorRoot(fs, shared, { untrusted: true, platform }), null);
    // The trusted store root is anchored as before, and the Linux platform still anchors an untrusted root.
    assert.ok(await anchorRoot(fs, shared, { platform }));
    assert.ok(await anchorRoot(fs, shared, { untrusted: true, platform: linuxPlatform() }));
    const found = await walkShared(fs, [shared], [5], undefined, { platform });
    assert.deepEqual([...found.entries()], [[5, []]], 'a candidate that is there is not found');
});

test('darwin-binding: the descriptor-path strategies are unreachable and never invent a path', async () => {
    const platform = darwinPlatform();
    await assert.rejects(platform.openedPathOf(fs, 3), { code: 'unsupported_platform' });
    assert.throws(() => platform.openatPath(3, 'x'), { code: 'unsupported_platform' });
    assert.equal(platform.unreadableReason, 'macOS has no /proc/self/fd');
    assert.equal(platform.name, 'darwin');
    assert.equal(platform.procDir, '/nonexistent/proc');
    assert.equal(platform.devDriDir, '/nonexistent/dri');
    assert.equal(platform.setOomScoreAdj(1, 1000, { fsApi: null }), undefined);
    assert.throws(() => darwinPlatform({ privateRoot: 'relative' }), { code: 'invalid_host' });
});

test('darwin-binding: platformFor chooses linux and darwin and refuses every other platform', () => {
    assert.equal(platformFor('linux').name, 'linux');
    assert.equal(platformFor('darwin').name, 'darwin');
    assert.equal(platformFor('darwin', { privateRoot: '/x/home' }).name, 'darwin');
    for (const name of ['win32', 'freebsd', 'aix', '', 'Linux', 'linux ']) {
        assert.throws(() => platformFor(name), { code: 'unsupported_platform' }, JSON.stringify(name));
    }
    assert.equal(platformFor().name, process.platform === 'darwin' ? 'darwin' : 'linux');
});

test('darwin-binding: a handle that is not a regular file is not bound', async (t) => {
    const { sub, anchor } = await store(t);
    const handle = await fs.promises.open(sub, fs.constants.O_RDONLY | fs.constants.O_DIRECTORY);
    try {
        assert.equal(await darwinPlatform().confirmOpened(fs, handle, path.join(sub, 'x'), anchor), false);
    } finally {
        await handle.close();
    }
    await assert.rejects(darwinPlatform().confirmOpened(fs, handle, path.join(sub, 'x'), null), unsafe);
});

// ------------------------------------------------------------------ the whole download, not only the open

const PAYLOAD = Buffer.from('tiny model bytes '.repeat(64));
const ARTIFACT = Object.freeze({
    source: 'huggingface', repo: 'owner/repo', file: 'm.gguf', revision: 'main', commit: 'c'.repeat(40),
    size: PAYLOAD.length, sha256: crypto.createHash('sha256').update(PAYLOAD).digest('hex'),
});
const bigDisk = async () => ({ bavail: 1e12, bsize: 1 });

// A fetch that counts its calls and answers with the payload (honoring a Range header).
function countingFetch() {
    const calls = [];
    const fetchImpl = async (url, init = {}) => {
        calls.push(String(url));
        const range = /^bytes=(\d+)-$/.exec(init.headers?.Range ?? init.headers?.range ?? '');
        const start = range ? Number(range[1]) : 0;
        const headers = { 'content-length': String(PAYLOAD.length - start) };
        if (range) headers['content-range'] = `bytes ${start}-${PAYLOAD.length - 1}/${PAYLOAD.length}`;
        return new Response(PAYLOAD.subarray(start), { status: range ? 206 : 200, headers });
    };
    return { calls, fetchImpl };
}

// What is under `dir`, with the type and the mode, so a test can say that nothing was added, removed or changed.
function listing(dir) {
    const out = [];
    const walk = (current) => {
        for (const name of fs.readdirSync(current).sort()) {
            const full = path.join(current, name);
            const stats = fs.lstatSync(full);
            out.push(`${path.relative(dir, full)} ${stats.isDirectory() ? 'dir' : `file ${stats.size}`}`);
            if (stats.isDirectory()) walk(full);
        }
    };
    walk(dir);
    return out;
}

test('darwin-binding: downloadArtifact refuses a commit directory others can write before it removes, writes or fetches anything', async (t) => {
    const root = directory(scratch(t), 'models');
    const dir = directory(root, 'owner', 'repo', ARTIFACT.commit);
    // A stale staging file a crash left: the refusal must not remove it either.
    fs.writeFileSync(path.join(dir, 'm.gguf.partial.copy-1'), 'stale');
    fs.chmodSync(dir, 0o775);
    const before = listing(root);
    const { calls, fetchImpl } = countingFetch();
    await assert.rejects(downloadArtifact({ artifact: ARTIFACT, root, fetchImpl, statfs: bigDisk, platform: darwinPlatform() }),
        { code: 'UNSAFE_PATH', message: /group or by others/ });
    assert.equal(calls.length, 0);
    assert.deepEqual(listing(root), before, 'nothing was added, removed or changed');
    assert.equal(fs.existsSync(path.join(dir, 'm.gguf.partial.json')), false);
    assert.equal(fs.existsSync(path.join(dir, 'm.gguf.partial')), false);
    // The same call with the directory at 755 downloads, so the refusal was the mode.
    fs.chmodSync(dir, 0o755);
    const ok = await downloadArtifact({ artifact: ARTIFACT, root, fetchImpl, statfs: bigDisk, platform: darwinPlatform() });
    assert.equal(ok.status, 'complete');
    assert.equal(fs.readFileSync(path.join(dir, 'm.gguf')).equals(PAYLOAD), true);
    assert.equal(calls.length, 1);
});

test('darwin-binding: directories the download itself creates are checked before the identity record is written', async (t) => {
    // A umask of 002 makes the directories created for the artifact group-writable; the second check refuses before any write.
    const previous = process.umask(0o002);
    t.after(() => process.umask(previous));
    const root = directory(scratch(t), 'models');
    const before = listing(root);
    const { calls, fetchImpl } = countingFetch();
    await assert.rejects(downloadArtifact({ artifact: ARTIFACT, root, fetchImpl, statfs: bigDisk, platform: darwinPlatform() }),
        { code: 'UNSAFE_PATH', message: /group or by others/ });
    assert.equal(calls.length, 0);
    const files = listing(root).filter((entry) => entry.includes(' file '));
    assert.deepEqual(files, [], 'no identity record, partial or staging file');
    assert.ok(listing(root).length >= before.length);
});

test('darwin-binding: downloadFile refuses a target directory others can write, with no request and no file left', async (t) => {
    const dir = directory(scratch(t), 'files');
    fs.writeFileSync(path.join(dir, 'runner.whl.partial.copy-1'), 'stale');
    fs.chmodSync(dir, 0o775);
    const before = listing(dir);
    const { calls, fetchImpl } = countingFetch();
    const options = { url: 'https://github.com/example/runner/releases/download/1/runner.whl', size: PAYLOAD.length, sha256: ARTIFACT.sha256,
        target: path.join(dir, 'runner.whl'), fetchImpl, statfs: bigDisk, platform: darwinPlatform() };
    await assert.rejects(downloadFile(options), { code: 'UNSAFE_PATH' });
    assert.equal(calls.length, 0);
    assert.deepEqual(listing(dir), before);
    fs.chmodSync(dir, 0o755);
    assert.equal((await downloadFile(options)).bytesTransferred, PAYLOAD.length);
    assert.equal(calls.length, 1);
});

test('darwin-binding: a snapshot file and a verify of a stored file are checked the same way', async (t) => {
    const root = directory(scratch(t), 'hf');
    const commit = 'd'.repeat(40);
    const artifact = { repo: 'owner/repo', commit, file: 'config.json', size: PAYLOAD.length, sha256: ARTIFACT.sha256 };
    const { dir, stateDir } = snapshotPaths({ root, repo: 'owner/repo', commit });
    directory(dir);
    directory(stateDir);
    fs.chmodSync(stateDir, 0o775);
    const before = listing(root);
    const { calls, fetchImpl } = countingFetch();
    await assert.rejects(downloadSnapshotFile({ artifact, root, fetchImpl, statfs: bigDisk, platform: darwinPlatform() }),
        { code: 'UNSAFE_PATH' });
    assert.equal(calls.length, 0);
    assert.deepEqual(listing(root), before);
    // A stored file in a directory others can write is not verified either: verifying may rewrite its record.
    const store = directory(scratch(t), 'models');
    const home = directory(store, 'owner', 'repo', ARTIFACT.commit);
    fs.writeFileSync(path.join(home, 'm.gguf'), PAYLOAD);
    fs.writeFileSync(path.join(home, 'm.gguf.json'), JSON.stringify({ ...ARTIFACT }));
    fs.chmodSync(home, 0o775);
    await assert.rejects(verifyArtifact({ root: store, artifact: ARTIFACT, platform: darwinPlatform() }), { code: 'UNSAFE_PATH' });
});
