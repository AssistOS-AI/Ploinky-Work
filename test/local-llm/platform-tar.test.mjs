// The extraction flags of the runner installer come from the platform: Linux keeps its argv byte for byte (`--touch`), and
// macOS gets `-m`, because bsdtar rejects `--touch`. The installer test runs the real `tar` of the machine it runs on with
// the macOS platform injected, so it runs on Linux (GNU tar accepts `-m`) and on macOS alike.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { darwinPlatform } from '../../lib/local-llm/darwinPlatform.mjs';
import { downloadFile } from '../../lib/local-llm/controller/downloader.mjs';
import { createRunnerInstaller, runTool } from '../../lib/local-llm/controller/runnerInstaller.mjs';
import { validateRunnerLock } from '../../lib/local-llm/controller/runnerLock.mjs';
import { linuxPlatform } from '../../lib/local-llm/host.mjs';

const SHA = (buffer) => crypto.createHash('sha256').update(buffer).digest('hex');

function tempDir(t, name) {
    const dir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), `local-llm-${name}-`)));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

test('platform-tar: Linux keeps the extraction argv it always had, byte for byte', () => {
    const platform = linuxPlatform();
    assert.deepEqual(platform.tarFlags('/opt/runners/x/1', 1), ['-C', '/opt/runners/x/1', '--strip-components=1', '--no-same-owner', '--touch']);
    assert.deepEqual(platform.tarFlags('/r', 0), ['-C', '/r', '--strip-components=0', '--no-same-owner', '--touch']);
});

test('platform-tar: macOS passes -m and never --touch', () => {
    const platform = darwinPlatform();
    assert.deepEqual(platform.tarFlags('/opt/runners/x/1', 1), ['-C', '/opt/runners/x/1', '--strip-components=1', '--no-same-owner', '-m']);
    assert.deepEqual(platform.tarFlags('/r', 0), ['-C', '/r', '--strip-components=0', '--no-same-owner', '-m']);
    assert.equal(platform.tarFlags('/r', 1).includes('--touch'), false);
});

// A .tar.gz whose top directory holds bin/runner and VERSION, and a lock for it.
function archiveFixture(t) {
    const dir = tempDir(t, 'platform-tar-src');
    const top = path.join(dir, 'runner-1.0.0');
    fs.mkdirSync(path.join(top, 'bin'), { recursive: true });
    fs.writeFileSync(path.join(top, 'bin', 'runner'), '#!/bin/sh\necho runner\n', { mode: 0o755 });
    fs.writeFileSync(path.join(top, 'VERSION'), '1.0.0\n');
    const archive = path.join(dir, 'runner-1.0.0.tar.gz');
    execFileSync('tar', ['-czf', archive, '-C', dir, 'runner-1.0.0']);
    const bytes = fs.readFileSync(archive);
    const lock = validateRunnerLock({
        schema: 'local-llm.runners-lock/v1',
        runners: { testrunner: {
            version: '1.0.0', kind: 'archive', licence: { name: 'MIT', url: 'https://github.com/example/runner/blob/main/LICENSE' },
            files: [{ name: 'runner-1.0.0.tar.gz', url: 'https://github.com/example/runner/releases/download/1.0.0/runner-1.0.0.tar.gz', size: bytes.length, sha256: SHA(bytes) }],
        } },
    });
    return { bytes, lock };
}

async function install(t, platform, { unpack = true } = {}) {
    // The store's directories get explicit modes: the macOS binder refuses a directory others can write.
    const previousMask = process.umask(0o022);
    t.after(() => process.umask(previousMask));
    const { bytes, lock } = archiveFixture(t);
    const server = http.createServer((req, res) => { res.writeHead(200, { 'Content-Length': bytes.length }); res.end(bytes); });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    t.after(() => server.close());
    const base = `http://127.0.0.1:${server.address().port}`;
    const root = tempDir(t, 'platform-tar-install');
    const steps = [];
    const installer = createRunnerInstaller({
        lock, cacheRoot: path.join(root, 'data', 'runners'), runRoot: path.join(root, 'opt', 'runners'), platform,
        download: (options) => downloadFile({ ...options, fetchImpl: (url, init) => fetch(`${base}/${path.basename(new URL(url).pathname)}`, init) }),
        statfs: async () => ({ bavail: 1e12, bsize: 1 }),
        // A stand-in that only records the command is for the test that must not run bsdtar with `--touch`.
        run: (options) => { steps.push(options); return unpack ? runTool(options) : { code: 0, signal: null, output: '', aborted: false }; },
    });
    await installer.fetchAll(installer.entryFor('testrunner'));
    await installer.ensureRunnable('testrunner');
    return { steps, runDir: path.join(root, 'opt', 'runners', 'testrunner', '1.0.0') };
}

test('platform-tar: the installer with the macOS platform unpacks with -m, and the archive is unpacked', async (t) => {
    const { steps, runDir } = await install(t, darwinPlatform());
    const untar = steps.find((step) => step.command === 'tar');
    assert.ok(untar, 'tar ran');
    assert.equal(untar.args[0], '-xzf');
    assert.ok(untar.args.includes('-m'));
    assert.equal(untar.args.includes('--touch'), false);
    assert.ok(untar.args.includes('--strip-components=1'));
    assert.ok(untar.args.includes('--no-same-owner'));
    assert.equal(fs.readFileSync(path.join(runDir, 'VERSION'), 'utf8'), '1.0.0\n');
    assert.equal(fs.existsSync(path.join(runDir, 'bin', 'runner')), true);
});

test('platform-tar: the installer passes the platform\'s flags unchanged (a macOS platform with the Linux tarFlags stands in for the Linux argv)', async (t) => {
    // The stand-in keeps the macOS binder, which runs here, and takes only tarFlags from Linux; the real Linux platform
    // is covered end to end by runner-installer.test.mjs, which asserts `--touch` on Linux. The unpack is not run: bsdtar
    // rejects `--touch`, which is why macOS has its own flag.
    const { steps } = await install(t, { ...darwinPlatform(), tarFlags: linuxPlatform().tarFlags }, { unpack: false });
    const untar = steps.find((step) => step.command === 'tar');
    assert.equal(untar.args[2], '-C');
    assert.deepEqual(untar.args.slice(-3), ['--strip-components=1', '--no-same-owner', '--touch']);
    assert.equal(untar.args.includes('-m'), false);
});
