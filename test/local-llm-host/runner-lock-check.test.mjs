// The runner-lock-check workflow and the checks it runs (lib/local-llm-host/lockcheck.mjs, elf.mjs, macho.mjs). The real proof of a lock
// is the workflow's own run on a runner of each OS and architecture; these tests keep the workflow honest about WHAT it runs (every host
// lock, Linux and darwin, never skipped, pinned actions, no push trigger) and keep the checks able to fail.
import '../home.mjs';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { classifyMissingLibraries } from '../../lib/local-llm/tools/runnerInstallCheck.mjs';
import { loadRunnerLock } from '../../lib/local-llm/controller/runnerLock.mjs';
import { findElfFiles, strictLibraryCheck } from '../../lib/local-llm-host/elf.mjs';
import { strictLibraryCheck as strictMachoCheck } from '../../lib/local-llm-host/macho.mjs';
import { checkLock, checksFor, reportProblems } from '../../lib/local-llm-host/lockcheck.mjs';
import { scratch } from './fixtures/engine.mjs';

const ROOT = fileURLToPath(new URL('../..', import.meta.url));
const WORKFLOW = path.join(ROOT, '.github', 'workflows', 'runner-lock-check.yml');
const LOCK_DIR = path.join(ROOT, 'lib', 'local-llm-host', 'locks');
const text = fs.readFileSync(WORKFLOW, 'utf8');
const lockFiles = fs.readdirSync(LOCK_DIR).filter((name) => /^runners\.lock\.host-linux-[a-z0-9]+\.json$/.test(name)).sort();
const darwinLockFiles = fs.readdirSync(LOCK_DIR).filter((name) => /^runners\.lock\.host-darwin-[a-z0-9]+\.json$/.test(name)).sort();

test('the workflow runs on pull_request and workflow_dispatch only, with read-only contents, and no job is conditional', () => {
    const triggers = /^on:\n((?: {2}.*\n)+)/m.exec(text)[1].split('\n').filter(Boolean).map((line) => line.trim().replace(/:$/, ''));
    assert.deepEqual(triggers, ['pull_request', 'workflow_dispatch']);
    assert.match(text, /^permissions:\n {2}contents: read\n/m);
    assert.ok(!/^\s*paths(-ignore)?:/m.test(text), 'no path filter');
    assert.ok(!/^\s+if: .*(needed|changes|paths)/m.test(text), 'no change detection');
    // Only the final job and the always-run steps carry an `if`, each `always()`, except the Linux-only disk step (on the OS, never on a result).
    const conditions = [...text.matchAll(/^\s+if: (.*)$/gm)].map((match) => match[1]);
    assert.ok(conditions.length > 0 && conditions.every((condition) => condition === '${{ always() }}' || condition === "${{ runner.os == 'Linux' }}"), conditions.join(' | '));
    assert.match(text, /- name: Free runner disk\n(?: {8}#.*\n)* {8}if: \$\{\{ runner\.os == 'Linux' \}\}/, 'the disk step is Linux-only');
});

test('every action is pinned to a full commit sha, and the pull request head is the revision under test', () => {
    const uses = [...text.matchAll(/^\s+uses: (\S+)(?: # (\S+))?$/gm)];
    assert.ok(uses.length >= 3);
    for (const [, action, tag] of uses) {
        assert.match(action, /^[\w.-]+\/[\w.-]+@[0-9a-f]{40}$/, action);
        assert.ok(tag, `${action} names its tag in a comment`);
    }
    assert.match(text, /actions\/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4/);
    assert.match(text, /actions\/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4/);
    assert.match(text, /ref: \$\{\{ github\.event\.pull_request\.head\.sha \|\| github\.sha \}\}/);
    assert.match(text, /persist-credentials: false/);
    assert.match(text, /git rev-parse HEAD/);
    assert.match(text, /GITHUB_SHA/);
});

test('the matrix lists exactly the host locks that exist (Linux and darwin), one leg each, on a runner of that architecture', () => {
    const listed = [...text.matchAll(/^\s+lock: (\S+)$/gm)].map((match) => match[1]).sort();
    assert.deepEqual(listed, [...lockFiles, ...darwinLockFiles].sort().map((name) => `lib/local-llm-host/locks/${name}`));
    assert.deepEqual(lockFiles, ['runners.lock.host-linux-amd64.json', 'runners.lock.host-linux-arm64.json']);
    assert.deepEqual(darwinLockFiles, ['runners.lock.host-darwin-amd64.json', 'runners.lock.host-darwin-arm64.json']);
    const runsOn = Object.fromEntries([...text.matchAll(/- arch: ([\w-]+)\n\s+machine: (\w+)\n\s+runs-on: (\S+)\n\s+lock: (\S+)\n/g)].map((match) => [match[1], { machine: match[2], runsOn: match[3], lock: path.basename(match[4]) }]));
    assert.deepEqual(runsOn, {
        amd64: { machine: 'x86_64', runsOn: 'ubuntu-24.04', lock: 'runners.lock.host-linux-amd64.json' },
        arm64: { machine: 'aarch64', runsOn: 'ubuntu-24.04-arm', lock: 'runners.lock.host-linux-arm64.json' },
        'darwin-arm64': { machine: 'arm64', runsOn: 'macos-15', lock: 'runners.lock.host-darwin-arm64.json' },
        'darwin-x64': { machine: 'x86_64', runsOn: 'macos-15-intel', lock: 'runners.lock.host-darwin-amd64.json' },
    });
    assert.match(text, /\[ "\$\(uname -m\)" = "\$MACHINE" \]/, 'a wrong label fails instead of running under Rosetta');
    assert.match(text, /fail-fast: false/);
    assert.match(text, /needs: \[install-check\]/, 'the final job waits for every architecture');
    assert.match(text, /\[ "\$INSTALLED" = success \] \|\| /, 'and fails unless the matrix succeeded');
});

test('the step runs lockcheck.mjs on the matrix lock with a work and an out folder, and expects the pinned build', () => {
    assert.match(text, /node lib\/local-llm-host\/lockcheck\.mjs --lock "\$LOCK" --out "\$out" --work "\$RUNNER_TEMP\/lockcheck-work" --expect-version "\$EXPECT_VERSION"/);
    assert.match(text, /EXPECT_VERSION: b11295/);
    assert.match(text, /node-version: 24/);
    assert.ok(fs.existsSync(path.join(ROOT, 'lib', 'local-llm-host', 'lockcheck.mjs')));
});

test('each host lock validates with the engine\'s validator, requires an executable and pins b11295 for every entry', () => {
    for (const name of [...lockFiles, ...darwinLockFiles]) {
        const lock = loadRunnerLock(path.join(LOCK_DIR, name), { requireExecutable: true });
        const ids = Object.keys(lock.runners).sort();
        // Linux locks carry the CUDA and the CPU build; the darwin-arm64 lock the Metal build, the darwin-amd64 lock the CPU build.
        assert.deepEqual(ids, name.includes('darwin-arm64') ? ['llama.cpp'] : name.includes('darwin-amd64') ? ['llama.cpp-cpu'] : ['llama.cpp', 'llama.cpp-cpu'], name);
        for (const id of ids) {
            assert.equal(lock.runners[id].version, 'b11295', `${name} ${id}`);
            assert.equal(lock.runners[id].kind, 'archive');
            assert.equal(typeof lock.runners[id].check.executable, 'string', `${name} ${id} names the file it launches`);
        }
        // The Linux CUDA entry ships its runtime in a second archive; the macOS entries are one archive each.
        assert.equal(lock.runners[ids[0]].files.length, name.includes('linux') && ids[0] === 'llama.cpp' ? 2 : 1, name);
    }
});

test('a missing, empty or unreadable lock is a failure, not an empty success', async (t) => {
    const out = scratch(t);
    const never = async () => { throw new Error('no entry to check'); };
    const missing = await checkLock({ lockFile: path.join(out, 'nope.json'), outDir: path.join(out, 'a'), workDir: path.join(out, 'w'), installCheck: never });
    assert.equal(missing.ok, false);
    assert.match(missing.problems.join(' '), /cannot be read/);
    const empty = path.join(out, 'empty.json');
    fs.writeFileSync(empty, JSON.stringify({ schema: 'local-llm.runners-lock/v1', runners: {} }));
    const none = await checkLock({ lockFile: empty, outDir: path.join(out, 'b'), workDir: path.join(out, 'w'), installCheck: never });
    assert.equal(none.ok, false);
    assert.match(none.problems.join(' '), /no entries/);
    assert.equal(JSON.parse(fs.readFileSync(path.join(out, 'b', 'summary.json'), 'utf8')).ok, false);
});

const GOOD_REPORT = { runnerId: 'x', version: 'b11295', ok: true, problems: [], executable: { path: 'llama-server', found: true, version: { status: 0, output: 'version: 11295', expected: '11295' } } };

test('reportProblems accepts a real install and refuses a skipped, failed or other-version one', () => {
    assert.deepEqual(reportProblems(GOOD_REPORT, { expectVersion: 'b11295' }), []);
    assert.match(reportProblems({ ...GOOD_REPORT, ok: false, problems: ['boom'] }).join(), /did not pass.*boom/);
    assert.match(reportProblems({ ...GOOD_REPORT, downloaded: false, skipped: 'validate-only' }).join(), /nothing was downloaded/);
    assert.match(reportProblems({ ...GOOD_REPORT, version: 'b1' }, { expectVersion: 'b11295' }).join(), /not b11295/);
    assert.match(reportProblems({ ...GOOD_REPORT, executable: { found: false } }).join(), /not found/);
    assert.match(reportProblems({ ok: false, error: 'not_installable', message: 'No runner' }).join(), /not_installable: No runner/);
    assert.ok(reportProblems(undefined).length >= 3);
});

test('every entry is installed with the lock, the cache and the run root, a failed entry does not stop the others, and the copy is removed after', async (t) => {
    const out = scratch(t);
    const lockFile = path.join(LOCK_DIR, lockFiles[0]);
    const calls = [];
    const strictSeen = [];
    const installCheck = async (args) => {
        calls.push(args);
        const [id] = args;
        const run = args[args.indexOf('--run-root') + 1];
        fs.mkdirSync(path.join(run, id, 'b11295'), { recursive: true });
        fs.mkdirSync(path.join(args[args.indexOf('--cache') + 1], id), { recursive: true });
        return { report: id === 'llama.cpp' ? { ...GOOD_REPORT, ok: false, problems: ['x'] } : { ...GOOD_REPORT, runnerId: id }, code: id === 'llama.cpp' ? 1 : 0 };
    };
    const strict = (dir) => { strictSeen.push(dir); return { ok: true, checked: 1, driverOnly: [], unresolved: {}, errors: {}, problems: [] }; };
    const lines = [];
    const result = await checkLock({ lockFile, outDir: path.join(out, 'reports'), workDir: path.join(out, 'work'), expectVersion: 'b11295', installCheck, strict, log: (line) => lines.push(line) });
    assert.equal(result.ok, false);
    assert.deepEqual(calls.map((args) => args[0]).sort(), ['llama.cpp', 'llama.cpp-cpu'], 'both entries were installed');
    for (const args of calls) {
        assert.ok(args.includes('--require-executable') && args.includes('--run-root') && args.includes('--cache') && args[args.indexOf('--lock') + 1] === lockFile);
    }
    assert.equal(strictSeen.length, 2, 'the strict library check ran on both copies');
    assert.match(lines.join('\n'), /llama\.cpp: FAILED/);
    assert.match(lines.join('\n'), /llama\.cpp-cpu: passed/);
    assert.deepEqual(fs.readdirSync(path.join(out, 'reports')).sort(), ['llama.cpp-cpu.json', 'llama.cpp.json', 'summary.json']);
    assert.equal(fs.existsSync(path.join(out, 'work', 'run', 'llama.cpp')), false, 'the runnable copy is removed once checked');
    assert.equal(fs.existsSync(path.join(out, 'work', 'cache', 'llama.cpp-cpu')), false);
    const passing = async (args) => { await installCheck(args); return { report: { ...GOOD_REPORT, runnerId: args[0] }, code: 0 }; };
    const allGood = await checkLock({ lockFile, outDir: path.join(out, 'again'), workDir: path.join(out, 'work'), expectVersion: 'b11295', strict, installCheck: passing });
    assert.equal(allGood.ok, true, JSON.stringify(allGood.entries));
});

test('a strict-check failure, or a copy that was never built, fails the entry', async (t) => {
    const out = scratch(t);
    const lockFile = path.join(LOCK_DIR, lockFiles[0]);
    const build = async (args) => {
        fs.mkdirSync(path.join(args[args.indexOf('--run-root') + 1], args[0], 'b11295'), { recursive: true });
        return { report: { ...GOOD_REPORT, runnerId: args[0] }, code: 0 };
    };
    const bad = await checkLock({ lockFile, outDir: path.join(out, 'a'), workDir: path.join(out, 'w'), installCheck: build,
        strict: () => ({ ok: false, problems: ['1 ELF files need libraries that are not found: llama-server (libfoo.so.1)'] }) });
    assert.equal(bad.ok, false);
    assert.match(bad.entries[0].problems.join(), /libfoo\.so\.1/);
    const unbuilt = await checkLock({ lockFile, outDir: path.join(out, 'b'), workDir: path.join(out, 'w2'), installCheck: async () => ({ report: GOOD_REPORT, code: 0 }), strict: () => ({ ok: true, problems: [] }) });
    assert.equal(unbuilt.ok, false);
    assert.match(unbuilt.entries[0].problems.join(), /does not exist/);
    const threw = await checkLock({ lockFile, outDir: path.join(out, 'c'), workDir: path.join(out, 'w3'), installCheck: async () => { throw new Error('network'); } });
    assert.equal(threw.ok, false);
    assert.match(threw.entries[0].problems.join(), /network/);
});

function elfTree(t) {
    const dir = scratch(t);
    const elf = Buffer.concat([Buffer.from([0x7f, 0x45, 0x4c, 0x46]), Buffer.from('rest of the file')]);
    fs.writeFileSync(path.join(dir, 'llama-server'), elf, { mode: 0o755 });
    fs.writeFileSync(path.join(dir, 'libggml.so.0'), elf);
    fs.writeFileSync(path.join(dir, 'looks-like-text.txt'), elf);
    fs.writeFileSync(path.join(dir, 'libnot.so'), 'plain text, not an ELF file');
    fs.mkdirSync(path.join(dir, 'sub'));
    fs.writeFileSync(path.join(dir, 'sub', 'deep'), elf);
    fs.symlinkSync('llama-server', path.join(dir, 'link-to-server'));
    return dir;
}

test('ELF files are found by their magic, in subfolders, whatever their name, and links are not followed', (t) => {
    const dir = elfTree(t);
    const names = findElfFiles(dir).map((file) => path.relative(dir, file));
    assert.deepEqual(names, ['libggml.so.0', 'llama-server', 'looks-like-text.txt', path.join('sub', 'deep')]);
});

test('the strict check fails on any library that is not found except libcuda.so.1, and on no ELF file at all', (t) => {
    const dir = elfTree(t);
    const answers = (table) => (file) => ({ missing: table[path.basename(file)] ?? [] });
    const clean = strictLibraryCheck(dir, { ldd: answers({}) });
    assert.equal(clean.ok, true);
    assert.equal(clean.checked, 4);
    const driver = strictLibraryCheck(dir, { ldd: answers({ 'libggml.so.0': ['libcuda.so.1'] }) });
    assert.equal(driver.ok, true, 'the driver library alone is acceptable');
    assert.deepEqual(driver.driverOnly, ['libggml.so.0']);
    const other = strictLibraryCheck(dir, { ldd: answers({ 'libggml.so.0': ['libcuda.so.1', 'libcudart.so.12'] }) });
    assert.equal(other.ok, false);
    assert.deepEqual(other.unresolved, { 'libggml.so.0': ['libcudart.so.12'] });
    const broken = strictLibraryCheck(dir, { ldd: (file) => (path.basename(file) === 'deep' ? { error: 'ENOENT' } : { missing: [] }) });
    assert.equal(broken.ok, false, 'an ELF file ldd could not check is not a pass');
    assert.match(broken.problems.join(), /ldd could not check/);
    const emptyDir = scratch(t);
    fs.writeFileSync(path.join(emptyDir, 'README'), 'nothing');
    assert.equal(strictLibraryCheck(emptyDir, { ldd: answers({}) }).ok, false, 'a copy without any ELF file proves nothing');
});

test('negative control: a library whose name is in the copy but which the loader does not find passes the engine\'s check and fails this one', (t) => {
    // libcudart.so.12 sits in the copy, but the run path of the binary does not reach it: ldd says "not found".
    const missing = { 'libggml-cuda.so': ['libcudart.so.12'] };
    const engine = classifyMissingLibraries({ missing, provided: new Set(['libcudart.so.12']), optional: {} });
    assert.deepEqual(engine.unresolved, {}, 'the engine\'s check counts the name as provided');
    const dir = scratch(t);
    fs.writeFileSync(path.join(dir, 'libggml-cuda.so'), Buffer.from([0x7f, 0x45, 0x4c, 0x46, 0]));
    fs.writeFileSync(path.join(dir, 'libcudart.so.12'), Buffer.from([0x7f, 0x45, 0x4c, 0x46, 1]));
    const strict = strictLibraryCheck(dir, { ldd: (file) => ({ missing: path.basename(file) === 'libggml-cuda.so' ? missing['libggml-cuda.so'] : [] }) });
    assert.equal(strict.ok, false);
    assert.deepEqual(strict.unresolved, { 'libggml-cuda.so': ['libcudart.so.12'] });
});

test('with the real ldd, a copy of a system binary resolves its libraries (Linux only)', { skip: process.platform !== 'linux' }, (t) => {
    const dir = scratch(t);
    const probe = spawnSync('ldd', ['--version']);
    if (probe.error) return t.skip('no ldd on this machine');
    fs.copyFileSync(process.execPath, path.join(dir, 'node-copy'));
    fs.chmodSync(path.join(dir, 'node-copy'), 0o755);
    const result = strictLibraryCheck(dir);
    assert.equal(result.checked, 1);
    assert.equal(result.ok, true, result.problems.join());
    assert.ok(os.platform() === 'linux');
});

test('the platform picks the installer\'s checks: Linux the defaults, macOS the Mach-O check, and any other platform is refused', async () => {
    assert.equal(checksFor('linux').strict, strictLibraryCheck);
    assert.throws(() => checksFor('win32'), /linux and darwin/);
    assert.equal(checksFor('darwin').strict, strictMachoCheck);
    // A lock that cannot be read fails the install check itself (no download); it resolves with a report rather than throwing.
    const { report } = await checksFor('darwin').installCheck(['llama.cpp', '--lock', '/nonexistent/lock.json']);
    assert.equal(report.ok, false);
});

test('the strict check is told which file the entry launches, so @executable_path means the launched program\'s directory', async (t) => {
    const out = scratch(t);
    const seen = [];
    const installCheck = async (args) => {
        fs.mkdirSync(path.join(args[args.indexOf('--run-root') + 1], args[0], 'b11295'), { recursive: true });
        return { report: { ...GOOD_REPORT, runnerId: args[0] }, code: 0 };
    };
    const strict = (dir, options) => { seen.push(options); return { ok: true, problems: [] }; };
    const result = await checkLock({ lockFile: path.join(LOCK_DIR, darwinLockFiles[0]), outDir: path.join(out, 'o'), workDir: path.join(out, 'w'), installCheck, strict, platform: 'darwin' });
    assert.equal(result.ok, true);
    assert.deepEqual(seen, [{ executable: 'llama-server' }]);
});
