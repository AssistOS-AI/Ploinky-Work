#!/usr/bin/env node
// The install check of one host runner lock (lib/local-llm-host/locks/runners.lock.host-linux-<arch>.json), as the runner-lock-check
// workflow runs it: EVERY entry of the lock is installed with pworker's own installer (download, sha256 check, unpack, then the checks
// of lib/local-llm/tools/runnerInstallCheck.mjs with --require-executable: the launched file exists and runs `--version` naming the
// pinned build), and the unpacked copy goes through the strict library check of elf.mjs (every ELF file, `libcuda.so.1` being the
// only library that may be missing). A report that is not a real install fails: not `ok`, `downloaded: false` (a validate-only or
// proprietary skip), or a version other than the one expected. A lock with no entry, or no file, fails too.
//
// Usage: node lockcheck.mjs --lock FILE --out DIR [--work DIR] [--expect-version b11295] [--keep]
//   writes DIR/<runner>.json (the install report and the strict check) and DIR/summary.json, prints one line per entry, and exits 1
//   when anything failed. The cache and the runnable copy of an entry go under --work (default: a new folder in the temporary
//   directory) and are removed once the entry is checked, unless --keep, so the disk holds one entry at a time.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { argv, exit } from 'node:process';
import { fileURLToPath } from 'node:url';

import { RUNNER_LOCK_SCHEMA } from '../local-llm/controller/runnerLock.mjs';
import { runInstallCheck } from '../local-llm/tools/runnerInstallCheck.mjs';
import { strictLibraryCheck } from './elf.mjs';

const RUNNER_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

/** The reasons a report of the install check does not prove an install (an empty list when it does). */
export function reportProblems(report, { expectVersion = null } = {}) {
    const problems = [];
    if (report?.ok !== true) problems.push(`the install check did not pass${report?.error ? ` (${report.error}: ${report.message})` : ''}${Array.isArray(report?.problems) && report.problems.length ? `: ${report.problems.join('; ')}` : ''}`);
    if (report?.downloaded === false) problems.push(`nothing was downloaded: ${report.skipped ?? 'the entry was only validated'}`);
    if (!report?.executable?.found) problems.push('the launched file was not found in the runnable copy');
    if (report?.executable?.version?.status !== 0) problems.push('the launched file did not run `--version` successfully');
    if (expectVersion !== null && report?.version !== expectVersion) problems.push(`the entry's version is ${JSON.stringify(report?.version)}, not ${expectVersion}`);
    return problems;
}

/**
 * Checks every runner of the lock at `lockFile`. `installCheck(args)` and `strict(dir)` are seams for tests; the defaults are the real
 * install check and strict library check. Resolves `{ ok, entries, problems }` and writes the reports to `outDir`.
 */
export async function checkLock({ lockFile, outDir, workDir, expectVersion = null, keep = false, log = () => {}, installCheck = runInstallCheck, strict = strictLibraryCheck }) {
    const problems = [];
    const entries = [];
    fs.mkdirSync(outDir, { recursive: true });
    let lock = null;
    try {
        lock = JSON.parse(fs.readFileSync(lockFile, 'utf8'));
    } catch (error) {
        problems.push(`the lock ${lockFile} cannot be read (${error.code || error.message})`);
    }
    if (lock && lock.schema !== RUNNER_LOCK_SCHEMA) problems.push(`the lock's schema is ${JSON.stringify(lock.schema)}, not ${RUNNER_LOCK_SCHEMA}`);
    const ids = lock && lock.runners && typeof lock.runners === 'object' ? Object.keys(lock.runners) : [];
    if (lock && ids.length === 0) problems.push('the lock has no entries, so there is nothing to prove');
    const cache = path.join(workDir, 'cache');
    const run = path.join(workDir, 'run');
    for (const id of ids) {
        if (!RUNNER_ID_RE.test(id)) { problems.push(`the runner id ${JSON.stringify(id)} is not a plain name`); continue; }
        const entry = { id, ok: false, problems: [] };
        entries.push(entry);
        const version = lock.runners[id]?.version;
        try {
            const args = [id, '--lock', lockFile, '--require-executable', '--cache', cache, '--run-root', run];
            const { report } = await installCheck(args);
            entry.problems.push(...reportProblems(report, { expectVersion }));
            const runDir = path.join(run, id, String(version));
            let strictReport = null;
            if (fs.existsSync(runDir)) {
                strictReport = strict(runDir);
                entry.problems.push(...strictReport.problems);
            } else {
                entry.problems.push(`the runnable copy ${runDir} does not exist`);
            }
            fs.writeFileSync(path.join(outDir, `${id}.json`), `${JSON.stringify({ runnerId: id, install: report, strict: strictReport }, null, 2)}\n`);
        } catch (error) {
            entry.problems.push(`the check threw: ${error?.message ?? error}`);
        } finally {
            if (!keep) {
                fs.rmSync(path.join(cache, id), { recursive: true, force: true });
                fs.rmSync(path.join(run, id), { recursive: true, force: true });
            }
        }
        entry.ok = entry.problems.length === 0;
        log(`${id}: ${entry.ok ? 'passed' : `FAILED (${entry.problems.join(' | ')})`}`);
    }
    const ok = problems.length === 0 && entries.length > 0 && entries.every((entry) => entry.ok);
    const summary = { lock: path.basename(lockFile), ok, entries: entries.map(({ id, ok: passed, problems: why }) => ({ id, ok: passed, problems: why })), problems };
    fs.writeFileSync(path.join(outDir, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`);
    for (const problem of problems) log(`FAILED: ${problem}`);
    return { ok, entries, problems };
}

function optionOf(args, name) {
    const index = args.indexOf(name);
    return index >= 0 && args[index + 1] ? args[index + 1] : null;
}

if (argv[1] && path.resolve(argv[1]) === fileURLToPath(import.meta.url)) {
    const args = argv.slice(2);
    const lockFile = optionOf(args, '--lock');
    const outDir = optionOf(args, '--out');
    if (!lockFile || !outDir) {
        process.stderr.write('Usage: node lockcheck.mjs --lock FILE --out DIR [--work DIR] [--expect-version VERSION] [--keep]\n');
        exit(2);
    }
    const workDir = optionOf(args, '--work') ?? fs.mkdtempSync(path.join(os.tmpdir(), 'lockcheck-'));
    const result = await checkLock({
        lockFile: path.resolve(lockFile), outDir: path.resolve(outDir), workDir: path.resolve(workDir), expectVersion: optionOf(args, '--expect-version'),
        keep: args.includes('--keep'), log: (line) => process.stdout.write(`${line}\n`),
    });
    exit(result.ok ? 0 : 1);
}
