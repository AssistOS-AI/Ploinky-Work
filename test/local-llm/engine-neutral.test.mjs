// V8, engine neutrality: no code of lib/local-llm names a deployment or a container limit. The scan strips comments, then
// counts one hit per source line that matches any term, case-insensitively, in every .mjs file below lib/local-llm
// (tools/ included). The physical share of the GPU pool the unified vLLM policy sizes (`estimate.budgetBytes`) is not a
// container limit and does not match: the term is `gpu.budgetBytes`. The scan fails when it finds no file or cannot read one,
// and a negative control shows that it sees the container CPU quota the engine used to read.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const ENGINE = fileURLToPath(new URL('../../lib/local-llm', import.meta.url));

// The terms of plan V8, as regular expression sources.
const TERMS = Object.freeze([
    'ploinky', 'soul_gateway', 'Settings > Agents', '/opt/local-llm', '/usr/local/nvidia', '/run/ploinky',
    '\\bmps\\b', 'CUDA_MPS', 'MPS_VARIABLES', 'qualifyMps', 'cgroup', 'cgroupMemory', 'cpu\\.max', 'memory\\.max',
    'container memory', 'gpu\\.budgetBytes', 'finiteMemoryBytes', 'headroomBytes', 'memoryBudget', 'gpuBudget', 'cpuQuota',
    'budgetGuard',
]);
const MATCHER = new RegExp(TERMS.join('|'), 'i');

/**
 * `source` with its comments blanked and its strings, template literals and regular expression literals kept, so a term
 * inside a string is a hit and a term inside a comment is not. Newlines survive, so line numbers stay.
 */
export function stripComments(source) {
    let out = '';
    let index = 0;
    let last = '';
    const blank = (text) => text.replace(/[^\n]/g, ' ');
    while (index < source.length) {
        const char = source[index];
        const next = source[index + 1];
        if (char === '/' && next === '/') {
            const end = source.indexOf('\n', index);
            const stop = end === -1 ? source.length : end;
            out += blank(source.slice(index, stop));
            index = stop;
        } else if (char === '/' && next === '*') {
            const end = source.indexOf('*/', index + 2);
            const stop = end === -1 ? source.length : end + 2;
            out += blank(source.slice(index, stop));
            index = stop;
        } else if (char === '\'' || char === '"' || char === '`') {
            let stop = index + 1;
            while (stop < source.length && source[stop] !== char) stop += source[stop] === '\\' ? 2 : 1;
            out += source.slice(index, stop + 1);
            index = stop + 1;
            last = char;
        } else if (char === '/' && (last === '' || '(,=:[!&|?{};'.includes(last))) {
            // A regular expression literal: skip to its closing slash, outside a character class.
            let stop = index + 1;
            let inClass = false;
            while (stop < source.length && (inClass || source[stop] !== '/') && source[stop] !== '\n') {
                if (source[stop] === '\\') stop += 1;
                else if (source[stop] === '[') inClass = true;
                else if (source[stop] === ']') inClass = false;
                stop += 1;
            }
            out += source.slice(index, stop + 1);
            index = stop + 1;
            last = '/';
        } else {
            out += char;
            if (!/\s/.test(char)) last = char;
            index += 1;
        }
    }
    return out;
}

/** The lines of `source` that match a term, comments stripped: [{ line, text }]. */
export function hitsIn(source) {
    return stripComments(source).split('\n')
        .map((text, at) => ({ line: at + 1, text: text.trim() }))
        .filter(({ text }) => MATCHER.test(text));
}

function sourceFiles(root) {
    const files = [];
    const walk = (directory) => {
        for (const name of fs.readdirSync(directory)) {
            const full = path.join(directory, name);
            // statSync follows links, so a dangling link is an error and a linked file is scanned.
            const stat = fs.statSync(full);
            if (stat.isDirectory()) walk(full);
            else if (stat.isFile() && name.endsWith('.mjs')) files.push(full);
        }
    };
    walk(root);
    return files.sort();
}

/** Every hit below `root`. Throws when no source file is found or one cannot be read. */
export function scan(root) {
    const files = sourceFiles(root);
    if (files.length === 0) throw new Error(`The scan found no .mjs file below ${root}`);
    const hits = [];
    for (const file of files) {
        for (const hit of hitsIn(fs.readFileSync(file, 'utf8'))) hits.push({ file: path.relative(root, file), ...hit });
    }
    return { files: files.map((file) => path.relative(root, file)), hits };
}

test('engine neutrality: no file below lib/local-llm, tools included, names a deployment or a container limit', () => {
    const { files, hits } = scan(ENGINE);
    for (const required of ['controller/deployments.mjs', 'controller/hardware.mjs', 'controller/admission.mjs', 'controller/profiles.mjs',
        'runners/vllm.mjs', 'tools/runnerInstallCheck.mjs', 'limits.mjs', 'host.mjs']) {
        assert.ok(files.includes(required), `${required} is scanned`);
    }
    assert.deepEqual(hits.map((hit) => `${hit.file}:${hit.line}: ${hit.text.slice(0, 120)}`), [], 'terms found in the engine');
});

test('engine neutrality: the scanner fails when it finds no file or cannot read one', (t) => {
    const empty = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'neutral-empty-'));
    t.after(() => fs.rmSync(empty, { recursive: true, force: true }));
    assert.throws(() => scan(empty), /found no \.mjs file/);
    assert.throws(() => scan(path.join(empty, 'missing')), { code: 'ENOENT' });
    // A file that cannot be read, and a link that leads nowhere, fail the scan.
    fs.writeFileSync(path.join(empty, 'ok.mjs'), 'export const a = 1;\n');
    assert.deepEqual(scan(empty).files, ['ok.mjs']);
    const linked = path.join(empty, 'link.mjs');
    fs.symlinkSync(path.join(empty, 'nowhere.mjs'), linked);
    assert.throws(() => scan(empty), { code: 'ENOENT' });
    fs.rmSync(linked);
    if (process.getuid?.() !== 0) {
        const locked = path.join(empty, 'locked.mjs');
        fs.writeFileSync(locked, 'export const b = 2;\n');
        fs.chmodSync(locked, 0o000);
        assert.throws(() => scan(empty), { code: 'EACCES' });
    }
});

// The four lines of the base engine that read the container CPU quota (local-llms 34f8f498, hardware.mjs:425-428).
const BASE_CPU_QUOTA = [
    "    const quota = readText(fsApi, '/sys/fs/cgroup/cpu.max');",
    "    const quotaMatch = quota && /^(\\d+)\\s+(\\d+)\\s*$/.exec(quota.trim());",
    "    if (quotaMatch && Number(quotaMatch[2]) > 0) {",
    "        cores = Math.min(cores, Math.max(1, Math.ceil(Number(quotaMatch[1]) / Number(quotaMatch[2]))));",
].join('\n');

test('engine neutrality: negative control, the base container CPU quota read gives exactly one hit', (t) => {
    assert.equal(hitsIn(BASE_CPU_QUOTA).length, 1);
    assert.equal(hitsIn(BASE_CPU_QUOTA)[0].line, 1);
    const scratch = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'neutral-control-'));
    t.after(() => fs.rmSync(scratch, { recursive: true, force: true }));
    fs.writeFileSync(path.join(scratch, 'hardware.mjs'), `export function cores(fsApi, cores, readText) {\n${BASE_CPU_QUOTA}\n    return cores;\n}\n`);
    const { hits } = scan(scratch);
    assert.equal(hits.length, 1);
    assert.deepEqual([hits[0].file, hits[0].line], ['hardware.mjs', 2]);
    // The engine's own copy of the same function, with the hook, is clean.
    const real = fs.readFileSync(path.join(ENGINE, 'controller/hardware.mjs'), 'utf8');
    assert.match(real, /limits\.capCores\(cores, \{ fsApi \}\)/);
    assert.equal(hitsIn(real).length, 0);
});

test('engine neutrality: the stripper keeps strings and regular expressions and drops comments', () => {
    const cases = [
        ["const a = 1; // cgroup", 0],
        ["/* container memory\n   cgroup */ const a = 1;", 0],
        ["const a = 'cgroup';", 1],
        ["const a = \"x // y\"; const b = 'memory.max';", 1],
        ["const a = `${x} // cpu.max`;", 1],
        ["const a = /['\"]/; const b = 'ploinky';", 1],
        ["const a = /a\\/\\/b/; // gpuBudget", 0],
        ["const a = 'http://x'; const mps = 1;", 1],
        ["const a = estimate.budgetBytes; const b = { budgetBytes };", 0],
        ["const a = gpu.budgetBytes;", 1],
        ["const amps = 1; const mps_total = 2;", 0],
        ["line one\n/* a\n b */ cgroup\nline four", 1],
    ];
    for (const [source, count] of cases) assert.equal(hitsIn(source).length, count, source);
    // Newlines survive stripping, so a hit reports its real line.
    assert.equal(hitsIn('a\n/* x\ny */\nconst c = "cgroup";')[0].line, 4);
});

test('engine neutrality: the physical share of the GPU pool in the unified vLLM policy does not match', () => {
    const profiles = fs.readFileSync(path.join(ENGINE, 'controller/profiles.mjs'), 'utf8');
    assert.match(profiles, /const budgetBytes = Math\.round\(/, 'the physical share is still computed');
    assert.match(profiles, /gpuMemoryUtilization: share, budgetBytes,/);
    assert.equal(hitsIn(profiles).length, 0);
});
