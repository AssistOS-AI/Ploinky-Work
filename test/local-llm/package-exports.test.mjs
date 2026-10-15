// The package surface of the engine: four stable subpaths, one unstable internal pattern, nothing excluded from the
// package, and no dependency (the engine runs on Node alone).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const ROOT = new URL('../../', import.meta.url);
const pkg = JSON.parse(fs.readFileSync(new URL('package.json', ROOT), 'utf8'));

const STABLE = {
    './local-llm': './lib/local-llm/index.mjs',
    './local-llm/chat': './lib/local-llm/chat.mjs',
    './local-llm/runner-process': './lib/local-llm/controller/runnerProcess.mjs',
    './local-llm/runner-install-check': './lib/local-llm/tools/runnerInstallCheck.mjs',
};

test('package: the stable engine subpaths and the internal pattern are exported as documented', () => {
    for (const [subpath, target] of Object.entries(STABLE)) {
        assert.equal(pkg.exports[subpath], target, subpath);
        assert.ok(fs.existsSync(new URL(target, ROOT)), target);
    }
    assert.equal(pkg.exports['./local-llm/internal/*'], './lib/local-llm/*');
    // Nothing else of the engine is exported: the rest of the tree is reachable only through the internal pattern.
    assert.deepEqual(Object.keys(pkg.exports).filter((key) => key.startsWith('./local-llm')).sort(), [...Object.keys(STABLE), './local-llm/internal/*'].sort());
});

test('package: the stable subpaths resolve through the package name, the internal pattern reaches every module and asset, and other paths are closed', async () => {
    const entry = await import('pworker/local-llm');
    assert.equal(typeof entry.createLocalLlmController, 'function');
    assert.equal(typeof (await import('pworker/local-llm/chat')).buildRunnerRequest, 'function');
    assert.equal(typeof (await import('pworker/local-llm/runner-process')).startRunnerProcess, 'function');
    assert.equal(typeof (await import('pworker/local-llm/runner-install-check')).runInstallCheck, 'function');
    assert.equal(typeof (await import('pworker/local-llm/internal/controller/hardware.mjs')).readSnapshot, 'function');
    const schema = new URL(import.meta.resolve('pworker/local-llm/internal/catalog/schema.json'));
    assert.ok(fs.existsSync(schema));
    assert.ok(fs.existsSync(new URL(import.meta.resolve('pworker/local-llm/internal/runners/lmStudioLoad.mjs'))));
    await assert.rejects(import('pworker/local-llm/controller/hardware.mjs'), { code: 'ERR_PACKAGE_PATH_NOT_EXPORTED' });
});

test('package: the runtime needs Node 22.13 and no dependency, and the package excludes none of the engine tree', () => {
    assert.equal(pkg.engines.node, '>=22.13');
    for (const field of ['dependencies', 'optionalDependencies', 'peerDependencies']) assert.deepEqual(pkg[field] ?? {}, {}, field);
    // No `files` list and no .npmignore: the package holds what git holds, and git ignores nothing below lib/local-llm.
    assert.equal('files' in pkg, false);
    assert.equal(fs.existsSync(new URL('.npmignore', ROOT)), false);
    const ignore = fs.readFileSync(new URL('.gitignore', ROOT), 'utf8').split('\n').map((line) => line.trim()).filter((line) => line && !line.startsWith('#') && !line.startsWith('!'));
    const walk = (dir) => fs.readdirSync(dir, { withFileTypes: true }).flatMap((item) => (item.isDirectory() ? walk(new URL(`${item.name}/`, dir)) : [new URL(item.name, dir)]));
    for (const file of walk(new URL('lib/local-llm/', ROOT))) {
        const relative = file.pathname.slice(ROOT.pathname.length);
        for (const pattern of ignore) {
            const bare = pattern.replace(/^\//, '').replace(/\/$/, '');
            const parts = relative.split('/');
            const hit = !bare.includes('*') && !bare.includes('/') ? parts.includes(bare) : false;
            assert.equal(hit, false, `${relative} matches .gitignore pattern ${pattern}`);
        }
    }
});

test('package: the test script names every test group', () => {
    const script = pkg.scripts.test;
    for (const glob of ['test/*.test.mjs', 'test/local-llm/*.test.mjs', 'test/local-llm-host/*.test.mjs']) assert.ok(script.includes(glob), glob);
});
