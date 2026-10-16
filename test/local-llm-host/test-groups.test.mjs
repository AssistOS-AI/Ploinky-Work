// Placeholder until the standalone host tests land: keeps the test/local-llm-host/ group non-empty, so the glob in the test script always matches.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const manifest = JSON.parse(fs.readFileSync(new URL('../../package.json', import.meta.url), 'utf8'));

test('the test script names the pworker, engine and host test groups explicitly, so no group is skipped silently', () => {
    for (const group of ['test/*.test.mjs', 'test/local-llm/*.test.mjs', 'test/local-llm-host/*.test.mjs']) {
        assert.ok(manifest.scripts.test.split(/\s+/).includes(group), group);
    }
});
