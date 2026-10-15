// The injection specification (DS007) documents every host field, platform member, message key and limits hook, and the
// matrix lists every specification. A field added without a row makes this fail.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import { defineHost, linuxPlatform } from '../../lib/local-llm/host.mjs';
import { LIMIT_HOOKS } from '../../lib/local-llm/limits.mjs';
import { MESSAGE_KEYS } from '../../lib/local-llm/messages.mjs';
import * as engine from '../../lib/local-llm/index.mjs';

const SPECS = new URL('../../docs/specs/', import.meta.url);
const read = (name) => fs.readFileSync(new URL(name, SPECS), 'utf8');
const DS007 = read('DS007-host-limits-and-messages.md');

// The first cell of each row of the first table after `heading`.
function tableAfter(text, heading) {
    const start = text.indexOf(heading);
    assert.ok(start >= 0, heading);
    const lines = text.slice(start).split('\n');
    const rows = [];
    let inTable = false;
    for (const line of lines.slice(1)) {
        if (line.startsWith('|')) { inTable = true; rows.push(line); } else if (inTable) break;
    }
    return rows.slice(2).map((line) => line.split('|')[1].trim());
}

// A cell may name several fields (`uv`, `python`); each name is the word after a backtick, up to a parenthesis or space.
const namesOf = (cells) => cells.flatMap((cell) => [...cell.matchAll(/`([A-Za-z.'_-]+)/g)].map((match) => match[1]));

test('DS007 has a row for every host field, platform member, message key and limits hook, and no row for anything else', () => {
    const host = defineHost({ dataDir: '/x/data', runRoot: '/x/run' });
    const hostRows = namesOf(tableAfter(DS007, '### The host (`defineHost`)'));
    assert.deepEqual([...hostRows].sort(), Object.keys(host).sort());
    const platformRows = namesOf(tableAfter(DS007, '### The platform (`linuxPlatform`)'));
    assert.deepEqual([...platformRows].sort(), Object.keys(linuxPlatform()).sort());
    const messageRows = namesOf(tableAfter(DS007, '### The messages (`defineMessages`)'));
    assert.deepEqual([...messageRows].sort(), [...MESSAGE_KEYS].sort());
    const hookRows = namesOf(tableAfter(DS007, 'A hook that throws propagates'));
    assert.deepEqual([...hookRows].sort(), [...LIMIT_HOOKS].sort());
});

test('DS007 lists the public exports of the entry and of each stable subpath', async () => {
    const rows = tableAfter(DS007, '### Public exports');
    assert.deepEqual(rows.map((cell) => cell.replaceAll('`', '')), [
        'pworker/local-llm', 'pworker/local-llm/chat', 'pworker/local-llm/runner-process', 'pworker/local-llm/runner-install-check', 'pworker/local-llm/internal/*',
    ]);
    const entryRow = DS007.split('\n').find((line) => line.startsWith('| `pworker/local-llm` |'));
    for (const name of Object.keys(engine)) {
        assert.ok(entryRow.includes(`\`${name}\``), `${name} is not in the export row of DS007`);
    }
    for (const [subpath, module] of [['chat', await import('pworker/local-llm/chat')], ['runner-process', await import('pworker/local-llm/runner-process')]]) {
        const row = DS007.split('\n').find((line) => line.startsWith(`| \`pworker/local-llm/${subpath}\` |`));
        for (const name of Object.keys(module)) assert.ok(row.includes(`\`${name}`), `${name} is not in the ${subpath} row of DS007`);
    }
});

test('the matrix lists every specification file, and every relative link in the specifications resolves', () => {
    const files = fs.readdirSync(SPECS).filter((name) => name.endsWith('.md') && name !== 'matrix.md').sort();
    const matrix = read('matrix.md');
    for (const file of files) assert.ok(matrix.includes(`(${file})`), `${file} is not in the matrix`);
    for (const file of [...files, 'matrix.md']) {
        for (const [, target] of read(file).matchAll(/\]\(([^)#:]+\.md)\)/g)) {
            assert.ok(fs.existsSync(new URL(target, SPECS)), `${file} links to missing ${target}`);
        }
    }
    // Every DSnnn named in a specification is carried here, or named as the agent contract that stays in local-llms.
    const carried = new Set(files.map((name) => name.slice(0, 5)));
    for (const file of files) {
        for (const [, id] of read(file).matchAll(/\b(DS\d{3})\b/g)) {
            // DS001 stays in local-llms; the other numbers above DS007 name the retired repository's own specifications.
            if (id === 'DS001' || Number(id.slice(2)) > 7) continue;
            assert.ok(carried.has(id), `${file} names ${id}, which is not carried`);
        }
    }
});
