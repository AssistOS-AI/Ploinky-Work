// F2: a server builds its local-llm host on its own home, so a fixture that gives the server `env: {}` never points the host at the
// real ~/.pworker (the host creates directories and binds a port under its home when it is used).
import '../home.mjs';
import assert from 'node:assert/strict';
import path from 'node:path';
import test from 'node:test';

import { createPworkerServer } from '../../lib/server.mjs';
import { scratch } from './fixtures/engine.mjs';
import { ownedHome } from './fixtures/proxy.mjs';

test('a server builds its host on its own home: config.taskHome, else the parent of its data folder, never the ~/.pworker of its env', async (t) => {
    const { home, later } = ownedHome(t);
    const byTaskHome = await createPworkerServer({ config: { taskHome: home, dataDir: path.join(home, 'elsewhere', 'data'), providers: {}, tiers: {} }, env: {} });
    later(() => byTaskHome.close());
    assert.equal(byTaskHome.localLlm.layout.home, home);
    assert.equal(byTaskHome.localLlm.layout.root, path.join(home, 'local-llm'));
    const { home: other, later: laterOther } = ownedHome(t);
    const byData = await createPworkerServer({ config: { dataDir: path.join(other, 'data'), providers: {}, tiers: {} }, env: { PWORKER_HOME: scratch(t) } });
    laterOther(() => byData.close());
    assert.equal(byData.localLlm.layout.home, other);
});

