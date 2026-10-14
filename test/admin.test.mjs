import './home.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pworkerHome } from '../lib/settings.mjs';
import { listProviders, listTiers, readUserConfig, removeTier, saveCustomProvider, setProviderKey, setTierChain } from '../lib/pworker/admin.mjs';

const catalog = (ids) => async (url, init) => {
  assert.match(String(init.headers.authorization ?? ''), /^Bearer /);
  return new Response(JSON.stringify({ data: ids.map((id) => ({ id })) }), { status: 200, headers: { 'content-type': 'application/json' } });
};

test('setProviderKey validates the key, writes only the key file and listProviders never returns the key', async () => {
  const result = await setProviderKey('deepseek', 'sk-test-123', { fetchImpl: catalog(['deepseek-flash']) });
  assert.deepEqual(result, { provider: 'deepseek', keyConfigured: true, models: 1 });
  const keyFile = path.join(pworkerHome(), 'keys', 'deepseek.env');
  assert.match(fs.readFileSync(keyFile, 'utf8'), /DEEPSEEK_API_KEY="sk-test-123"/);
  assert.equal(fs.statSync(keyFile).mode & 0o777, 0o600);
  const row = listProviders().find((p) => p.name === 'deepseek');
  assert.equal(row.keyConfigured, true);
  assert.ok(!JSON.stringify(listProviders()).includes('sk-test-123'));
  assert.ok(!JSON.stringify(readUserConfig()).includes('sk-test-123'));
});

test('setProviderKey keeps other variables of the key file and an empty key removes the key', async () => {
  const keyFile = path.join(pworkerHome(), 'keys', 'openai.env');
  fs.mkdirSync(path.dirname(keyFile), { recursive: true });
  fs.writeFileSync(keyFile, 'OPENAI_BASE_URL=https://example.test\nOPENAI_API_KEY=old\n');
  await setProviderKey('openai', 'new-key', { validate: false });
  assert.equal(fs.readFileSync(keyFile, 'utf8'), 'OPENAI_BASE_URL=https://example.test\nOPENAI_API_KEY="new-key"\n');
  await setProviderKey('openai', '', { validate: false });
  assert.equal(fs.readFileSync(keyFile, 'utf8'), 'OPENAI_BASE_URL=https://example.test\n');
  assert.equal(listProviders().find((p) => p.name === 'openai').keyConfigured, false);
});

test('a refused key is not saved', async () => {
  const refuse = async () => new Response('{}', { status: 401 });
  await assert.rejects(setProviderKey('grok', 'bad', { fetchImpl: refuse }), /401/);
  assert.equal(listProviders().find((p) => p.name === 'grok').keyConfigured, false);
});

test('local and unknown providers take no key', async () => {
  await assert.rejects(setProviderKey('local', 'x', { validate: false }), /needs no key/);
  await assert.rejects(setProviderKey('nope', 'x', { validate: false }), /Unknown provider/);
});

test('saveCustomProvider stores an OpenAI-compatible endpoint, marked custom', async () => {
  await saveCustomProvider('myapi', { endpoint: 'https://api.example.test/v1', key: 'k-1', rpm: 30, fetchImpl: catalog(['m1']) });
  const row = listProviders().find((p) => p.name === 'myapi');
  assert.equal(row.custom, true);
  assert.equal(row.keyConfigured, true);
  assert.equal(readUserConfig().providers.myapi.limits.maxPerMinute, 30);
});

test('setTierChain writes chains, extraBody, aliases and batching; removeTier restores the default', async () => {
  await setTierChain('tiny', [{ provider: 'zai', model: 'glm-flash', extraBody: { thinking: { type: 'disabled' } } }, { upstream: 'deepseek', model: 'deepseek-flash' }], { batch: true });
  const { tiers, batching } = listTiers();
  assert.deepEqual(tiers.tiny, [{ upstream: 'zai', model: 'glm-flash', extraBody: { thinking: { type: 'disabled' } } }, { upstream: 'deepseek', model: 'deepseek-flash' }]);
  assert.equal(batching.tiny, true);
  await setTierChain('fast', 'tiny');
  assert.equal(listTiers().tiers.fast, 'tiny');
  await setTierChain('glm', [{provider: 'zai', model: 'glm-flash', extraBody: {reasoning_effort: 'low'}, minTokens: 512}]);
  assert.deepEqual(listTiers().tiers.glm, [{upstream: 'zai', model: 'glm-flash', extraBody: {reasoning_effort: 'low'}, minTokens: 512}]);
  await assert.rejects(setTierChain('glm', [{provider: 'zai', model: 'x', minTokens: 0}]), /minTokens/);
  await assert.rejects(setTierChain('bad tier', []), /Invalid tier name/);
  await assert.rejects(setTierChain('small', [{ provider: 'zai' }]), /model id/);
  await assert.rejects(setTierChain('small', [{ provider: 'zai', model: 'x', extraBody: [] }]), /JSON object/);
  assert.equal(removeTier('tiny'), true);
  assert.deepEqual(listTiers().tiers.tiny, [{ upstream: 'local', model: 'Qwen3.6-35B-A3B' }]);
});
