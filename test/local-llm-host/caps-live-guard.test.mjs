// test/live/caps-live.mjs downloads a runner and a model and starts processes: it must refuse unless PWORKER_LIVE_CAPS=1, and `npm test`
// must not run it. The helpers it derives its cap values with (test/live/caps-derive.mjs) are pure, and are pinned here on fixed inputs,
// because a wrong band would place a cap where the live case does not mean it to.
import '../home.mjs';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { scratch } from './fixtures/engine.mjs';
import {
    CAP_LIMIT_SLACK_BYTES, GIB, MIB, availableNowBytes, gpuCapAdmitting, gpuVerdict, halvedContext, insufficientNowCause, parseRunnerArgv,
    deploymentIdentity, deploymentUnchanged, explicitCtxExpectation, explicitCtxLaunchMatches, explicitCtxPreviewMatches, fitLoweredSteps,
    planGpuFitContext, planGpuFitLayers, planOurUseCap, planRamFit, ramAdmitMargin, ramNeedOf, settleFailedCase, within,
} from '../live/caps-derive.mjs';

const ROOT = fileURLToPath(new URL('../..', import.meta.url));
const LIVE = path.join(ROOT, 'test', 'live', 'caps-live.mjs');

function live(args, env) {
    const { NODE_TEST_CONTEXT: _context, PWORKER_LIVE_CAPS: _live, ...parent } = process.env;
    return spawnSync(process.execPath, [LIVE, ...args], { encoding: 'utf8', env: { ...parent, ...env }, timeout: 20_000 });
}

test('without PWORKER_LIVE_CAPS=1 the caps suite refuses, with exit 2, and creates nothing', (t) => {
    const out = path.join(scratch(t), 'out');
    const result = live(['--out', out], {});
    assert.equal(result.status, 2);
    assert.match(result.stderr, /Set PWORKER_LIVE_CAPS=1 to run it/);
    assert.equal(fs.existsSync(out), false, 'no output folder, no home, no process');
    assert.equal(live(['--out', out], { PWORKER_LIVE_CAPS: '0' }).status, 2, 'only the value 1 enables it');
    assert.equal(live(['--out', out], { PWORKER_LIVE_ENGINE: '1' }).status, 2, 'the engine smoke variable does not enable it');
});

test('a bad option, case or port is refused before anything starts, even with the variable set', (t) => {
    const out = path.join(scratch(t), 'out');
    // Off Linux and Apple Silicon the platform refusal comes first; the message is asserted either way.
    const supported = process.platform === 'linux' || (process.platform === 'darwin' && process.arch === 'arm64');
    const platform = /Refusing to run on (\w+|darwin-\w+): the caps suite/;
    const cases = [
        [['--out', out, '--only', 'L99'], /Unknown case L99; use L1, L2/],
        [['--out', out, '--home', path.join(path.dirname(out), 'missing-home')], /--home .*missing-home is not an existing folder/],
        [['--out', out, '--port', '18080'], /--port 18080 is not a usable port/],
        [['--out', out, '--bogus'], /Unknown or incomplete option --bogus/],
        [[], /^Usage: PWORKER_LIVE_CAPS=1 node test\/live\/caps-live\.mjs --out DIR/],
    ];
    for (const [args, message] of cases) {
        const result = live(args, { PWORKER_LIVE_CAPS: '1' });
        assert.equal(result.status, 2, args.join(' '));
        assert.match(result.stderr, supported ? message : platform, args.join(' '));
    }
    assert.equal(fs.existsSync(out), false);
});

test('a home that is not empty is refused, so a run never mixes with older state', (t) => {
    if (!(process.platform === 'linux' || (process.platform === 'darwin' && process.arch === 'arm64'))) return t.skip('platform refusal comes first');
    const dir = scratch(t);
    fs.writeFileSync(path.join(dir, 'older'), 'x');
    const result = live(['--out', path.join(dir, 'out'), '--home', dir], { PWORKER_LIVE_CAPS: '1' });
    assert.equal(result.status, 2);
    assert.match(result.stderr, /is not empty; use a fresh folder/);
    assert.equal(fs.existsSync(path.join(dir, 'out')), false);
});

test('npm test does not run the caps suite, and npm run test:live:caps does', () => {
    const scripts = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).scripts;
    assert.ok(!/live/.test(scripts.test), scripts.test);
    assert.equal(scripts['test:live:caps'], 'node test/live/caps-live.mjs');
    assert.ok(fs.existsSync(LIVE));
});

test('derived values: sizes, the halved context, the argv reader and the parsers', () => {
    assert.equal(ramNeedOf({ needBytes: 5, ramBytes: 9 }), 5, 'Metal names needBytes');
    assert.equal(ramNeedOf({ ramBytes: 9 }), 9);
    assert.equal(halvedContext(4096), 2048);
    assert.equal(halvedContext(600), 512, 'never below the schema minimum');
    assert.equal(ramAdmitMargin('dedicated'), 1024 ** 3);
    assert.equal(ramAdmitMargin('metal'), 0);
    assert.equal(availableNowBytes('Needs about 1.78 GiB of memory and 1.60 GiB kept free; 3.35 GiB is available now. Other processes hold the rest.'), Math.round(3.35 * 1024 ** 3));
    assert.equal(availableNowBytes('Needs about 2.0 GiB of RAM; 812 MiB is available now.'), 812 * MIB);
    assert.equal(availableNowBytes('Needs about 1.78 GiB.'), null);
    const metal = '/x/llama-server --host /s/0123abcd.sock --api-key *** -m /m.gguf -c 2048 -ngl 999 -fa auto --threads 2 -np 1 --cache-ram 256';
    assert.deepEqual(parseRunnerArgv(metal), { ctx: 2048, threads: 2, parallel: 1, layers: 999 });
    assert.deepEqual(parseRunnerArgv(['llama-server', '--ctx-size', '8192', '--n-gpu-layers', '17', '--threads', '6', '-np', '2']), { ctx: 8192, threads: 6, parallel: 2, layers: 17 });
    assert.deepEqual(parseRunnerArgv('llama-server --flash-attn on'), { ctx: null, threads: null, parallel: null, layers: null });
    assert.ok(within(105, 100, 0.1) && !within(111, 100, 0.1) && !within(null, 100, 0.1));
});

test('ram fit plan: the cap sits between the two needs, and a flat need has no plan', () => {
    const plan = planRamFit({ ctx: 4096, needDefault: 1_909_599_072, needHalved: 1_603_152_736 });
    assert.deepEqual({ from: plan.from, to: plan.to }, { from: 4096, to: 2048 });
    assert.ok(plan.cap > 1_603_152_736 && plan.cap < 1_909_599_072);
    assert.equal(planRamFit({ ctx: 4096, needDefault: 1_000_000_000, needHalved: 1_000_000_000 }), null);
});

test('GPU bands: incompatible above 0.97 of the cap, else ok (the cap adds the 256 MiB margin: no band is refused for now)', () => {
    const cap = 2000 * MIB;
    assert.equal(gpuVerdict(1000 * MIB, cap), 'ok');
    assert.equal(gpuVerdict(cap - 256 * MIB + 1, cap), 'ok', 'the range the old rule refused for now');
    assert.equal(gpuVerdict(cap * 0.97, cap), 'ok');
    assert.equal(gpuVerdict(cap * 0.97 + 1, cap), 'incompatible');
    // The cap that admits a need leaves it `ok`, and a cap one byte lower for the same need does not.
    for (const need of [400 * MIB, 1000 * MIB, 9000 * MIB]) assert.equal(gpuVerdict(need, gpuCapAdmitting(need)), 'ok', String(need));
    assert.ok(gpuCapAdmitting(1000 * MIB) > 1000 * MIB / 0.97, 'with slack above the 0.97 limit');
    assert.equal(gpuVerdict(1000 * MIB, Math.ceil(1000 * MIB / 0.97) - 1), 'incompatible');
});

test('GPU fit-context plan: skipped at or below the floor and when the KV gap is inside the slack of the cap, planned when it is not', () => {
    const small = planGpuFitContext({ ctx: 16384, gpuAt: (ctx) => (ctx === 16384 ? 1000 * MIB : 984 * MIB) });
    assert.match(small.skip, /too small to open a gap/);
    // A little wider than the slack of the cap (3 %) is enough to plan one.
    const narrow = planGpuFitContext({ ctx: 16384, gpuAt: (ctx) => (ctx === 16384 ? 1080 * MIB : 984 * MIB) });
    assert.equal(narrow.skip, undefined);
    assert.equal(gpuVerdict(1080 * MIB, narrow.cap), 'incompatible');
    assert.equal(gpuVerdict(984 * MIB, narrow.cap), 'ok');
    assert.match(planGpuFitContext({ ctx: 8192, gpuAt: () => 1 }).skip, /not above the fit context floor 8192/);
    // A model whose KV cache is large: halving 32768 -> 16384 saves 6 GiB of 20 GiB, wider than the slack.
    const big = planGpuFitContext({ ctx: 32768, gpuAt: (ctx) => (ctx === 32768 ? 20_000 * MIB : 14_000 * MIB) });
    assert.equal(big.skip, undefined);
    assert.deepEqual({ from: big.from, to: big.to }, { from: 32768, to: 16384 });
    assert.equal(gpuVerdict(20_000 * MIB, big.cap), 'incompatible');
    assert.equal(gpuVerdict(14_000 * MIB, big.cap), 'ok');
});

test('GPU fit-layers plan: layers move, and the end state is launchable (ok): no cap leaves one the machine refuses for now', () => {
    // 24 layers, 23.5 MiB per layer over a 415 MiB fixed part, as for the 0.5B model.
    const tiny = (n) => Math.round((415 + 23.5 * n) * MIB);
    const plan = planGpuFitLayers({ layers: 24, gpuAtLayers: tiny });
    assert.ok(plan.expectLayers >= 1 && plan.expectLayers < 24);
    assert.equal(plan.expectStatus, 'ok');
    assert.equal(gpuVerdict(tiny(24), plan.cap), 'incompatible');
    assert.notEqual(gpuVerdict(tiny(plan.expectLayers), plan.cap), 'incompatible');
    assert.equal(gpuVerdict(tiny(plan.expectLayers + 1), plan.cap), 'incompatible', 'it ends at the largest layer count that is not incompatible');
    // A large model (80 layers, 400 MiB per layer).
    const large = (n) => Math.round((2000 + 400 * n) * MIB);
    const better = planGpuFitLayers({ layers: 80, gpuAtLayers: large });
    assert.equal(better.expectStatus, 'ok');
    assert.equal(gpuVerdict(large(80), better.cap), 'incompatible');
    // Nothing to move when one layer already needs more than any cap that admits a smaller need could give.
    assert.equal(planGpuFitLayers({ layers: 1, gpuAtLayers: () => 1 }), null);
});

test('a failed case becomes inconclusive for the memory guard only when no check failed', () => {
    const pass = { name: 'a', status: 'pass' };
    const fail = { name: 'b', status: 'fail' };
    assert.equal(settleFailedCase({ status: 'fail', checks: [pass], guardStopped: true }), 'inconclusive', 'an error with only passed checks, after the guard stopped the runner');
    assert.equal(settleFailedCase({ status: 'fail', checks: [], guardStopped: true }), 'inconclusive', 'a case that failed before any check');
    assert.equal(settleFailedCase({ status: 'fail', checks: [pass, fail], guardStopped: true }), 'fail', 'a failed check is a finding, whatever the guard did');
    assert.equal(settleFailedCase({ status: 'fail', checks: [fail], guardStopped: false }), 'fail', 'no guard stop: unchanged');
    assert.equal(settleFailedCase({ status: 'pass', checks: [pass], guardStopped: true }), 'pass', 'only a failed case is downgraded');
    assert.equal(settleFailedCase({ status: 'skip', checks: [], guardStopped: true }), 'skip');
});

test('the figure of a refusal for now: RAM and GPU wordings, and none for a disk shortage', () => {
    assert.equal(availableNowBytes('Needs about 2.0 GiB of RAM; 3.00 GiB is available now.'), 3 * GIB);
    assert.equal(availableNowBytes('Needs about 4.00 GiB of GPU memory; 2.50 GiB is free now; other GPU users: Xorg (0.20 GiB).'), Math.round(2.5 * GIB));
    assert.equal(availableNowBytes('Only 512 MiB of GPU memory is free; other GPU users: x (1.00 GiB).'), 512 * MIB);
    assert.equal(availableNowBytes('The weights need 4.2 GiB of free disk (download or copy, with a 5 % reserve); 1.0 GiB is free.'), null);
});

test('insufficient-now is the cap\'s doing when the reported figure is clearly below the physical one, and the machine\'s otherwise', () => {
    const hardware = { memory: { availableBytes: 20 * GIB }, gpu: { freeBytes: 10 * GIB } };
    const ram = (gib) => ({ status: 'insufficient-now', reason: `Needs about 2.00 GiB of RAM; ${gib} GiB is available now.` });
    assert.equal(insufficientNowCause(ram('1.50'), hardware).cause, 'cap', 'RAM reported far below what the machine has free');
    assert.equal(insufficientNowCause(ram('20.00'), hardware).cause, 'machine', 'RAM reported equal to the physical figure');
    assert.equal(insufficientNowCause(ram('19.95'), hardware).cause, 'machine', 'a gap inside the slack is noise');
    const gap = (20 * GIB - CAP_LIMIT_SLACK_BYTES) / GIB;
    assert.equal(insufficientNowCause(ram((gap - 0.01).toFixed(2)), hardware).cause, 'cap', 'just beyond the slack');
    const gpu = (gib) => ({ status: 'insufficient-now', reason: `Needs about 6.00 GiB of GPU memory; ${gib} GiB is free now.` });
    assert.equal(insufficientNowCause(gpu('3.00'), hardware).cause, 'cap', 'a GPU reason is compared with the GPU free figure, not the RAM one');
    assert.equal(insufficientNowCause(gpu('10.00'), hardware).cause, 'machine', 'a busy GPU reports its physical free memory');
    assert.equal(insufficientNowCause(gpu('15.00'), { memory: { availableBytes: 20 * GIB }, gpu: { freeBytes: 15 * GIB } }).cause, 'machine');
    assert.equal(insufficientNowCause({ reason: 'The weights need 4.2 GiB of free disk; 1.0 GiB is free.' }, hardware).cause, 'unknown', 'no figure to compare');
    assert.equal(insufficientNowCause(ram('1.50'), {}).cause, 'unknown', 'no physical figure to compare');
    assert.equal(insufficientNowCause(ram('1.50'), hardware).physical, 20 * GIB);
});

test('L7 ram cap: the need plus 64 MiB, never below 1.25 x the measured RSS, and a second instance stays refused because of ourUse', () => {
    const need = 1_800 * MIB;
    const small = planOurUseCap({ need, rssBytes: 900 * MIB });
    assert.equal(small.cap, need + 64 * MIB, 'a runner below the need leaves the old cap');
    assert.equal(small.secondRefused, true);
    // A CUDA runner whose RSS is above its estimate: the cap follows 1.25 x the RSS, so the guard cannot trip on the steady size.
    const cuda = planOurUseCap({ need, rssBytes: 2_400 * MIB });
    assert.equal(cuda.cap, Math.ceil(2_400 * MIB * 1.25));
    assert.ok(cuda.cap >= 2_400 * MIB * 1.25 && cuda.cap > need + 64 * MIB);
    assert.equal(cuda.room, cuda.cap - 2_400 * MIB);
    assert.equal(cuda.secondRefused, true, 'the room beside the runner (0.6 GiB) is below the need');
    // A runner so small that the cap leaves room for a second instance: the case cannot show ourUse.
    assert.equal(planOurUseCap({ need, rssBytes: 100 * MIB }).secondRefused, true);
    assert.equal(planOurUseCap({ need: 100 * MIB, rssBytes: 4_000 * MIB }).secondRefused, false);
    // Nothing measured yet: the old cap, and the live check decides.
    assert.deepEqual(planOurUseCap({ need, rssBytes: null }), { cap: need + 64 * MIB, room: need + 64 * MIB, secondRefused: null });
    assert.equal(planOurUseCap({ need, rssBytes: 0 }).secondRefused, null);
});

test('L5 explicit ctx under fit: refused on cpu and metal, admitted on dedicated with layers lowered and the ctx kept', () => {
    const model = 'qwen2.5-0.5b-instruct-q4_k_m';
    assert.equal(explicitCtxExpectation('dedicated'), 'admitted');
    assert.equal(explicitCtxExpectation('cpu'), 'refused');
    assert.equal(explicitCtxExpectation('metal'), 'refused');
    assert.deepEqual(fitLoweredSteps(`${model}: ok, fit lowered: nGpuLayers 99 -> 21`), [{ param: 'nGpuLayers', from: 99, to: 21 }]);
    assert.deepEqual(fitLoweredSteps(`${model}: incompatible`), []);
    const admitted = `${model}: ok, fit lowered: nGpuLayers 99 -> 21\n`;
    assert.equal(explicitCtxPreviewMatches({ profile: 'dedicated', stdout: admitted, model }), true);
    assert.equal(explicitCtxPreviewMatches({ profile: 'dedicated', stdout: `${model}: incompatible\nNeeds about 5 GiB`, model }), false, 'the old expectation (refused) fails on dedicated');
    assert.equal(explicitCtxPreviewMatches({ profile: 'dedicated', stdout: `${model}: ok, fit lowered: ctxSize 8192 -> 4096`, model }), false, 'an explicit ctx is never lowered');
    assert.equal(explicitCtxPreviewMatches({ profile: 'dedicated', stdout: `${model}: ok`, model }), false, 'no layers moved');
    assert.equal(explicitCtxPreviewMatches({ profile: 'metal', stdout: `${model}: incompatible\nNeeds about 5 GiB`, model }), true);
    assert.equal(explicitCtxPreviewMatches({ profile: 'cpu', stdout: admitted, model }), false, 'a pool profile must refuse');
    assert.equal(explicitCtxPreviewMatches({ profile: 'cpu', stdout: `${model}: incompatible, fit lowered: ctxSize 8192 -> 4096`, model }), false);
    assert.equal(explicitCtxLaunchMatches({ parsed: parseRunnerArgv('llama-server -c 8192 -ngl 21'), ctx: 8192, layers: 99 }), true);
    assert.equal(explicitCtxLaunchMatches({ parsed: parseRunnerArgv('llama-server -c 4096 -ngl 21'), ctx: 8192, layers: 99 }), false, 'ctx changed');
    assert.equal(explicitCtxLaunchMatches({ parsed: parseRunnerArgv('llama-server -c 8192 -ngl 99'), ctx: 8192, layers: 99 }), false, 'layers not lowered');
    assert.equal(explicitCtxLaunchMatches({ parsed: parseRunnerArgv('llama-server -c 8192'), ctx: 8192, layers: 99 }), false, 'no -ngl');
});

test('L3 a refused run leaves the deployment as it was: the same remembered idle one, or none', () => {
    const idle = { id: 'a1', phase: 'idle', updatedAt: '2026-10-19T10:00:00.000Z' };
    assert.equal(deploymentIdentity({ phase: 'idle', deployment: null }), null);
    assert.deepEqual(deploymentIdentity({ deployment: idle }), { ...idle, hasRunner: false });
    const before = deploymentIdentity({ deployment: idle });
    assert.equal(deploymentUnchanged(before, deploymentIdentity({ deployment: { ...idle } })), true, 'a remembered stopped deployment is not "no deployment"');
    assert.equal(deploymentUnchanged(null, null), true);
    assert.equal(deploymentUnchanged(null, before), false, 'a refusal created a deployment');
    assert.equal(deploymentUnchanged(before, null), false);
    assert.equal(deploymentUnchanged(before, deploymentIdentity({ deployment: { ...idle, id: 'b2' } })), false, 'a different deployment');
    assert.equal(deploymentUnchanged(before, deploymentIdentity({ deployment: { ...idle, updatedAt: '2026-10-19T10:00:01.000Z' } })), false, 'the timestamp moved');
    assert.equal(deploymentUnchanged(before, deploymentIdentity({ deployment: { ...idle, phase: 'starting' } })), false);
    const running = deploymentIdentity({ deployment: { ...idle, runner: { pid: 7 } } });
    assert.equal(deploymentUnchanged(running, running), false, 'a deployment with a runner is not the idle one');
});
