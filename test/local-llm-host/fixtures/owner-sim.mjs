// A pworker process reduced to its local-llm host, for the crash tests on a real /proc (Linux): it takes the home, starts the
// stand-in runner on the seeded model and holds. argv[2] is { home, ownerPort }. With PWORKER_LOCAL_LLM_CRASH_AT set it kills
// itself (SIGKILL) at that point of the launch. Prints `READY <runner pid>` once the model is ready.
import { createLocalLlmHost } from '../../../lib/local-llm-host/host.mjs';
import { controllerSeams, runModel } from './engine.mjs';

const { home, ownerPort } = JSON.parse(process.argv[2]);
const host = createLocalLlmHost({
    config: {}, home, env: { ...process.env, PATH: '/usr/local/bin:/usr/bin:/bin' }, platformName: 'linux', ownerPort,
    log: (line) => console.error(line), controllerOptions: controllerSeams(),
});
await host.acquire();
const controller = await host.controller();
await runModel(controller);
console.log(`READY ${controller.state.deployment.runner.pid}`);
setInterval(() => {}, 1000);
