// The process starter of a Unix-socket host: it records the start (launch.mjs) around the spawn, so that no runner can be
// loading a model without a record the reaper can use.
//
//   1. the intent record  { startId, socketPath }  is written durably       (a failure here: nothing is spawned)
//   2. the runner is spawned, in its own process group
//   3. the pid and start time are added to the record
//   4. when the runner and its group are gone, the record and the socket file are removed
//
// `crashHook(point)` is the seam V16/V17 use to stop between the steps without timing luck. The points are
// 'after-intent' (step 1 done, nothing spawned) and 'after-spawn' (the runner exists, only the intent is on disk). The default
// hook kills this process with SIGKILL when `env.PWORKER_LOCAL_LLM_CRASH_AT` names the point; it is for tests and the live
// smoke script, and does nothing when the variable is not set.

import { startRunnerProcess } from '../local-llm/controller/runnerProcess.mjs';

export const CRASH_POINTS = Object.freeze(['after-intent', 'after-spawn']);
export const CRASH_ENV = 'PWORKER_LOCAL_LLM_CRASH_AT';

/** The hook that crashes this process (SIGKILL) at the point `env[CRASH_ENV]` names. */
export function envCrashHook(env, { kill = process.kill } = {}) {
    return (point) => {
        if (env?.[CRASH_ENV] === point) kill(process.pid, 'SIGKILL');
    };
}

export function createRunnerStarter({ recorder, socketDir, platform, killImpl = null, crashHook = () => {}, log = () => {}, startProcess = startRunnerProcess }) {
    return function startRunner(options) {
        const { listen } = options;
        if (!listen || typeof listen.startId !== 'string' || typeof listen.socketPath !== 'string') {
            throw new Error('A Unix-socket host starts every runner with a listen address');
        }
        recorder.intent(listen);
        crashHook('after-intent');
        const child = startProcess({ platform, ...(killImpl ? { killImpl } : {}), ...options });
        crashHook('after-spawn');
        if (Number.isInteger(child.pid)) {
            try {
                recorder.spawned({ ...listen, pid: child.pid });
            } catch (error) {
                // A runner nobody can find again must not keep running.
                log(`local-llm: the launch record of runner ${listen.startId} could not be completed (${error?.code ?? error?.message}); the runner is stopped`);
                child.kill();
                throw error;
            }
        }
        child.exited.then(() => {
            try {
                recorder.clear({ startId: listen.startId, socketPath: listen.socketPath, socketDir });
            } catch (error) {
                log(`local-llm: the launch record of runner ${listen.startId} could not be removed (${error?.code ?? error?.message})`);
            }
        });
        return child;
    };
}
