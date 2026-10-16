// A command already in progress may finish before the drain goes on.
export const DRAIN_QUEUE_WAIT_MS = 1_000;
// A hardware query (nvidia-smi) that is stopped is killed at once and waited
// for this long to exit. The drain stops every query when it starts, so this
// wait runs during the command wait above and adds nothing to the worst case.
export const HARDWARE_QUERY_REAP_MS = 1_000;
// SIGTERM to SIGKILL for the runner during a drain.
export const DRAIN_RUNNER_GRACE_MS = 3_000;
