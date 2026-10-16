// Skip reasons for the tests that need a Linux mechanism. A skip is a string on other platforms and false on Linux, where the test runs.
// Each skipped test failed at the macOS baseline of local-llms 34f8f498; none of them is skipped on Linux.
export const LINUX_PROC_FD = process.platform !== 'linux' && 'needs Linux: the download binder confirms which file was opened through /proc/self/fd';
export const LINUX_PROC_OOM = process.platform !== 'linux' && 'needs Linux: starting a runner writes /proc/<pid>/oom_score_adj';
