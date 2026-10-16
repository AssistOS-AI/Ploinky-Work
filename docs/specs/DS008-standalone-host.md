---
title: DS008-standalone-host
summary: The local-llm host of a pworker home (lib/local-llm-host): its layout, the ownership of a home, the launch records and the reaper, the Unix-socket runners, the host runner registry and locks, and the API the gate and the routes use.
---

# DS008 Standalone Host

## Introduction

`pworker serve` runs the engine of `lib/local-llm` for one Pworker home. The code that makes that safe lives in `lib/local-llm-host`, apart from the engine, because it belongs to pworker and not to every host: it decides where the files of the engine are, who may run the engine for a home, how a runner an earlier process left behind is found and ended, and which runners pworker offers. This release covers Linux. On another platform the host reports itself unsupported and creates nothing (the macOS host is a later phase). The engine's side of the same design is the `host.listen` input of DS007.

## Core Content

### Layout of a home

| Path under `<PWORKER_HOME>` | Content | Mode |
| --- | --- | --- |
| `local-llm/models`, `local-llm/runners`, `local-llm/run`, `local-llm/state`, `local-llm/logs`, `local-llm/home` | Weights, the verified runner cache, the runnable copies of runners, the engine's state, its logs, and the `HOME` of runner processes | 0700 |
| `local-llm/state/owner.json` | The pid, port and start time of the last owner. Informational only (see Ownership) | 0600 |
| `local-llm/state/launch/<startId>.json` | The launch record of one runner start (see Launch records) | 0600 |
| `r/` | The Unix sockets of runners, `<8 hex>.sock` (see Sockets). `localLlm.socketDir` replaces it | 0700, checked |
| `keys/huggingface.env` | `HF_TOKEN=...`, mode 0600. The environment variable wins over the file | 0600 |

Nothing is created until a process acquires the home. Creating the host (`createLocalLlmHost`) has no side effect.

### Ownership

A home is owned by the process that holds an exclusive loopback TCP listener on the home's own port. The kernel grants a bind to one process at a time and takes it back when the holder exits, however it exits, so there is never a stale lock to reclaim, and nothing in the host reclaims anything.

| Rule | Detail |
| --- | --- |
| The port | A function of the realpath of the home: 20000 plus a SHA-256 value modulo 12768, so it lies in 20000-32767, below the Linux (32768-60999) and macOS (49152-65535) ephemeral ranges and away from the proxy's 18080 and the 196xx ports. There is no fallback port |
| The grant | A successful bind on `127.0.0.1` with no `reusePort` and without `exclusive: false`. `EADDRINUSE` means held: the controller is not created |
| The handshake | The owner answers every connection with one JSON line (a tag, its pid and the home) and closes it. After a failed bind the contender reads that line, only to word the refusal: `owner_held` (a pworker of this home, with its pid), `owner_port_collision` (a pworker of another home that hashes to the same port) or `owner_port_taken` (anything else, including silence). The handshake never grants ownership |
| `owner.json` | Written in place after the bind, for people. Nothing reads it to decide anything, and no code path moves, renames or removes it (`test/local-llm-host/ownership.test.mjs` scans the source and traces the file system calls) |
| Child processes | The listener has close-on-exec, so a runner does not inherit it: after `SIGKILL` of the owner the next contender acquires at once, although the runner is still alive |
| Release | After the controller has drained (`host.drain()`), when the process closes, or when it dies |
| A wildcard holder on macOS | On macOS a program listening on `0.0.0.0:<port>` does not stop a bind on `127.0.0.1:<port>` (observed). Two pworker processes bind the same address, so they still exclude each other; only an unrelated wildcard holder goes unnoticed there |

`pworker serve` acquires the home before it listens, and waits up to 45 s for a predecessor that is still draining. If another pworker still owns the home, serve exits with `already` (exit code 3). Any other refusal (a port held by another program, an unusable socket directory) leaves the proxy running with the engine unavailable: the refusal is logged at start and every use of the engine reports it.

### Launch records and the reaper

The identity of a runner (start id, socket path, pid and start time) is recorded durably before the runner can load a model:

| Step | Record |
| --- | --- |
| Before the spawn | `{ schema: 1, startId, socketPath, phase: 'intent' }` |
| Right after it | `{ ..., phase: 'spawned', pid, startTime }`, where `startTime` is field 22 of `/proc/<pid>/stat` |
| When the runner and its group are gone | The record and the socket file are removed |

Every write goes to a temporary file in the same directory, is synced, renamed over the record, and the directory is synced. A failure to write the intent means nothing is spawned; a failure to complete the record stops the runner. A crash between the spawn and the second write leaves the intent record and a running child.

The reaper runs inside `host.acquire()` after the bind succeeded, so only an owner reaps; a process that did not get the port reads no record and signals nothing. For each record it finds the runner by identity, never by pid alone: the recorded pid when its start time and its command line (`--host <socket path>` as two arguments) still match, and any process whose command line names the socket path as its `--host` (the child of a crash before the second write). It reads the identity again right before each signal. A group leader gets SIGTERM for its group, then SIGKILL for the group after 10 s; members of the group seen before the first signal are ended one by one afterwards, each only if its start time is unchanged. A pid that now runs something else is never signalled. It then removes the socket, only when the file is named `<8 hex>.sock`, lies directly in the socket directory, is a socket and belongs to the user.

A record that is not valid (not JSON, wrong schema, a start id that is not 8 hex characters or does not match its file name, a socket path that is not `<startId>.sock`, a pid that is not above 1, an unknown phase) fails safe: nothing is signalled, the record is renamed `<name>.malformed`, and the log says why. Where `/proc` does not exist the reaper does nothing and logs that.

`CRASH_ENV` (`PWORKER_LOCAL_LLM_CRASH_AT`) is the deterministic seam for the crash tests and the live smoke script: with `after-intent` or `after-spawn` the process kills itself with SIGKILL at that point of the launch.

### Sockets

Runners listen on a Unix socket, not on a TCP port (the engine side is `host.listen`, DS007), so no program can squat a port before readiness and receive the per-start key. `--host <socketDir>/<8 hex>.sock` is the whole address; llama.cpp ignores the port for such a host. Readiness and requests go over the socket with `httpFetch(url, { socketPath })`.

| Rule | Detail |
| --- | --- |
| The directory | `<PWORKER_HOME>/r`, or `localLlm.socketDir`, which must be absolute. It is created 0700 when missing and checked either way: a real directory (not a link), owned by the user, mode exactly 0700. Anything else is refused with `invalid_socket_dir` |
| The length | The full socket path must fit `sun_path`: 107 bytes on Linux (103 on macOS). A longer path fails the run before anything is downloaded or launched, with `socket_path_too_long`, which names the limit and both fixes: a shorter `PWORKER_HOME`, or `localLlm.socketDir` |

### Runners and locks

| Runner | Offered | Source |
| --- | --- | --- |
| `llama.cpp` | When `nvidia-smi` is on PATH | The CUDA build of the host lock (b11295): the main archive and the CUDA runtime archive, both unpacked into the same runnable copy. The binaries have `$ORIGIN` as their run path, so they find `libcudart` and `libcublas` beside them. `libcuda.so.1` comes from the driver and is listed as optional, so the install check passes on a machine without a GPU |
| `llama.cpp-cpu` | Always | The CPU build of the host lock (b11295) |
| ik_llama.cpp, Ollama, vLLM, TabbyAPI, LM Studio, the Vulkan build | Never | Listed as unsupported with a reason |

The locks are `lib/local-llm-host/locks/runners.lock.host-linux-{amd64,arm64}.json`, passed to the engine as its image lock, because the engine offers an agent-lock entry on the cpu profile only. The seed catalog is `lib/local-llm-host/catalog/models.json`. Both are explicit inputs: the engine has no default for either.

The engine's environment holds `PATH` (the system directories and the directory of `nvidia-smi`), `HOME`, `LANG` and `HF_TOKEN` when there is one; nothing else, so a `CUDA_MPS_*` variable or a provider key of the process never reaches a runner.

### The host API

`lib/local-llm-host/host.mjs` documents it at its top. In short: `acquire({ waitMs })`, `controller()`, `current()`, `drain()`, `describe()`, `engineEnvironment()` and `redact(text)`. The server creates one host per home (`server.localLlm`), passes it to the core as `localLlm`, and `close()` awaits `drain()`, so a process that exits leaves no runner and no ownership behind. The gate and the `/v1/local-llm/*` routes use `await host.controller()`.

## Decisions and Rationale

| Decision | Reason |
| --- | --- |
| Ownership by a bind, not by a file | A file lock needs a stale-lock rule, and every such rule has an interleaving in which a delayed reclaimer displaces a live owner. A bind has no stale state |
| The reaper identifies by the socket path as well as by the pid | The window between the spawn and the second record write cannot be closed by writing faster; it can be covered by a name only the runner has |
| The refusal of an unrelated port holder does not stop serve | One chance in about 12768 per home should not make the whole proxy unusable; the engine's refusal is clear and specific |
| `test/local-llm-host` uses a fake `/proc` and a recording kill | No test may signal a process it did not start. The real `/proc` is exercised only on Linux, in `linux-orphan.test.mjs`, through a kill that reaches one pid |
