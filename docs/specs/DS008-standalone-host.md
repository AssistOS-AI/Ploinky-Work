---
title: DS008-standalone-host
summary: The local-llm host of a pworker home (lib/local-llm-host): its layout, the ownership of a home, the launch records and the reaper, the Unix-socket runners, the host runner registry and locks, the gate and admin routes of the local-llm provider, the pworker models commands, and the host API.
---

# DS008 Standalone Host

## Introduction

`pworker serve` runs the engine of `lib/local-llm` for one Pworker home (the home of the server, `serverHome` of `lib/settings.mjs`: `config.taskHome`, an explicit home for tests and embedders, else the Pworker home, which for `pworker serve` is `PWORKER_HOME`; the data folder never moves it, so a `dataDir` such as `~/pw-data` cannot make a user's home folder the engine's home). The code that makes that safe lives in `lib/local-llm-host`, apart from the engine, because it belongs to pworker and not to every host: it decides where the files of the engine are, who may run the engine for a home, how a runner an earlier process left behind is found and ended, and which runners pworker offers. This release covers Linux. On another platform the host reports itself unsupported and creates nothing (the macOS host is a later phase). The engine's side of the same design is the `host.listen` input of DS007.

## Core Content

### Layout of a home

| Path under `<PWORKER_HOME>` | Content | Mode |
| --- | --- | --- |
| `local-llm/models`, `local-llm/runners`, `local-llm/run`, `local-llm/state`, `local-llm/logs`, `local-llm/home` | Weights, the verified runner cache, the runnable copies of runners, the engine's state, its logs, and the `HOME` of runner processes | 0700 |
| `local-llm/state/owner.json` | The pid, port and start time of the last owner. Informational only (see Ownership) | 0600 |
| `local-llm/state/launch/<startId>.json` | The launch record of one runner start (see Launch records) | 0600 |
| `r/` | The Unix sockets of runners, `<8 hex>.sock` (see Sockets). `localLlm.socketDir` replaces it | 0700, checked |
| `keys/huggingface.env` | `HF_TOKEN=...`, mode 0600. The environment variable wins over the file. `pworker models hf-token` writes it in the home of the server, as the host reads it | 0600 |

Nothing is created until a process acquires the home. Creating the host (`createLocalLlmHost`) has no side effect.

### Ownership

A home is owned by the process that holds an exclusive loopback TCP listener on the home's own port. The kernel grants a bind to one process at a time and takes it back when the holder exits, however it exits, so there is never a stale lock to reclaim, and nothing in the host reclaims anything.

| Rule | Detail |
| --- | --- |
| The port | A function of the realpath of the home: 20000 plus a SHA-256 value modulo 12768, so it lies in 20000-32767, below the Linux (32768-60999) and macOS (49152-65535) ephemeral ranges and away from the proxy's 18080 and the 196xx ports. There is no fallback port |
| The grant | A successful bind on `127.0.0.1` with no `reusePort` and without `exclusive: false`. `EADDRINUSE` means held: the controller is not created |
| The handshake | The owner answers every connection with one JSON line (a tag, its pid and the home) and closes it. After a failed bind the contender reads that line, only to word the refusal: `owner_held` (a pworker of this home, with its pid), `owner_port_collision` (a pworker of another home that hashes to the same port) or `owner_port_taken` (anything else, including silence). A connection that is refused, or closed without a byte, means the holder left between the failed bind and the handshake: the contender binds again (up to 40 times, 25 ms apart) before it calls the holder unrelated. The handshake never grants ownership |
| `owner.json` | Written in place after the bind, for people. Nothing reads it to decide anything, and no code path moves, renames or removes it (`test/local-llm-host/ownership.test.mjs` scans the source and traces the file system calls) |
| Child processes | The listener has close-on-exec, so a runner does not inherit it: after `SIGKILL` of the owner the next contender acquires at once, although the runner is still alive |
| Release | After the controller has drained (`host.drain()`), when the process closes, or when it dies |
| A wildcard holder on macOS | On macOS a program listening on `0.0.0.0:<port>` does not stop a bind on `127.0.0.1:<port>` (observed). Two pworker processes bind the same address, so they still exclude each other; only an unrelated wildcard holder goes unnoticed there |

`pworker serve` acquires the home before it listens, and waits up to 45 s for a predecessor that is still draining. If another pworker still owns the home, serve exits with `already` (exit code 3). Any other refusal (a port held by another program, an unusable socket directory) leaves the proxy running with the engine unavailable: the refusal is logged at start, is part of `GET /v1/local` (`local-llm.ownership.refusal`) and of `pworker models status`, and every use of the engine reports it. `host.drain()` called while an acquisition is still waiting for the home cancels it: nothing is bound or reaped after the shutdown request. A proxy that starts is waited for by `startProxy` for `START_WAIT_MS` (the 45 s wait for a predecessor plus 15 s) and by an autostarting client for 75 s; `startProxy` fails at once when the process exits without answering (exit 4: its port is taken by another program).

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
| `llama.cpp` | When `nvidia-smi` is on PATH, and on the dedicated and unified profiles only: a machine that commits to the cpu profile (a broken driver, no usable device) hides it and refuses Run and Install for it | The CUDA build of the host lock (b11295): the main archive and the CUDA runtime archive, both unpacked into the same runnable copy. The binaries have `$ORIGIN` as their run path, so they find `libcudart` and `libcublas` beside them. `libcuda.so.1` comes from the driver and is listed as optional, so the install check passes on a machine without a GPU |
| `llama.cpp-cpu` | Always | The CPU build of the host lock (b11295) |
| ik_llama.cpp, Ollama, vLLM, TabbyAPI, LM Studio, the Vulkan build | Never | Listed as unsupported with a reason |

The locks are `lib/local-llm-host/locks/runners.lock.host-linux-{amd64,arm64}.json`, passed to the engine as its image lock, because the engine offers an agent-lock entry on the cpu profile only. The seed catalog is `lib/local-llm-host/catalog/models.json`. Both are explicit inputs: the engine has no default for either.

The locks are proven by `.github/workflows/runner-lock-check.yml` (pull requests and on demand, never on a push, never skipped): one job per Linux lock on a runner of its own architecture (`ubuntu-24.04`, `ubuntu-24.04-arm`) runs `lib/local-llm-host/lockcheck.mjs`, which installs EVERY entry with the engine's installer (`runnerInstallCheck`, `--require-executable`: download, sha256, unpack, the launched file runs `--version` and names the pinned build) and then runs `ldd` on every ELF file of the copy (`lib/local-llm-host/elf.mjs`). A library that is not found fails the entry, except `libcuda.so.1`, which only the driver supplies. The strict check exists because the engine's own check counts a library as provided when a file of that name is anywhere in the copy, which a broken run path does not make loadable. A lock with no entry, a skipped or validate-only report, or another version than `b11295` fails, and the final job `runner-lock-check` fails unless every architecture's job succeeded. The reports are uploaded as an artifact together with the revision tested (`git rev-parse HEAD`, `GITHUB_SHA`, the head of the pull request).

The runner reports the version of the host lock (`b11295` for `llama.cpp`; the agent's image pins another build, which the agent path keeps).

The engine's environment holds `PATH` (the system directories and the directory of `nvidia-smi`), `HOME`, `LANG` and `HF_TOKEN` when there is one; nothing else, so a `CUDA_MPS_*` variable or a provider key of the process never reaches a runner.

### The local-llm provider

`config.default.json` has a provider `local-llm` of `kind: "local-llm"`, enabled and in no tier. At most one such provider may exist, and it cannot have a `start` block (`createCore` refuses otherwise). Each provider kind has its own boot, status and action in the core: `boot` only starts `start` servers, `GET /v1/local` and `/stats` add the engine's row (`gate.status()`, which never creates the engine), `POST /v1/local/local-llm/stop` stops the deployment, and a start is refused with the command to use (`pworker models run <id>`).

A tier entry `{ "upstream": "local-llm", "model": "<catalog id>", "runner": "<id>" }` asks the gate (`gate.ensure({ model, runner })`) before the request is sent. The answer is `{ ok: true, target }` or `{ ok: false, reason }`, and a refusal takes the path of an unavailable local server: the entry is logged as not sent (503), the request falls to the next entry of the chain (`x-pworker-fallback-reason: unavailable`), and without one it is a `503 local_unavailable`.

| Rule | Detail |
| --- | --- |
| The base | `pworker models run` starts the single deployment. The gate uses a deployment that is ready for the requested model (and runner, when the entry names one) and never replaces or idle-stops one an admin started |
| On demand | With `localLlm.onDemand.enabled` the gate may start a model itself: only when its weights are complete and its runner is installed (a request never downloads; a run that finds its weights gone and starts fetching them is cancelled), with one start shared by all requests that arrive meanwhile, and a deployment it started is replaced only while nothing is running or queued for the provider. `idleStopMs` (default 15 min) stops a deployment the gate started, again only when nothing runs or waits. `startWaitMs` (default 120 s) bounds the wait for readiness |
| The forward | The request is sent over the target's Unix socket (`httpFetch(url, { socketPath })`) with the per-start key as the bearer token. The client's `Authorization` is never forwarded, and `body.model` becomes the runner's chat model; the request log keeps the catalog id. Streamed answers pass through unchanged |
| The cache | The identity of a cached answer is `local-llm:<model>:<runner>:<runner version>:<sha256 of the weights>`. It is known without the engine only when the entry names a runner, or a deployment of the model is ready; otherwise nothing is read or stored, and a `strict` request is a miss |
| `/v1/models` | `GET /u/local-llm/v1/models` is built fresh from the seed catalog, the user's models and the deployment on every call; there is no time-to-live |
| Secrets | The Hugging Face token and every per-start key are added to the secrets the proxy removes from its logs and error texts |

### Admin routes

`/v1/local-llm/*` are answered before the request admission and the `POST /v1/*` forwarder, behind the Host/Origin guard and the optional `PWORKER_TOKEN` (no other credential).

| Route | Answer |
| --- | --- |
| `GET /v1/local-llm` | The overview: profile, hardware, runners, models with their weights and admission, the deployment, and `unsupportedRunners` (id and reason). When the engine cannot be had (disabled, unsupported, owned elsewhere) it answers 200 with `available: false` and the refusal |
| `GET /v1/local-llm/status?sinceSeq=N` | Phase, deployment, download progress, log lines since `N`, `startedBy` (`admin` or `on-demand`) |
| `POST /v1/local-llm/preview` | `{ modelId, runnerId?, params? }`: the admission verdict, no process |
| `POST /v1/local-llm/run` | `{ modelId, runnerId?, params?, replace?, requestId? }`: 202 `{ accepted \| duplicate, deployment }`; a repeated `requestId` is a duplicate |
| `POST /v1/local-llm/stop`, `/cancel` | Stop the deployment; cancel its download (409 `not_downloading` when there is none). Neither creates the engine |
| `POST /v1/local-llm/weights/delete`, `/models/lookup`, `/models/add`, `/models/update`, `/models/remove`, `/runners/<id>/install`, `/runners/<id>/uninstall` | The engine's operations of the same names |

Errors are `{ error: { type, message } }` with the message redacted. A body that is not a JSON object is 400 `invalid_json`, one above 32 MiB is 413, an unknown model or runner is 404, a conflict (busy, in use, runner not installed, licence needed) is 409, an admission refusal is 422, a failed Hugging Face lookup is 502 and an engine that is unavailable is 503.

### Commands

`lib/pworker/models-cli.mjs` runs the engine verbs of `pworker models` against these routes: `catalog`, `runners`, `lookup`, `add`, `update`, `remove`, `preview`, `run [--detach]`, `stop`, `status [--follow] [--json]`, `cancel`, `delete-weights`, `runner install|uninstall` and `hf-token`. `models run` polls the status every second and shows the phase, bytes, rate and time left (a line that updates in place on a terminal, a line on each change otherwise); it exits 0 when the model is ready, 1 when it fails or pauses, and 130 when interrupted (the download goes on). `models stop NAME` and `models start NAME` are not engine verbs: they keep calling `/v1/local/NAME/(stop|start)`. `models hf-token` writes `keys/huggingface.env` in the home of the server (`serverHome`, the same function the host's home comes from: `PWORKER_HOME` unless the layered configuration names a `taskHome`; the configuration is the one of the current directory, as `pworker serve` reads it) as one line (`HF_TOKEN="..."`), through a temporary file created 0600 and renamed over the old one, so a replacement is 0600 whatever the old mode was; the token is read with hidden input or from stdin when it is not a terminal, and is never printed or logged. It does not restart the proxy: the engine reads the token when it is created, so the command says to run `pworker stop` and `pworker start`.

### Tests that touch a home

No test creates directories, binds an owner port or reaps in a home it does not own. Nothing in the gate, the core, `/v1/local`, `/stats`, `/v1/models` or the stop routes creates the engine (`test/local-llm-host/proxy-gate.test.mjs` runs them against a file system and a network that refuse every change, with a negative control); a server builds its host on its own home; `test/home.mjs` gives every test a scratch `PWORKER_HOME` and points `HOME` at it; and each test of the host uses its own temporary home.

### The live smoke test

`test/live/engine-smoke.mjs` is the proof on a real Linux machine (it is not part of `npm test`, and refuses unless `PWORKER_LIVE_ENGINE=1`; `test/local-llm-host/live-smoke-guard.test.mjs` pins that). On a fresh short home (`/tmp/pws-*`, so the socket path fits) and its own port it starts `pworker serve`, installs the runner (`llama.cpp` when `nvidia-smi` answers, else `llama.cpp-cpu`; `--runner` overrides), runs the seed model until ready, reads the launch record, and asserts: the runner listens on a Unix socket under `<home>/r` and has no TCP listener (`/proc/<pid>/fd` against `/proc/net/tcp*` and `/proc/net/unix`); one chat through a tier answers 200 with `x-pworker-model: local-llm/...`; on a GPU host the runner log and the status report show `offloaded N/N` and a CUDA device; `ldd` finds every library of every ELF file of the runnable copy (with a GPU, `libcuda.so.1` too); after `models stop` no runner, socket or launch record is left, and after `pworker stop` nothing of the home runs. It writes `steps.json` (also the architecture, GPU, driver, git revision and whether the tree is dirty) and never writes the Hugging Face token or the key of a run.

Two probes drive the crash recovery on the real system, each ending inconclusive (exit 3) rather than passing when the setup it needs did not happen. `--probe ready-kill` sends SIGKILL to the serve process while the model is ready, requires that the orphaned runner is still alive (same pid and start time and `--host`), runs `pworker start`, and requires the reaper's log line for that runner, the runner gone, and no socket or record left. `--probe record-kill [--at after-spawn|after-intent]` starts serve with `PWORKER_LOCAL_LLM_CRASH_AT`, so that serve kills itself at the launch-record transition, checks the state it left (an intent record and, after the spawn, one live child that the record does not name), runs `pworker start`, and requires the reaper's line and no process, socket or record left.

### The host API

`lib/local-llm-host/host.mjs` documents it at its top. In short: `acquire({ waitMs })`, `controller()`, `current()`, `drain()`, `describe()` (which never throws: an unusable socket directory is reported as `socketDirProblem`), `engineEnvironment()`, `redact(text)` and `secretValues()`. A failure to read the launch records at acquire is logged and does not undo the ownership. The server creates one host per home (`server.localLlm`), passes it to the core as `localLlm`, and `close()` awaits `drain()`, so a process that exits leaves no runner and no ownership behind. The gate and the `/v1/local-llm/*` routes use `await host.controller()`.

## Decisions and Rationale

| Decision | Reason |
| --- | --- |
| Ownership by a bind, not by a file | A file lock needs a stale-lock rule, and every such rule has an interleaving in which a delayed reclaimer displaces a live owner. A bind has no stale state |
| The reaper identifies by the socket path as well as by the pid | The window between the spawn and the second record write cannot be closed by writing faster; it can be covered by a name only the runner has |
| The refusal of an unrelated port holder does not stop serve | One chance in about 12768 per home should not make the whole proxy unusable; the engine's refusal is clear and specific |
| `test/local-llm-host` uses a fake `/proc` and a recording kill | No test may signal a process it did not start. The real `/proc` is exercised only on Linux, in `linux-orphan.test.mjs`, through a kill that reaches one pid |
