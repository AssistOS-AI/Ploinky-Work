---
title: DS008-standalone-host
summary: The local-llm host of a pworker home (lib/local-llm-host) on Linux and macOS: its layout, the ownership of a home, the launch records and the reaper, the Unix-socket runners, the host runner registry and locks, the macOS profiles, readers and directory checks, the gate and admin routes of the local-llm provider, the pworker models commands, and the host API.
---

# DS008 Standalone Host

## Introduction

`pworker serve` runs the engine of `lib/local-llm` for one Pworker home (the home of the server, `serverHome` of `lib/settings.mjs`: `config.taskHome`, an explicit home for tests and embedders, else the Pworker home, which for `pworker serve` is `PWORKER_HOME`; the data folder never moves it, so a `dataDir` such as `~/pw-data` cannot make a user's home folder the engine's home). The code that makes that safe lives in `lib/local-llm-host`, apart from the engine, because it belongs to pworker and not to every host: it decides where the files of the engine are, who may run the engine for a home, how a runner an earlier process left behind is found and ended, and which runners pworker offers. The host runs on Linux and on macOS (see macOS below). On any other platform, and on a macOS architecture other than `arm64` and `x64`, it reports itself unsupported and creates nothing. The engine's side of the same design is the `host.listen` input of DS007.

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
| A wildcard holder on macOS | macOS lets a bind on `127.0.0.1:<port>` succeed beside a listener of another process on `0.0.0.0:<port>` or `[::]:<port>` (observed with two processes on 2026-10-16). Linux refuses the loopback bind in that case, so Linux makes no extra probe. On macOS only, after the loopback bind succeeds, the owner binds `0.0.0.0:<port>` with a listener that destroys every connection and closes it again at once. `EADDRINUSE` there releases the loopback listener and refuses the home with `owner_port_taken`, worded as held by a program that is not pworker "on a wildcard address". Any other error of the probe is thrown. The probe binds `0.0.0.0` only; a dual-stack `[::]` holder makes that bind fail too (observed on macOS with a separate holder process, 2026-10-16), so both are refused. A holder on `::1` alone is not refused, as on Linux, where it does not block the loopback bind either. pworker never binds a wildcard address, so nothing of its own makes the probe fail |

`pworker serve` acquires the home before it listens, and waits up to 45 s for a predecessor that is still draining. If another pworker still owns the home, serve exits with `already` (exit code 3). Any other refusal (a port held by another program, an unusable socket directory) leaves the proxy running with the engine unavailable: the refusal is logged at start, is part of `GET /v1/local` (`local-llm.ownership.refusal`) and of `pworker models status`, and every use of the engine reports it. `host.drain()` called while an acquisition is still waiting for the home cancels it: nothing is bound or reaped after the shutdown request. A proxy that starts is waited for by `startProxy` for `START_WAIT_MS` (the 45 s wait for a predecessor plus 15 s) and by an autostarting client for 75 s; `startProxy` fails at once when the process exits without answering (exit 4: its port is taken by another program).

### Launch records and the reaper

The identity of a runner (start id, socket path, pid and start time) is recorded durably before the runner can load a model:

| Step | Record |
| --- | --- |
| Before the spawn | `{ schema: 1, startId, socketPath, phase: 'intent' }` |
| Right after it | `{ ..., phase: 'spawned', pid, startTime }`, where `startTime` is field 22 of `/proc/<pid>/stat` on Linux and, on macOS, the `lstart` column of `ps` as UTC epoch seconds (null when the process cannot be read) |
| When the runner and its group are gone | The record and the socket file are removed |

Every write goes to a temporary file in the same directory, is synced, renamed over the record, and the directory is synced. A failure to write the intent means nothing is spawned; a failure to complete the record stops the runner. A crash between the spawn and the second write leaves the intent record and a running child.

The reaper runs inside `host.acquire()` after the bind succeeded, so only an owner reaps; a process that did not get the port reads no record and signals nothing. For each record it finds the runner by identity, never by pid alone: the recorded pid when its start time and its command line (`--host <socket path>` as two arguments on Linux) still match, and any process whose command line names the socket path as its `--host` (the child of a crash before the second write). It reads the identity again right before each signal. A group leader gets SIGTERM for its group, then SIGKILL for the group after 10 s; members of the group seen before the first signal are ended one by one afterwards, each only if its start time is unchanged. A pid that now runs something else is never signalled. It then removes the socket, only when the file is named `<8 hex>.sock`, lies directly in the socket directory, is a socket and belongs to the user.

A record that is not valid (not JSON, wrong schema, a start id that is not 8 hex characters or does not match its file name, a socket path that is not `<startId>.sock`, a pid that is not above 1, an unknown phase) fails safe: nothing is signalled, the record is renamed `<name>.malformed`, and the log says why. On a platform that is neither Linux nor macOS the reaper does nothing and logs that.

On macOS the reaper reads the process table with `/bin/ps -ww -o pid=,pgid=,state=,lstart=,command=` under `TZ=UTC LC_ALL=C LANG=C` and `PATH=/usr/bin:/bin` (`lib/local-llm-host/darwinProcs.mjs`); on Linux the table is `/proc` (`procfs.mjs`). Everything above the table (the records, the signals, the group handling, the re-read before each signal) is shared. One difference holds. `ps` prints the command as one string, so the identity check on macOS is a text match: the command must contain ` --host <socket path>` followed by a space or the end of the string. Linux compares exact arguments, so on macOS an argument that itself contains that text cannot be told from the real one. Only `ps -p <pid>` exiting 1 with no output means the process does not exist. Any other outcome is unreadable: a timeout, a signal, a spawn error, a status other than 1, partial output, or a row that is not the pid asked for. An unreadable listing throws, so the reaper stops before it removes a record on the strength of "no such runner". An unreadable single read raises `ProcessUnreadable`: the reaper keeps that record and its socket, sends no further signal to it, and logs why. A state with no letter is read as dead only when it carries the `E` (exiting) flag, as `?NEs` does while a runner exits; any other letterless state is unreadable.

`CRASH_ENV` (`PWORKER_LOCAL_LLM_CRASH_AT`) is the deterministic seam for the crash tests and the live smoke script: with `after-intent` or `after-spawn` the process kills itself with SIGKILL at that point of the launch.

### Sockets

Runners listen on a Unix socket, not on a TCP port (the engine side is `host.listen`, DS007), so no program can squat a port before readiness and receive the per-start key. `--host <socketDir>/<8 hex>.sock` is the whole address; llama.cpp ignores the port for such a host. Readiness and requests go over the socket with `httpFetch(url, { socketPath })`.

| Rule | Detail |
| --- | --- |
| The directory | `<PWORKER_HOME>/r`, or `localLlm.socketDir`, which must be absolute. It is created 0700 when missing and checked either way: a real directory (not a link), owned by the user, mode exactly 0700. Anything else is refused with `invalid_socket_dir` |
| The length | The full socket path must fit `sun_path`: 107 bytes on Linux (103 on macOS). A longer path fails the run before anything is downloaded or launched, with `socket_path_too_long`, which names the limit and both fixes: a shorter `PWORKER_HOME`, or `localLlm.socketDir` |

### Runners and locks

On Linux:

| Runner | Offered | Source |
| --- | --- | --- |
| `llama.cpp` | When `nvidia-smi` is on PATH, and on the dedicated and unified profiles only: a machine that commits to the cpu profile (a broken driver, no usable device) hides it and refuses Run and Install for it | The CUDA build of the host lock (b11295): the main archive and the CUDA runtime archive, both unpacked into the same runnable copy. The binaries have `$ORIGIN` as their run path, so they find `libcudart` and `libcublas` beside them. `libcuda.so.1` comes from the driver and is listed as optional, so the install check passes on a machine without a GPU |
| `llama.cpp-cpu` | Always | The CPU build of the host lock (b11295) |
| ik_llama.cpp, Ollama, vLLM, TabbyAPI, LM Studio, the Vulkan build | Never | Listed as unsupported with a reason |

On macOS the host commits to one profile by CPU architecture and offers the one runner its lock holds:

| Machine | Profile | Runner offered | Lock |
| --- | --- | --- | --- |
| `darwin-arm64` | `metal`, fixed (no probe) | `llama.cpp`, the Metal build; its `profiles` are `metal` only | `runners.lock.host-darwin-arm64.json` |
| `darwin-x64` | `cpu`, fixed | `llama.cpp-cpu`. Not verified on hardware | `runners.lock.host-darwin-amd64.json` |

Every other runner is listed as unsupported with the reason "not offered by the standalone pworker on macOS in this release". The macOS host looks for no `nvidia-smi`.

The Linux locks are `lib/local-llm-host/locks/runners.lock.host-linux-{amd64,arm64}.json` and the macOS locks are `runners.lock.host-darwin-{arm64,amd64}.json` in the same directory (DS004 lists the macOS assets). They are passed to the engine as its image lock, because the engine offers an agent-lock entry on the cpu profile only. The seed catalog is `lib/local-llm-host/catalog/models.json`. Both are explicit inputs: the engine has no default for either.

The locks are proven by `.github/workflows/runner-lock-check.yml` (pull requests and on demand, never on a push, never skipped): one job per lock on a runner of its own architecture (`ubuntu-24.04`, `ubuntu-24.04-arm`, `macos-15` for `darwin-arm64`, `macos-15-intel` for `darwin-x64`; the macOS legs also check `uname -m`) runs `lib/local-llm-host/lockcheck.mjs`, which installs EVERY entry with the engine's installer (`runnerInstallCheck`, `--require-executable`: download, sha256, unpack, the launched file runs `--version` and names the pinned build) and then checks the libraries of the copy strictly. On Linux that is `ldd` on every ELF file (`lib/local-llm-host/elf.mjs`); a library that is not found fails the entry, except `libcuda.so.1`, which only the driver supplies. macOS has no `ldd`: the installer gets the macOS platform (`tar -m`), and `lib/local-llm-host/macho.mjs` finds every Mach-O file of the copy by its magic number and reads it with `otool -L` and `otool -l`. A dependency passes only when it is under `/usr/lib/` or `/System/Library/`, or is an `@rpath`, `@loader_path` or `@executable_path` name that resolves through the file's own run paths to a file inside the copy. An `otool` line the checker cannot read, and a slice with no library, fail the entry. The strict check exists because the engine's own check counts a library as provided when a file of that name is anywhere in the copy, which a broken run path does not make loadable. A lock with no entry, a skipped or validate-only report, or another version than `b11295` fails, and the final job `runner-lock-check` fails unless every leg (Linux amd64 and arm64, darwin-arm64 and darwin-x64) succeeded. The reports are uploaded as an artifact together with the revision tested (`git rev-parse HEAD`, `GITHUB_SHA`, the head of the pull request).

The runner reports the version of the host lock of its platform (`b11295` for `llama.cpp` and `llama.cpp-cpu`; the agent's image pins another build, which the agent path keeps). The gate reads that version through `host.runnerVersion(runnerId)` for the cache identity.

The engine's environment holds `PATH` (the system directories and, on Linux, the directory of `nvidia-smi`), `HOME`, `LANG` and `HF_TOKEN` when there is one; nothing else, so a `CUDA_MPS_*` variable or a provider key of the process never reaches a runner.

### macOS

On macOS `host.mjs` gives the engine its own platform and readers in place of the Linux ones:

| Piece | Behavior |
| --- | --- |
| Platform | `darwinPlatform` (DS007) with `privateRoot` set to the real path of `<PWORKER_HOME>/local-llm`. `socketPathMax` is 103, so the socket-path limit above is 103 bytes |
| Hardware snapshot, memory, pressure | The readers of `lib/local-llm-host/darwin.mjs` (`sysctl`, `vm_stat` and `ps` by absolute path, `LC_ALL=C`, `TZ=UTC`), given to the controller as `readHardwareSnapshot`, `readMemory` and `readPressure`. A figure that cannot be read is null and admission refuses (DS005) |
| Device probe | On the `metal` profile the device limit comes from `llama-server --list-devices`, run from the installed runnable copy `<home>/local-llm/run/llama.cpp/<version>/` once its `.ready.json` marker and launch file exist. Its environment holds `PATH`, `HOME` and `LANG` only, so `HF_TOKEN` is not passed. While the runner is not installed (or the probe fails or exceeds 15 s) the snapshot has no device limit, and admission refuses for that reason. The `cpu` profile of an Intel Mac runs no probe |
| Memory guard | The `metal` profile and, because `darwinPlatform` declares `pressureScale: 'level'`, the `cpu` profile of an Intel Mac use the kernel pressure-level guard of DS005, not a PSI share |
| Default runner | `defaultRunnerFor` gives `llama.cpp` for `metal` (and for `dedicated` and `unified`), `llama.cpp-cpu` for `cpu`, and `llama.cpp-vulkan` for `vulkan`, `venus` and `apu`. The standalone host offers no Vulkan build, so a request on such a profile is refused with that runner's own reason |
| Seed catalog | `qwen3-0.6b` (`Qwen3-0.6B-Q8_0.gguf`, pinned commit, size and sha256) is a seed model. Its `profiles` are `cpu`, `dedicated` and `unified`: `metal` is not a value of `profiles`, so a model reaches the metal profile through its `cpu` eligibility (`catalogProfileOf('metal')`). The entry carries `recommended.metal` (`ctxSize` 2048) |
| Reaper and launch records | `ps` as the process table (see Launch records and the reaper) |
| Ownership | The wildcard probe (see Ownership) |

**Directory chain check.** Before `prepareLayout` creates anything (and before the home itself is made), the host walks the path of `PWORKER_HOME` as it was given, component by component from `/`, the way OpenSSH's StrictModes does (`checkHomePath`). It follows each symbolic link itself, with a limit of 40 expansions, and keeps the real directory reached so far; a `..` in a link's target is applied to that real directory after the link, as the kernel does, and a home that itself has a `..` component is refused. Every component, link or directory, must be owned by the user or by root, and the real directory that holds it must be owned by the user or by root and have no group or other write bit unless it has the sticky bit. Otherwise another user could replace a link, or rename a directory, after the check and make the host's files somewhere else (the same shape as an operator-linked store root inside a writable directory, DS002). A holder with the sticky bit (`/tmp` is one) leaves only the case of the user's own processes, because other users cannot replace or rename what they do not own; so a link or directory owned by another user is refused even there. A component that is neither a directory nor a link, or cannot be read, is refused; a component that does not exist is allowed only while the home is not made yet, and then everything above it has been checked as its holder. The usual homes pass: `/var` and `/tmp` (links in `/` owned by root), `/private/var/folders/..` and a directory under `/Users/<user>`. A home under a directory that others can write and that is not sticky is refused, a real one included (it could be renamed away), and so is a link through such a directory, even when the link points at a safe one. The refusal is `invalid_host`, naming the component, its owner and mode and the holder, and suggesting a `PWORKER_HOME` under a directory only the user can write; nothing is created. Then the host takes the realpath of `PWORKER_HOME` (a `/var` that is a link to `/private/var` is the file system's own and is not refused). It checks with `lstat` that this directory and each of the host's directories under it that already exists (`local-llm`, `models`, `runners`, `run`, `state`, `logs`, `home`, `state/launch`) is a real directory, not a link, owned by the user, with no group or other write bit. Anything else is refused with `invalid_host`, naming the directory and the fix, and nothing is created. After the directories are created both checks run again with every directory required. The socket directory keeps its own check (Sockets). Linux makes no such check: its download binder asks the kernel where a descriptor points through `/proc/self/fd`, which macOS lacks (DS002). The engine's store removals on macOS check their directories first as well (DS002). These checks read the owner and the mode bits only: an access control list that lets others write a directory (`chmod +a`) is not seen, so a home whose directories carry such an entry is accepted (observed 2026-10-16).

**Pending.** The Metal inference smoke on a real model is run by the operator at pressure level 1 and has not run. `darwin-x64` has not run on hardware. The Metal constants of DS005 are provisional.

### The local-llm provider

`config.default.json` has a provider `local-llm` of `kind: "local-llm"`, enabled and in no tier. At most one such provider may exist, and it cannot have a `start` block (`createCore` refuses otherwise). Each provider kind has its own boot, status and action in the core: `boot` only starts `start` servers, `GET /v1/local` and `/stats` add the engine's row (`gate.status()`, which never creates the engine), `POST /v1/local/local-llm/stop` stops the deployment, and a start is refused with the command to use (`pworker models run <id>`).

A tier entry `{ "upstream": "local-llm", "model": "<catalog id>", "runner": "<id>" }` asks the gate (`gate.ensure({ model, runner })`) before the request is sent. The answer is `{ ok: true, target }` or `{ ok: false, reason }`, and a refusal takes the path of an unavailable local server: the entry is logged as not sent (503), the request falls to the next entry of the chain (`x-pworker-fallback-reason: unavailable`), and without one it is a `503 local_unavailable`.

| Rule | Detail |
| --- | --- |
| The base | `pworker models run` starts the single deployment. The gate uses a deployment that is ready for the requested model (and runner, when the entry names one) and never replaces or idle-stops one an admin started |
| On demand | With `localLlm.onDemand.enabled` the gate may start a model itself: only when its weights are complete and its runner is installed (a request never downloads; a run that finds its weights gone and starts fetching them is cancelled), with one start shared by all requests that arrive meanwhile, and a deployment it started is replaced only while nothing is running or queued for the provider. `idleStopMs` (default 15 min) stops a deployment the gate started, again only when nothing runs or waits. `startWaitMs` (default 120 s) bounds the wait for readiness |
| The forward | The request is sent over the target's Unix socket (`httpFetch(url, { socketPath })`) with the per-start key as the bearer token. The client's `Authorization` is never forwarded, and `body.model` becomes the runner's chat model; the request log keeps the catalog id. Streamed answers pass through unchanged |
| The cache | The identity of a cached answer is `local-llm:<model>:<runner>:<runner version>:<sha256 of the weights>`. With `localLlm.caps` set, `:caps-<digest of the caps>` is appended, so an answer cached under other caps is not reused. It is known without the engine only when the entry names a runner, or a deployment of the model is ready; otherwise nothing is read or stored, and a `strict` request is a miss |
| `/v1/models` | `GET /u/local-llm/v1/models` is built fresh from the seed catalog, the user's models and the deployment on every call; there is no time-to-live |
| Secrets | The Hugging Face token and every per-start key are added to the secrets the proxy removes from its logs and error texts |

### Caps

`localLlm.caps` in the user or project configuration declares the hardware that local models may plan for. `config.default.json` has no entry, and there are no command-line flags: an auto-started proxy would not carry them. The section is read once when the host is created (`lib/local-llm-host/caps.mjs`), so changing it means restarting `pworker serve`. With no `caps` the engine runs with its identity limits and every output, overview and cache identity is as without this feature.

| Key | Value | Effect |
| --- | --- | --- |
| `ram` | Bytes (a safe integer of at least 1 MiB), a string with a binary unit (`"12GiB"`, `"512MiB"`, `"1.5GiB"`), or a fraction above 0 and at most 1 of the machine's total memory (the first reading, then kept) | Admission sizes against this much host memory instead of the machine's: the cpu and Metal pool (`memoryPool`, whose refusal and warning name the cap) and the memory view of the dedicated and unified policies (`effective`) |
| `gpuMemory` | The same forms; a fraction is of the GPU's total (dedicated) or of system memory (unified) | Dedicated: the GPU the policy sizes against is `{ totalBytes: min(total, cap), freeBytes: min(free, cap) }`. Unified: folded into the memory view, as the smaller of `ram` and `gpuMemory`. Published in the overview only when the machine has a usable NVIDIA GPU. No effect on the cpu and Metal profiles, and refused on macOS (`invalid_caps`): use `ram` there |
| `mode` | Only `"refuse"` (the default); `"fit"` is not supported yet and is refused | A request over a cap is refused; there is no mode that shrinks it |
| `threads` | A whole number of at least 1 | An upper bound on the `--threads` the llama-server runners are started with. The default thread count is lowered to it on every profile, the Metal and high-performance-core path included (`capThreads`, DS007). A `threads` the caller sets above it is refused |
| `maxContext` | A whole number of at least the smallest context a runner accepts (512) | An upper bound on `ctxSize` (`--ctx-size`, `-c` on Metal). A default or saved context above it is lowered to it, and one the caller sets above it is refused |
| `maxParallel` | A whole number of at least 1 | An upper bound on `parallel` (`-np`), with the same rule |

A bad section (an unknown key, a wrong type, zero or a negative size, `mode: "fit"`) is kept when the host is created and thrown as `invalid_caps`, naming the key, by `host.controller()` before the engine is created. `acquire()` is not affected, so the orphan reap still runs first. `pworker models status` prints the caps in force from `limits.budget.caps` of the overview, and so do a refused `preview`, `run` and a run that does not start.

What a cap is, and is not:

| Cap | Enforced by | Strength |
| --- | --- | --- |
| `gpuMemory` | Admission's estimate against the capped GPU | Sizing only. Nothing stops a runner using more GPU memory once it runs |
| `threads`, `maxContext`, `maxParallel` | The llama-server flags `--threads`, `--ctx-size` and `-np` | Hard bounds on the flags the runner is started with: the runner cannot exceed them |
| `ram` | Admission's estimate against the capped memory | Sizing only in this release. The memory guard is unchanged and watches the machine's available memory, not the cap, and the runners' own memory is not subtracted from the cap (a later release) |
| GPU compute share | Nothing | Not offered |
| llama-servers that the proxy itself starts (`upstreams.<name>.start`) | Nothing | Outside the caps: their memory is neither limited nor counted |
| Runners that pworker does not offer (Ollama, vLLM, TabbyAPI, LM Studio, the Vulkan build) | Nothing | Not offered, so not capped |

A value above a count cap is handled by where it came from (`capParams`, DS007). A default, a catalog-recommended or a saved value is lowered to the cap. A value the caller set in this request (`--ctx`, `--parallel`, `--param threads=8`, or the HTTP `params`) is refused as `invalid_params` (HTTP 400, and the CLI prints the message), naming `localLlm.caps.<key>` and both numbers, before anything is downloaded or started. A `preview` throws the same refusal instead of returning it as a form error. What a Run saves is the set before the caps, so removing a cap restores the earlier values; the deployment and its launch use the capped set. A `threads` left empty is the engine's default thread count, which `capThreads` lowers.

A cap bounds what a model may plan for; the host reserve and the memory-guard floor stay those of the physical machine (the capped pool and memory view carry the physical figures beside the capped ones, DS007). A model must fit both the cap and the machine less its reserve. The unified policy has a fixed need of about 10 GiB (prompt cache, runtime and buffers) before the weights, so a `gpuMemory` under about 11 GiB admits nothing there.

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

Without `runnerId`, `preview` and `run` use the profile's default runner, the one the gate also picks: `llama.cpp-cpu` on the `cpu` profile, `llama.cpp` on `dedicated`, `unified` and `metal`, and `llama.cpp-vulkan` on `vulkan`, `venus` and `apu` (which the standalone host does not offer). When that runner is not supported on this machine, both answer 409 `runner_unsupported`, name the reason and point at `--runner`; nothing runs.

Errors are `{ error: { type, message } }` with the message redacted. A body that is not a JSON object is 400 `invalid_json`, one above 32 MiB is 413, an unknown model or runner is 404, a conflict (busy, in use, runner not installed, licence needed) is 409, an admission refusal is 422, invalid parameters (a value over a cap among them) are 400, a failed Hugging Face lookup is 502 and an engine that is unavailable is 503.

### Commands

`lib/pworker/models-cli.mjs` runs the engine verbs of `pworker models` against these routes: `catalog`, `runners`, `lookup`, `add`, `update`, `remove`, `preview`, `run [--detach]`, `stop`, `status [--follow] [--json]`, `cancel`, `delete-weights`, `runner install|uninstall` and `hf-token`. `models run` polls the status every second and shows the phase, bytes, rate and time left (a line that updates in place on a terminal, a line on each change otherwise); it exits 0 when the model is ready, 1 when it fails or pauses, and 130 when interrupted (the download goes on). `models stop NAME` and `models start NAME` are not engine verbs: they keep calling `/v1/local/NAME/(stop|start)`. `models hf-token` writes `keys/huggingface.env` in the home of the server (`serverHome`, the same function the host's home comes from: `PWORKER_HOME` unless the layered configuration names a `taskHome`; the configuration is the one of the current directory, as `pworker serve` reads it) as one line (`HF_TOKEN="..."`), through a temporary file created 0600 and renamed over the old one, so a replacement is 0600 whatever the old mode was; the token is read with hidden input or from stdin when it is not a terminal, and is never printed or logged. It does not restart the proxy: the engine reads the token when it is created, so the command says to run `pworker stop` and `pworker start`.

### Tests that touch a home

No test creates directories, binds an owner port or reaps in a home it does not own. Nothing in the gate, the core, `/v1/local`, `/stats`, `/v1/models` or the stop routes creates the engine (`test/local-llm-host/proxy-gate.test.mjs` runs them against a file system and a network that refuse every change, with a negative control); a server builds its host on its own home; `test/home.mjs` gives every test a scratch `PWORKER_HOME` and points `HOME` at it; and each test of the host uses its own temporary home.

### The live smoke test

`test/live/engine-smoke.mjs` is the proof on a real Linux machine; it has no macOS counterpart yet (see the pending items under macOS). The script is not part of `npm test`, and refuses unless `PWORKER_LIVE_ENGINE=1` (`test/local-llm-host/live-smoke-guard.test.mjs` pins that). On a fresh short home (`/tmp/pws-*`, so the socket path fits) and its own port it starts `pworker serve`, installs the runner (`llama.cpp` when `nvidia-smi` answers, else `llama.cpp-cpu`; `--runner` overrides), runs the seed model until ready, reads the launch record, and asserts: the runner listens on a Unix socket under `<home>/r` and has no TCP listener (`/proc/<pid>/fd` against `/proc/net/tcp*` and `/proc/net/unix`); one chat through a tier answers 200 with `x-pworker-model: local-llm/...`; on a GPU host the runner log and the status report show `offloaded N/N` and a CUDA device; `ldd` finds every library of every ELF file of the runnable copy (with a GPU, `libcuda.so.1` too); after `models stop` no runner, socket or launch record is left, and after `pworker stop` nothing of the home runs. It writes `steps.json` (also the architecture, GPU, driver, git revision and whether the tree is dirty) and never writes the Hugging Face token or the key of a run.

Two probes drive the crash recovery on the real system, each ending inconclusive (exit 3) rather than passing when the setup it needs did not happen. `--probe ready-kill` sends SIGKILL to the serve process while the model is ready, requires that the orphaned runner is still alive (same pid and start time and `--host`), runs `pworker start`, and requires the reaper's log line for that runner, the runner gone, and no socket or record left. `--probe record-kill [--at after-spawn|after-intent]` starts serve with `PWORKER_LOCAL_LLM_CRASH_AT`, so that serve kills itself at the launch-record transition, checks the state it left (an intent record and, after the spawn, one live child that the record does not name), runs `pworker start`, and requires the reaper's line and no process, socket or record left.

### The host API

`lib/local-llm-host/host.mjs` documents it at its top. In short: `acquire({ waitMs })`, `controller()`, `current()`, `drain()`, `describe()` (which never throws: an unusable socket directory is reported as `socketDirProblem`), `engineEnvironment()`, `redact(text)` and `secretValues()`. A failure to read the launch records at acquire is logged and does not undo the ownership. The server creates one host per home (`server.localLlm`), passes it to the core as `localLlm`, and `close()` awaits `drain()`, so a process that exits leaves no runner and no ownership behind. `close()` releases the listener and drains the engine even when the home's `jobs` folder was removed under the server (there is then no pending task to hand back) or handing the pending tasks back fails; such a failure is raised after the drain (a failure of the drain itself takes its place). The gate and the `/v1/local-llm/*` routes use `await host.controller()`.

## Decisions and Rationale

| Decision | Reason |
| --- | --- |
| Ownership by a bind, not by a file | A file lock needs a stale-lock rule, and every such rule has an interleaving in which a delayed reclaimer displaces a live owner. A bind has no stale state |
| The reaper identifies by the socket path as well as by the pid | The window between the spawn and the second record write cannot be closed by writing faster; it can be covered by a name only the runner has |
| The refusal of an unrelated port holder does not stop serve | One chance in about 12768 per home should not make the whole proxy unusable; the engine's refusal is clear and specific |
| `test/local-llm-host` uses a fake `/proc` and a recording kill | No test may signal a process it did not start. The real `/proc` is exercised only on Linux, in `linux-orphan.test.mjs`, through a kill that reaches one pid. The macOS tests (`darwin-procs.test.mjs`, `darwin-host.test.mjs`) feed recorded `ps` text and a recording kill |
| A wildcard probe on macOS, not a different port rule | The port stays a function of the home, with no fallback. The probe closes the one gap macOS has, a loopback bind beside a foreign wildcard listener, and Linux keeps its single bind |
