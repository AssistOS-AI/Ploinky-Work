---
title: DS007-host-limits-and-messages
summary: The three inputs a host gives the engine, its host (paths and platform), its messages (wording) and its limits (hooks), and how createLocalLlmController composes them.
---

# DS007 Host, Limits and Messages

## Introduction

The engine knows no container, no gateway and no limit. Everything that depends on the machine or on the program that embeds the engine comes in through three frozen values: the **host** (`defineHost`: where things are), the **messages** (`defineMessages`: how the engine words what names its surroundings) and the **limits** (`defineLimits`: hooks the engine calls at fixed points). With the defaults, an engine behaves as on a machine that has no limit and speaks neutral English. A host that must reproduce its own paths, wording and limits exactly passes its own values and gets its own results back unchanged. This specification lists every field, key and hook. Nothing in `lib/local-llm` reads a path, a variable or a limit that is not named here; `test/local-llm/engine-neutral.test.mjs` fails when engine code names a container, a gateway or a limit mechanism.

## Core Content

### Composing a controller

```js
import { createLocalLlmController, defineHost, defineMessages, defineLimits } from 'pworker/local-llm';

const controller = createLocalLlmController({
    host: defineHost({ dataDir: '/var/lib/models', runRoot: '/var/lib/run' }),
    messages: defineMessages({}),        // optional; DEFAULT_MESSAGES
    limits: defineLimits({}),            // optional; IDENTITY_LIMITS
    gatewayModelName: null,              // the model name the overview publishes
    env: process.env,                    // used by reference, never copied
    // ...seams
});
```

`host`, `messages` and `limits` may be the values the three `define` functions return or the plain objects they take; each is validated, and validating a defined value returns an equal one. The call composes the following, then calls `createController` (`lib/local-llm/controller/deployments.mjs`).

| Step | What is composed | Skipped when the caller passes |
| --- | --- | --- |
| Reaper | One hardware-query reaper for the controller, so one drain waits for every hardware query it started and no other controller's | `reaper` |
| Runners | `createRunners({ host, messages, limits })`: a frozen table with the ids, ports and schemas listed in `RUNNER_IDS` | `runners` |
| Locks | `loadRunnerLocks({ image: host.imageRunnerLockFile, agent: host.agentRunnerLockFile(hostArch) })`, with `hostArch` defaulting to `process.arch` | `runnerLocks` |
| Installer | `createRunnerInstaller` with the cache under `<dataDir>/runners`, `host.runRoot`, `host.uv`, `host.python` and `host.platform`, and a warning sink the controller later binds to its log | `installer` |
| Everything else | `createController` derives its other defaults from the host (state store, seed catalog, Hugging Face base URL, shared models root, shared-memory directory, caches under `runRoot`, readers of `/proc` through the platform) | the option of the same name |

The options after `env` are **seams**: any option of `createController` (a snapshot reader, a downloader, a clock, `killImpl`, a port table, ...). They are spread last, so they win over everything composed above. Tests use them; a host rarely needs one.

The overview key `gatewayModel` keeps its place and holds `gatewayModelName` (null by default). `env` is the object the caller passed: a switch the operator changes in it is seen at the next read, and every limits hook that receives `env` receives that same object.

### Public exports

| Subpath | Stability | Contents |
| --- | --- | --- |
| `pworker/local-llm` | stable | `createLocalLlmController`, `createController` (raw), `defineHost`, `linuxPlatform`, `defineMessages`, `DEFAULT_MESSAGES`, `MESSAGE_KEYS`, `defineLimits`, `IDENTITY_LIMITS`, `LIMIT_HOOKS`, `createRunners`, `RUNNER_IDS`, `runnerSummaries`, `ENGINE_DRAIN`, `LocalLlmError`, `serializeError`, `OBSERVATION_ID_RE`, `LMSTUDIO_SWITCH`, `VLLM_UNIFIED_SWITCH`, `createStateStore`, `validateModel`, `loadSeedCatalog`, `loadRunnerLocks`, `createRunnerInstaller`, `readSnapshot`, `computeDiagnostic`, `createStartupObservations`, `createObservationClock` |
| `pworker/local-llm/chat` | stable | The pure helpers of a chat responder: `buildRunnerRequest`, `requestLimitProblem`, `chatLimits(env)`, `cpuCompletionBudget(env)`, `completionStats`, `createStreamStatsReader`, `MAX_COMPLETION_TOKENS`, `MAX_COMPLETION_TOKENS_LIMIT`, `CPU_MAX_COMPLETION_TOKENS`, `RUNNER_TIMEOUT_MS`. The environment names (`LOCAL_LLM_MAX_COMPLETION_TOKENS`, `LOCAL_LLM_RUNNER_TIMEOUT_MS`) and the defaults are part of the contract. The responder that reads a request, calls the controller and writes the answer belongs to the host |
| `pworker/local-llm/runner-process` | stable | `startRunnerProcess`, `createLogBuffer`, `setOomScoreAdj`, `parseRunnerReport`, `createReportCollector` |
| `pworker/local-llm/runner-install-check` | stable | `runInstallCheck(argv, { host })` (DS004) |
| `pworker/local-llm/internal/*` | unstable | Maps to `lib/local-llm/*`. For the local-llms agent only: its tests and its limits plug-in reach modules the stable surface does not export. It carries no compatibility promise |

`ENGINE_DRAIN` is `{ queueWaitMs, hardwareQueryReapMs, runnerGraceMs }`, the stages of a drain that belong to the engine (1,000 ms, 1,000 ms and 3,000 ms). A host adds its own stages on top.

### The host (`defineHost`)

`defineHost(spec)` returns a deep-frozen value and touches neither the file system nor the environment. It throws `LocalLlmError('invalid_host', message, { field })` for a missing required field, a path that is not absolute, not normalized or that contains `:` (`toolPath` is a PATH list and is exempt), an unknown key at any level, a malformed part, or an `executables['llama.cpp-vulkan']` that is not the `vulkan.serverPath`. A path that is not given is null, and null switches off the feature that needs it instead of guessing a location.

| Field | If omitted | Controls |
| --- | --- | --- |
| `dataDir` (required) | throws | The persistent directory: state (`state/controller.json`), models, the runner cache (`runners/`), logs and the runners' `HOME` |
| `runRoot` (required) | throws | The directory of runnable copies (`<id>/<version>/`), staging (`.stage-<id>`), scratch (`.tmp-<id>`), the CUDA cache (`.cuda-cache`) and the Mesa cache (`.mesa-cache`) |
| `sharedModelsRoot` | null: no workspace-shared weights | The read-only store the weight stores may adopt from or copy (DS002) |
| `imageContractFile` | null: no contract, every runner counts as available | Key=value lines describing what the image contains (`vulkan_contract`, `gpu_compute_capabilities`, `architecture`, ...; DS005) |
| `imageRunnerLockFile` | null: an empty image lock | The image's runner lock (DS004) |
| `agentRunnerLockFile(arch)` | `() => null` | A function returning the host's own lock file for a CPU architecture (`process.arch` values), or null. It must return null, not undefined, for an architecture it does not know |
| `seedCatalogFile` | null: `createController` throws `missing_file` unless `seedCatalog` is injected | The seed catalog (DS002). The engine ships no catalog data |
| `hfBaseUrl` | `https://huggingface.co` | The Hugging Face origin. `http:` is accepted because tests use local servers |
| `nvidiaSmi` | null: the NVIDIA GPU is absent | The `nvidia-smi` executable |
| `nvidiaLibDir` | null: no `LD_LIBRARY_PATH`, no `TRITON_LIBCUDA_PATH`, no libcuda link | The driver library directory runners and the install check put on the library path |
| `executables` | `{}` | `{ 'llama.cpp', 'ik_llama.cpp', 'llama.cpp-vulkan', ollama }`: the executables the image holds. A runner without one is offered only through a lock's runnable copy |
| `vulkan` | null: no Vulkan contract, Vulkan launches refused | `{ serverPath, vulkaninfo, icdDir }`: the Vulkan build of llama.cpp, `vulkaninfo`, and the directory of the ICD files |
| `lmStudioSdkDir` | null: LM Studio is `supported: false` with a reason | The directory of the LM Studio SDK the adapter loads |
| `uv`, `python` | null: Python runners are not installable and the build is refused | The tools of the Python runners' virtual environments |
| `shmDir` | `/dev/shm` | The shared-memory directory where PyTorch runners keep sockets |
| `toolPath` | `/usr/local/bin:/usr/bin:/bin` | The `PATH` of the processes the engine starts |
| `gpuAccess` | null: never reported | `{ report(env) -> { unavailable, reason }, source }`: why the host has no GPU device. The engine passes `reason` through `plainDiagnosticText(reason, 300)` and words it with `gpuAccessReported`; `source` is the diagnostic source tag |
| `platform` | `linuxPlatform()` | The operating-system strategies (below) |
| `listen` | null: a runner listens on loopback TCP, on the port its adapter names, exactly as before | `{ socketDir, pathHint? }`: a runner listens on the Unix socket `<socketDir>/<8 hex>.sock` of its start and has no TCP port. `socketDir` is an absolute path with no `:` or `,`. The engine checks the byte length of the socket path against `platform.socketPathMax` before it downloads or launches anything and refuses with `socket_path_too_long`, which names the limit; `pathHint` is the host's own advice, appended to that message. Only a runner that declares `listensOnSocket` (llama.cpp's own server) can start on such a host; any other is refused with `runner_unavailable`. The host creates and checks `socketDir` (owner, mode 0700, not a link); the engine does not touch it |

### The platform (`linuxPlatform`)

`linuxPlatform({ procDir = '/proc', devDriDir = '/dev/dri' })` is frozen. `defineHost` accepts any value with the same members, so a platform for another operating system can be written without changing the engine.

| Member | Meaning | Linux value |
| --- | --- | --- |
| `name` | A label for logs | `'linux'` |
| `procDir` | Where `meminfo`, `cpuinfo`, `pressure/memory` and `<pid>/oom_score_adj` are read | `/proc` |
| `devDriDir` | Where GPU render nodes are looked for | `/dev/dri` |
| `openedPathOf(fsApi, fd)` | Where the kernel says an open descriptor points; used to confirm which file a bound path opened | `readlink <procDir>/self/fd/<fd>` |
| `openatPath(fd, name)` | `name` inside an open directory, the way `openat(2)` names it, so a file below an untrusted root is reached through a descriptor and not through a path | `<procDir>/self/fd/<fd>/<name>` |
| `unreadableReason` | The text appended to `UNSAFE_PATH` when the descriptor's path cannot be read | `<procDir>/self/fd is not readable` |
| `setOomScoreAdj(pid, value, { fsApi })` | Makes a process the out-of-memory killer's first choice | writes `<procDir>/<pid>/oom_score_adj` |
| `tarFlags(into, strip)` | The extraction flags of the runner installer | `['-C', into, '--strip-components=<strip>', '--no-same-owner', '--touch']` |
| `socketPathMax` | The longest Unix socket path in bytes (`sun_path` less its terminator); optional, 107 when absent | `107` |

### The messages (`defineMessages`)

`defineMessages(overrides)` takes one function per key, each returning a whole sentence, so a host can keep its wording byte for byte. An unknown key or a value that is not a function throws `LocalLlmError('invalid_messages', message, { key })`. Wording that only describes the engine's own behavior stays in the engine and is not a key.

| Key (arguments) | Used when | Neutral default |
| --- | --- | --- |
| `notReady()` | The chat target is asked for and no model is ready (`not_ready`) | `No local model is ready; run one first.` |
| `gpuAccessReported(reason \| null)` | `nvidia-smi` is unavailable and `host.gpuAccess` reported a reason (null: none given) | `The host reports: <reason or "GPU access is unavailable; the cause is unknown.">` |
| `gpuAbsent()` | `nvidia-smi` is unavailable and the host reported nothing | `No GPU is available to this server.` |
| `hfTokenHint()` | The end of the refusal for a gated or private Hugging Face repository | `provide a Hugging Face token (HF_TOKEN) and restart the server` |
| `vulkanRuntimeMissing()` | A render node is attached and the host declares no Vulkan layout | `A GPU render node is attached, but this host declares no supported Vulkan runtime.` |
| `apuSwitchOff(deviceName, switchName, switchValue)` | An integrated AMD GPU is attached and its switch is off | `An AMD integrated GPU (<name>) is attached. APU use is off: set <switch>=<value> and restart to enable it.` |
| `vllmUnifiedSwitchHint(switchName, switchValue)` | The end of vLLM's "not enabled on unified memory" reasons | `Set <switch>=<value> to turn it on; it has not been measured on this hardware.` |
| `lmStudioEnableHint(switchName, switchValue)` | The end of LM Studio's "not enabled" reason | `Set <switch>=<value> and restart to enable it.` |
| `lmStudioStagedUpdate()` | LM Studio staged an update in its home, which would change the pinned bytes | `LM Studio has staged an update in its home; restart to rebuild LM Studio from its verified download.` |
| `profileRestartGpuRecovered()` | The GPU is usable again after the profile was decided without it | `GPU access is available now; restart to select it. Stopping and starting the model does not change the compute profile.` |
| `profileRestartChanged()` | The sentence that says what changes the compute profile | `After changing GPU access, machine or provider, restart to select a new compute profile.` |
| `restartToUseDevice(profile)` | A Vulkan profile no longer matches the device; the caller adds a leading space | `This engine started with the <profile> profile; restart it to use the current device or run on the CPU.` |
| `restartToRunOnCpu(profile)` | The GPU is gone under a committed GPU profile; the caller adds a leading space | `This engine started with the <profile> profile; restart it to run on the CPU.` |
| `restartForMemoryModel(current, selected)` | The GPU's memory model differs from the committed profile | `The GPU now reports <current> memory, but this engine started with the <selected> profile; restart it to use it.` |

### The limits (`defineLimits`)

A limit is anything that narrows the physical machine: a CPU quota, a memory limit, a GPU share. The engine implements none. It calls one hook at each place where a limit could matter, passes it what the engine holds there, and uses the answer as it is. A hook keeps its state in its own closure, one per engine instance: the engine keeps none for it. `defineLimits(overrides)` takes one function per hook (`vllmQualification` may also be null), fills the rest with the identity default, freezes the result and throws `LocalLlmError('invalid_limits', message, { key })` for an unknown key or a value of the wrong type. `limits` is an option of `createLocalLlmController` and `createController`, and of the lower-level readers, policies and `createRunners` that call hooks, so a limit reaches the CPU counts the runners use for their default threads as well as the admission policies.

A hook that throws propagates, except `guardSample`, whose throw the engine counts as "no reading", and `runnerEnvironment`, whose throw refuses the start with the thrown message. The identity default of every hook caps nothing, refuses nothing, decorates nothing, adds nothing to a runner's environment and watches nothing.

| Hook | Called by | Inputs | Returns | Identity default |
| --- | --- | --- | --- | --- |
| `capCores` | `physicalCoreCount` and `performanceCoreCount` in `controller/hardware.mjs`, at the end, only when a count was found | `(count, { fsApi })`: a safe positive integer and the reader the count used | The count to use | `count` |
| `observeSnapshot` | `readSnapshot`, once per snapshot, after the GPU and disk reads and before memory is read and cores are counted | `({ fsApi, env })`: `fsApi` is never undefined (the engine passes `fs`) | An observation the engine does not look at, or null | `null` |
| `decorateSnapshot` | `readSnapshot`, last | `({ snapshot, fsApi, env, observation })`: `snapshot` has the keys `at, gpu, memory, disk, cpus, cores` in that order; `observation` is the result of `observeSnapshot` for this snapshot | The snapshot to publish. A hook may append keys (a key assigned to an existing property keeps its place), replace `gpu` when it knows the GPU cannot be used, and attach non-enumerable facts for its other hooks. It is not called when a read between the two hooks throws | `snapshot` |
| `memoryPool` | `cpuPool` in `controller/profiles.mjs`, for the CPU, Venus and APU policies, Ollama on the CPU, and the controller's pool figures | `(memory, snapshot, { amount })`: `memory` has finite `totalBytes` and `availableBytes` (the engine handles unreadable memory itself); `amount` formats a byte count as the policies do | `null` (the physical figures stand) or `{ totalBytes, availableBytes, capped, warning }`. The engine adds `warning` to the policy's warnings only when `capped` is true and `warning` is set | `null` |
| `refuseProfile` | `admit` (stage `'admission'`, before any policy, every profile) and the APU policy (stage `'policy'`, after its device-memory check) | `({ stage, profile, snapshot, refuse })`: `refuse(status, constraint, reason)` builds the result in the shape of that call point | A refusal or null | `null` |
| `admit` | `admit` in `controller/admission.mjs`, at three stages in the order of the checks | `({ stage, profile, snapshot, env, result, ... })`; `result(status, reason, estimate, warnings, reasonCode)` is the admission result builder. Stage `'before-policy'`: after the runner and source checks and `refuseProfile`. Stage `'vulkan'`: in a Vulkan profile, after the runner is known to have a policy. Stage `'gpu'`: on a dedicated or unified GPU after the engine checked it is usable and matches the profile; also `selected` (`'dedicated'` or `'unified'`) and `gpu` (the GPU as read) | A result (refusal) or null; at stage `'gpu'` also `{ refusal }` or `{ share: { gpu } }`, where `gpu` is the view the policies then size against | `null` |
| `effective` | `admit`, at the Vulkan, unified and dedicated policy calls | `({ snapshot })`; `snapshot` may be undefined on the Vulkan branch | `{ memory }`: the memory view the policies size against | `{ memory: snapshot?.memory \|\| {} }` |
| `refuseResourceRequest` | `admitOllama` (`policy: 'ollama'`) and `admitVllm` (`policy: 'vllm'`) | `({ policy, gpu, estimate, refuse, gib, ... })`; Ollama also `marginBytes`; vLLM also `params`, `gpuMemoryUtilization`, `denominator`. `refuse(status, reason)` builds the policy's result with its own estimate and warnings; `gib` is its byte formatter | A refusal or null | `null` |
| `overview` | The controller's overview, when it builds the pool limits | `({ snapshot, env })`; `snapshot` may be null | An object, published as the `budget` key appended after the pool figures, or null (no key) | `null` |
| `runnerEnvironment` | The controller, when it builds a runner's environment | `({ env, profile, adapterEnvironment })`: `adapterEnvironment` is a fresh copy of what the adapter asked for | The object that replaces the adapter layer. The final environment is `PATH`, `HOME`, `LANG`, `CUDA_CACHE_PATH`, then this object, then (Vulkan) `MESA_SHADER_CACHE_DIR` | `adapterEnvironment` |
| `guardSample` | The memory guard: once when it starts, once per tick of the APU guard, and once per tick of a guard whose first sample was `watched` | `()` | A sample object or null. The first sample's `watched: true` makes a dedicated guard use its short interval and ask `guardBreach` for headroom on every tick | `null` |
| `guardBreach` | The memory guard | `(sample, context)`. `context.check` is `'watchable'` (APU guard, every tick, with `profile`) or `'headroom'` (with `memory`, the meminfo reading or null, and `available`, its available bytes; only for a watched guard) | `'watchable'`: null or `{ message }` (stops the runner with that message). `'headroom'`: null, `{ available }` (the figure to compare with the floor) or `{ unreadable: message }`. After two consecutive unreadable answers the engine stops the runner with the last message; any other answer resets the count | `null` |
| `vllmQualification` | `admit`, for a runner flagged `requiresShareQualification`, under a share | `({ ...runnerContext, runner, gpu })` where `runnerContext` is `{ runnerLockEntry }` and `gpu` is the GPU as read | `{ qualified, code, fix }`; when `qualified` is not true, `fix` is the refusal's reason and `code` its reason code | `null` (no function bound) |

**Order inside `readSnapshot`.** The GPU and disk reads, then `observeSnapshot`, then the memory read and the core count (which calls `capCores`), then `decorateSnapshot`. A hook that must record a fact even when the memory read throws takes it in `observeSnapshot`; the engine hands the result on to `decorateSnapshot` as `observation`. A throwing memory read rejects `readSnapshot` and `decorateSnapshot` is not called.

**Share qualification.** A runner declares `requiresShareQualification: true` when it must be proven for a GPU share before it runs under one (only vLLM does). When `admit` stage `'gpu'` returns `{ share }`, the engine asks `vllmQualification` for a flagged runner. With no function bound it refuses as `incompatible` with the reason code `share_unqualified` and the reason `<display name> cannot be used under a GPU share here: this engine has no qualification for it.`; with a function, an answer that is not `{ qualified: true }` refuses as `incompatible` with the answer's `fix` as the reason and its `code` as the reason code. A runner without the flag is not asked, and nothing is asked when no share applies.

**What a plug-in does not receive.** The engine holds no limit state: the established memory limit of a container, a share's parsed values and a guard's last sample belong to the plug-in's closure, one per engine instance.

### Other inputs

| Input | Behavior |
| --- | --- |
| `env` | Passed by reference. The switches `LOCAL_LLM_AMD_APU`, `LOCAL_LLM_VLLM_UNIFIED`, `LOCAL_LLM_LMSTUDIO`, `LOCAL_LLM_APU_GUARD_FLOOR_BYTES` and the `HF_TOKEN` token are engine reads at call time. The engine reads no path from the environment: every path comes from the host |
| `killImpl` | The signal function of a runner stop (`process.kill` by default). A test passes a recording fake. `startRunnerProcess` also takes `platform` |
| `startRunner` | Starts a runner process (`startRunnerProcess` by default). On a Unix-socket host (`host.listen`) its options carry `listen: { startId, socketPath }`, so a host can record the start before the process exists |
| `socketFetchImpl`, `startIdFactory` | The fetch of a Unix-socket start's readiness probes (`httpFetch` with `socketPath` by default; global `fetch` cannot reach a socket) and the generator of the 8-hex start id |
| `chatTarget()` | The ready runner's address. A TCP start answers `baseUrl: 'http://127.0.0.1:<port>'`, as before. A Unix-socket start answers `baseUrl: 'http://localhost'` and `socketPath`, which a caller passes to `httpFetch`; the base URL then only names the request path |
| Hardware reaper | `createHardwareQueryReaper()` keeps the queries a stop killed but that have not exited. One reaper belongs to one controller; `readGpu` and `readSnapshot` take it as `reaper` and the drain waits on it |
| Missing files | `loadSeedCatalog(file)`, `loadRunnerLock(file)` and `loadRunnerLocks({ image, agent })` refuse an undefined argument with `missing_file`; null means "none"; an `ENOENT` on a given path is an empty lock |
| `.tar.zst` | `installRunner` and a Run that must rebuild a runnable copy are refused with `zstd_unsupported` where `zlib.createZstdDecompress` is missing, before any state is written or any byte is fetched (DS004) |

### Rationale

#### Hooks instead of options

A limit touches admission, the snapshot, the guard and a runner's environment, and the results of those places must be reproducible byte for byte by the host that already had them. One hook per place, with the inputs the old code held there, lets a host move its logic out of the engine without changing a result, and lets the engine prove with a recording fake that each hook is called where it is documented.

#### Flat, whole-sentence message keys

A key is one function returning a whole sentence rather than a fragment, so a host's wording is a copy of its old text and not a recomposition. A fragment-based API would force every host to restate the engine's sentence structure.

#### No default paths

A default path that is relative to the engine's own files would let a host silently read the engine's files instead of its own. The engine therefore has no default seed catalog, no default runner lock and no default install directory: a host passes each one, and an omitted one is null or an error that names the field.
