---
title: DS000-engine-scope
summary: What the local-llm engine in lib/local-llm is, which hosts embed it, and what it deliberately leaves to them.
---

# DS000 Engine Scope

## Introduction

`lib/local-llm` is a Node.js library that runs open-weight language models, one deployment at a time, through on-demand runner processes on the machine it is started on. It owns the model lifecycle, the hardware snapshot, admission, the runner adapters, the runner installer and the memory guards. It is embedded by a **host**, a program that supplies the paths, wording and limits of its own surroundings (DS007) and that adds everything the engine leaves out. The first host is the `local-llm` agent of the local-llms repository, which serves the engine through a Ploinky agent. The pworker program is the second.

## Core Content

### What the engine does

| Area | Specification |
| --- | --- |
| Catalog, registry, downloads, the deployment state machine, restart behavior | DS002 |
| The hardware snapshot, admission, runner isolation, memory estimates | DS003 |
| Runner locks, the verified cache, runnable copies, the install check | DS004 |
| Hardware profiles: NVIDIA dedicated and unified memory, CPU, Vulkan (RADV, Venus, experimental APU), Apple Metal, their detection, admission and guards | DS005 |
| Coding conventions and layout of the engine subtree | DS006 |
| The host, messages and limits that the engine takes as input, and the public exports | DS007 |
| The standalone pworker host on Linux and macOS: ownership of a home, launch records, the reaper, Unix-socket runners, the host runner locks and the macOS profiles, readers and directory checks (not part of the engine; `lib/local-llm-host`) | DS008 |

The runners are llama.cpp, its CPU and Vulkan builds, ik_llama.cpp, Ollama, vLLM, ExLlamaV3 with TabbyAPI, and LM Studio. LM Studio is for internal use only and sits behind an operator switch. A runner is an adapter plus data: the controller has no per-runner branches.

### What the engine leaves to the host

| Left to the host | Why |
| --- | --- |
| The transport: tools, the control socket, the chat endpoint envelope, authentication and authorization | They depend on the program that serves the engine. `pworker/local-llm/chat` offers the pure helpers a responder needs |
| Seed catalog data and runner locks | The engine ships the schema and the validator, not the data. It has no default path for either (DS007) |
| Container paths, vendor wording and process environment variables that name a path | They are `host` and `messages` values |
| Every hardware limit: CPU quotas, memory limits, GPU shares and the code that parses them | They are `limits` hooks. The identity default applies none |
| User interfaces | The dashboard and the Settings page belong to the agent that shows them |

### The neutrality rule

No file below `lib/local-llm` may name a container runtime, a gateway, a path of a particular image or a limit mechanism. `test/local-llm/engine-neutral.test.mjs` strips comments, scans every `.mjs` file below `lib/local-llm`, fails when it finds no file or cannot read one, and proves it can fail by scanning a literal that holds a forbidden term. A value a host needs goes into `defineHost`, a sentence into `defineMessages`, a limit into `defineLimits`.

### Where the other specifications live

DS001 is the agent contract (manifest, tools, authorization, the chat endpoint, drain). It stays in the local-llms repository with the hardware-limits content that the hooks of DS007 carry out of the engine. The documents here carry the engine's own text from local-llms at `34f8f498`; their wording note says how the agent's names map to engine names.

### macOS and the standalone Metal scope

The engine also runs on macOS when its host gives it the macOS pieces: `darwinPlatform()` (DS007, the download binder of DS002), readers for the hardware snapshot, the memory and the memory pressure level, and a runner lock for the Mac (DS004). `lib/local-llm-host/darwin.mjs` and the `host-darwin` locks in `lib/local-llm-host/locks` hold the readers and the locks as host-side files that the engine never imports; they enter as the `readHardwareSnapshot`, `readMemory` and `readPressure` seams and as `imageRunnerLockFile`. A platform other than Linux and macOS is refused (`platformFor`), not guessed. `pworker serve` is the host that supplies the macOS pieces (DS008): it fixes the profile by architecture, checks the directories of the home, probes the Metal device through the installed runner, and recovers orphaned runners through `ps`.

| Host | Profile | Runners offered | Not offered |
| --- | --- | --- | --- |
| macOS on Apple Silicon (`darwin-arm64`) | `metal`, fixed by the host (DS005) | `llama.cpp`, the Metal build, installed from the Mac lock | ik_llama.cpp, Ollama, vLLM, ExLlamaV3 with TabbyAPI, LM Studio, the Vulkan, Venus and APU profiles, workspace-shared weights |
| macOS on Intel (`darwin-x64`) | `cpu`, fixed by the host; guarded by the kernel pressure level like `metal` (`pressureScale`, DS007) | `llama.cpp-cpu`, installed from the Mac lock | the same list. Not verified on hardware |

The Linux behavior is unchanged: `test/local-llm/agent-unchanged-metal.test.mjs` pins the eight runner ids, the default ports and the overview of the dedicated, unified, cpu, vulkan, venus and apu profiles, which stay deep-equal whether or not a catalog carries `metal` metadata. The Metal constants are provisional (DS005) and the inference smoke on a real model is run by the operator, so a Metal deployment is unit-tested and light-checked here, not certified. Pending: the Metal inference smoke on a real model (the operator's, at pressure level 1), a run of `darwin-x64` on hardware, and a measurement for the Metal constants.
