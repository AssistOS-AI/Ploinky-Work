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
| Hardware profiles: NVIDIA dedicated and unified memory, CPU, Vulkan (RADV, Venus, experimental APU), their detection, admission and guards | DS005 |
| Coding conventions and layout of the engine subtree | DS006 |
| The host, messages and limits that the engine takes as input, and the public exports | DS007 |
| The standalone pworker host: ownership of a home, launch records, the reaper, Unix-socket runners and the host runner locks (not part of the engine; `lib/local-llm-host`) | DS008 |

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
