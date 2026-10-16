---
title: DS006-engine-coding-style
summary: Language, layout, conventions and tests of the engine subtree lib/local-llm; the moved code keeps its own style inside the pworker repository.
---

# DS006 Engine Coding Style

## Introduction

This specification is the coding-style authority for `lib/local-llm` and `test/local-llm`. The engine was written in the local-llms repository under a style of its own, and it keeps that style here without reformatting, so the history of a moved file stays readable and a change in either repository can be compared with the other. The rest of the pworker repository keeps the style of the files around it. `AGENTS.md` points here.

## Core Content

### Language and runtime

| Rule | Detail |
| --- | --- |
| Runtime | Node.js 22.13 or newer, ES modules (`.mjs`). The package declares `engines.node >= 22.13`. The engine was written for Node 24 and its tests have run on Node 25.8 only. A search of `lib/local-llm` for the built-ins added after 22.13 that are likely to appear (`Promise.try`, `RegExp.escape`, `Error.isError`, `Iterator` helpers, `fs.glob`, `URLPattern`, `import.meta.main`, `Float16Array`, set methods, `Symbol.dispose`) found none; the one dependence on a later version is zlib's zstd, added in Node 22.15 and checked at run time (DS004). The search is not a proof for the whole interface of Node |
| Dependencies | None. The package declares none, so the engine runs on Node alone |
| Other languages | No Python and no shell scripts in the engine. Runners that are Python programs are installed on demand from a lock (DS004) |
| Formatting | Four-space indentation, single quotes and semicolons, as the moved files have. New files in the subtree follow the file next to them |

### Layout

```
lib/local-llm/
  index.mjs            public entry: createLocalLlmController and the re-exports (DS007)
  chat.mjs             pure helpers of a chat responder
  host.mjs             defineHost, linuxPlatform
  darwinPlatform.mjs   darwinPlatform, platformFor (the macOS strategies and the choice of platform)
  limits.mjs           the hook contract and its identity default
  messages.mjs         message keys and neutral defaults
  errors.mjs           LocalLlmError, serializeError
  promptObservations.mjs, drainBudget.mjs
  catalog/schema.json  the catalog schema
  controller/          state, admission, profiles, hardware, downloads, installer, locks, runner processes
  runners/             one adapter per runner, and the table built from a host
  tools/runnerInstallCheck.mjs   the install check, also runnable as a script
test/local-llm/        the engine tests and their fixtures
test/local-llm-darwin/ tests of the macOS readers and locks of a standalone host (lib/local-llm-darwin)
test/local-llm-host/   tests of a standalone host
docs/specs/            these specifications
```

### Conventions

| Rule | Detail |
| --- | --- |
| Validate at the boundary and fail closed | Inputs have strict schemas; runner parameters are validated against the runner's schema for the hardware profile; identifiers use restrictive patterns (ids `[a-z0-9][a-z0-9._-]{1,63}`, request ids `[A-Za-z0-9_-]{8,128}`), which keeps path traversal and injection out of argv, file paths and registry entries. A value that cannot be read (a GPU's memory, `/proc/meminfo`) means "unknown" and refuses, never a default. |
| No shell between input and a process | Runners and helpers are spawned with an argv array, never through a shell, and get a minimal environment (DS003). |
| Errors carry a code | Engine errors are `LocalLlmError` with a stable `code` (for example `admission_incompatible`, `runner_unavailable`, `zstd_unsupported`, `invalid_host`) and a message that says what to do. |
| Nothing unpinned at run time | Downloads are pinned by commit, size and sha256 (models, DS002) or by URL, size and sha256 (runners, DS004). |
| No per-runner branches | The controller has none: a runner is an adapter plus data |
| Nothing host-specific in the engine | A path, a sentence or a limit that depends on the host is an input (DS007), and the neutrality test enforces it (DS000). |
| No hidden default paths | The engine never reads a file relative to its own location (DS007). |
| English | All documentation, specifications and comments are in English. Comments say why, not what |

### JSON

The catalog and the locks are JSON, never YAML, formatted with 2-space indentation. Catalog entries conform to `lib/local-llm/catalog/schema.json` and to `validateModel`.

### Tests

Tests use `node:test` and `node:assert/strict` with no external framework, one file per area, and need no GPU, container runtime or network: hardware, downloads, runners, signals and time are injected. They run with `npm test`, which names the groups `test/*.test.mjs`, `test/local-llm/*.test.mjs`, `test/local-llm-darwin/*.test.mjs` and `test/local-llm-host/*.test.mjs` explicitly so no group is skipped silently. Tests that exercise Linux-only mechanisms (`/proc/self/fd`, abstract sockets, `/proc/<pid>/oom_score_adj`) skip with a stated reason on other platforms; nothing else skips. The macOS tests inject their platform, their `exec` and their captured tool output, so they run on Linux as well as on macOS and never skip. A test that passes a hook or a platform stand-in names what it stands in for. Characterization tests pin today's exact behavior and change only with a deliberate change, which the commit names.

### File size

Prefer splitting a module that grows past about 800 lines by concern. `controller/deployments.mjs` is the known exception: it is the controller's single command queue.

### Rationale

#### Why the moved style is kept

Reformatting a moved file would hide the real differences between the two repositories in one large diff and make a later comparison of the engine with its origin impossible. The cost is two styles in one repository, kept apart by the directory.
