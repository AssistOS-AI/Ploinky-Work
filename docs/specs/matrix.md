# Specification Matrix

| Name | Description |
| --- | --- |
| [DS000-engine-scope](DS000-engine-scope.md) | What the local-llm engine is, which hosts embed it, and what it leaves to them. |
| DS001 (agent contract) | Not carried. The agent contract (manifest, tools, authorization, the chat endpoint, drain) stays in the local-llms repository. |
| [DS002-model-lifecycle](DS002-model-lifecycle.md) | Catalog, registry, on-demand download, the deployment state machine, and restart behaviour. |
| [DS003-gpu-and-resources](DS003-gpu-and-resources.md) | The hardware snapshot, admission, runner isolation and memory estimates of the engine. |
| [DS004-on-demand-runners](DS004-on-demand-runners.md) | Runners installed on demand from the image's lock or the host's own lock, a verified cache and a runnable copy; the zstd requirement and the install check. |
| [DS005-hardware-profiles](DS005-hardware-profiles.md) | NVIDIA dedicated and unified memory, CPU fallback, discrete AMD Vulkan, experimental shared-memory AMD APU and Apple Venus profiles, their detection, admission and memory guards. |
| [DS006-engine-coding-style](DS006-engine-coding-style.md) | Language, layout, conventions and tests of the engine subtree. |
| [DS007-host-limits-and-messages](DS007-host-limits-and-messages.md) | The host, messages and limits a host gives the engine, every field, key and hook, and the public exports. |
| [DS008-standalone-host](DS008-standalone-host.md) | The local-llm host of a pworker home: layout, ownership, launch records and the reaper, Unix-socket runners, the host runner registry and locks, the gate and admin routes of the local-llm provider, the `pworker models` commands, and the host API. |
