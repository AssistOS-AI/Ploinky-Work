import { LOAD_MODES, createLlamaServerRunner, hostOptions } from './llamaServer.mjs';

// b10105 added --load-mode; b10875 removed --no-mmap, --mmap, --mlock and
// --direct-io. `auto` passes no flag: the server then avoids mmap on an
// integrated GPU by itself.
function loadArgs(loadMode) {
    return loadMode && loadMode !== 'auto' ? ['--load-mode', loadMode] : [];
}

function parseVersion(output) {
    const match = /\bbuild (\d+)\b/.exec(output);
    return match ? `b${match[1]}` : null;
}

const dialect = Object.freeze({
    // -lv 4 prints the device, offload and CUDA buffer lines the status reads.
    quietArgs: Object.freeze(['--no-webui', '-lv', '4']),
    unifiedKv: true,
    loadModes: LOAD_MODES,
    loadArgs,
    unified: true,
    cpu: true,
    jinja: (model) => Boolean(model?.requiresJinja),
    parseVersion,
});

/**
 * The three llama.cpp runners for a host. A runner whose executable the host does not give
 * (`host.executables['llama.cpp']`, and for the Vulkan build `host.executables['llama.cpp-vulkan']` or
 * `host.vulkan.serverPath`) runs from the runnable copy of an install from a runner lock.
 */
export function createLlamaCppRunners({ host, limits, ...seams }) {
    const common = { ...hostOptions(host, limits), ...seams };
    const llamaCpp = createLlamaServerRunner({
        id: 'llama.cpp',
        displayName: 'llama.cpp',
        executable: host.executables['llama.cpp'] ?? null,
        runnable: 'llama-server',
        pinnedVersion: 'b11159',
        port: 18080,
        dialect,
        ...common,
    });

    // ggml-org's own CPU build of a newer release (its ubuntu-arm64 and ubuntu-x64 assets), installed on
    // demand from the agent's lock (DS004) and run from its runnable copy. It is for the cpu profile
    // alone: no GPU policy, so a GPU host neither offers nor runs it. The tag is the one both locks pin.
    const cpu = createLlamaServerRunner({
        id: 'llama.cpp-cpu',
        displayName: 'llama.cpp (CPU build)',
        executable: null,
        runnable: 'llama-server',
        pinnedVersion: 'b11295',
        port: 18085,
        profiles: ['cpu'],
        dialect: Object.freeze({ ...dialect, unified: false, cpu: true }),
        ...common,
    });

    const vulkan = createLlamaServerRunner({
        id: 'llama.cpp-vulkan', displayName: 'llama.cpp (Vulkan build)',
        executable: host.executables['llama.cpp-vulkan'] ?? host.vulkan?.serverPath ?? null,
        runnable: 'llama-server', pinnedVersion: 'b11159', port: 18086,
        profiles: ['vulkan', 'venus', 'apu'], backend: 'vulkan',
        dialect: Object.freeze({ ...dialect, unified: false, cpu: false }),
        ...common,
    });
    return { llamaCpp, cpu, vulkan };
}
