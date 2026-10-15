// Excerpts of two fixtures that stay in local-llms (tests/overview-cpu-scenarios.mjs and tests/overview-scenarios.mjs at 34f8f498):
// VULKAN_KEYS, cpuFs and IMAGE_LOCK, byte-identical, with the SHA constant IMAGE_LOCK uses.
export const VULKAN_KEYS = Object.freeze({
    vulkan_contract: '1', vulkan_llama_cpp: 'b11159',
    vulkan_llama_cpp_server: '/opt/llama.cpp-vulkan/llama-server', vulkan_mesa: '25.3.6+libkrun.3',
    vulkan_icd_radv: '/opt/local-llm/vulkan/icd.d/radeon_icd.json',
    vulkan_icd_venus: '/opt/local-llm/vulkan/icd.d/virtio_icd.json', vulkaninfo: '/usr/bin/vulkaninfo',
});

export function cpuFs() {
    return {
        readFileSync(file) {
            if (file === '/proc/meminfo') return 'MemTotal: 6291456 kB\nMemAvailable: 5242880 kB\nSwapFree: 0 kB\nMemFree: 4194304 kB\nCached: 1048576 kB\n';
            if (file === '/proc/cpuinfo') return 'processor: 0\ncore id: 0\n';
            throw Object.assign(new Error('absent'), { code: 'ENOENT' });
        },
        readdirSync() { throw Object.assign(new Error('absent'), { code: 'ENOENT' }); },
    };
}

const SHA = 'a'.repeat(64);

// The image's runner lock, reduced: one python and one proprietary archive runner, as the amd64 image has.
export const IMAGE_LOCK = {
    schema: 'local-llm.runners-lock/v1',
    runners: {
        vllm: {
            version: '0.30.0', kind: 'python',
            licence: { name: 'Apache-2.0', url: 'https://github.com/vllm-project/vllm/blob/v0.30.0/LICENSE' },
            files: [{ name: 'vllm-0.30.0-py3-none-any.whl', url: 'https://files.pythonhosted.org/packages/vllm-0.30.0-py3-none-any.whl', size: 1000, sha256: SHA }],
        },
        lmstudio: {
            version: '0.0.25-1', kind: 'archive',
            licence: { name: 'LM Studio Terms', url: 'https://lmstudio.ai/app-terms', requiresAcceptance: true, proprietary: true },
            files: [{ name: 'llmster.tar.gz', url: 'https://llmster.lmstudio.ai/download/llmster.tar.gz', size: 2000, sha256: 'b'.repeat(64), strip: 0 }],
        },
    },
};
