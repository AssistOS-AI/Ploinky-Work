import { DEFAULT_MESSAGES } from '../messages.mjs';

export function plainDiagnosticText(value, limit = 400) {
    return String(value || '').replace(/[\x00-\x1f\x7f\u202a-\u202e\u2066-\u2069]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, limit);
}

export function computeDiagnostic(code, { backend = null, driver = null, source = 'agent-query', message } = {}) {
    return { code, backend, driver, source, message: plainDiagnosticText(message) };
}

export function cpuFallbackExplanation(decision, recovery = null, messages = DEFAULT_MESSAGES) {
    const diagnostic = decision?.diagnostic;
    const name = diagnostic?.driver === 'venus' ? 'Apple Venus' : (diagnostic?.backend || decision?.gpuBackend) === 'vulkan' ? 'Vulkan GPU' : 'GPU';
    const leads = { absent: `no usable ${name} was observed`, unusable: `the ${name} cannot be used`,
        mismatch: "this image's CUDA runners were not built for this GPU", 'unreadable-timeout': `the ${name} could not be read for 60 s` };
    const reason = plainDiagnosticText(diagnostic?.message || decision?.reason, 300).replace(/\.+$/, '');
    const summary = `Runs on the CPU: ${leads[decision?.cause] || 'no GPU is used'}${reason ? `. ${reason}` : ''}.`;
    return { summary, restartHint: recovery?.gpuRecovered ? messages.profileRestartGpuRecovered() : messages.profileRestartChanged() };
}
