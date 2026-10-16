// What the live smoke script concludes about the GPU. A machine that has an NVIDIA driver or device but whose `nvidia-smi` does not
// answer is broken, not GPU-less: running on the CPU there would pass without testing the CUDA runner.
import fs from 'node:fs';

export const NVIDIA_PATHS = Object.freeze(['/proc/driver/nvidia', '/dev/nvidia0']);

/** Whether an NVIDIA driver or device node exists on this machine. */
export const nvidiaPresent = (exists = fs.existsSync) => NVIDIA_PATHS.some((candidate) => exists(candidate));

/**
 * `{ gpuPresent, problem }`: the GPU is present when nvidia-smi answered; with `auto` and an NVIDIA device that did not answer,
 * `problem` says why the run must fail rather than fall back to the CPU.
 */
export function detectGpu({ smiAnswered, setting = 'auto', exists = fs.existsSync } = {}) {
    if (smiAnswered) return { gpuPresent: true, problem: null };
    if (setting === 'auto' && nvidiaPresent(exists)) {
        return { gpuPresent: false, problem: `an NVIDIA driver or device is present (${NVIDIA_PATHS.filter((candidate) => exists(candidate)).join(', ')}) but nvidia-smi did not answer; refusing to fall back to the CPU. Fix the driver, or pass --expect-gpu no to run the CPU build on purpose.` };
    }
    return { gpuPresent: false, problem: null };
}
