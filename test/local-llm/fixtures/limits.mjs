// Test-local limits for the moved engine tests. The engine knows no limit; a test that needs
// the memory pool capped says so through this fake, which reads a `cgroupMemory`-shaped
// {maxBytes, currentBytes} fact from the snapshot the test built.
import { defineLimits } from '../../../lib/local-llm/limits.mjs';

/** Caps the pool at the snapshot's `cgroupMemory.maxBytes`, less its `currentBytes`, and words the cap. */
export const MEMORY_CAP_LIMITS = defineLimits({
    memoryPool(memory, snapshot, { amount }) {
        const limit = snapshot?.cgroupMemory?.maxBytes;
        if (!Number.isSafeInteger(limit) || limit < 0) return null;
        const totalBytes = Math.min(memory.totalBytes, limit);
        let availableBytes = Math.min(memory.availableBytes, totalBytes);
        if (Number.isFinite(snapshot.cgroupMemory.currentBytes)) {
            availableBytes = Math.min(availableBytes, Math.max(0, limit - snapshot.cgroupMemory.currentBytes));
        }
        return {
            totalBytes, availableBytes, capped: totalBytes < memory.totalBytes || availableBytes < memory.availableBytes,
            warning: `A container memory limit of ${amount(limit)} applies.`,
        };
    },
});
