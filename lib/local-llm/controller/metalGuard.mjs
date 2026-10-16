// The memory guard's verdict for the metal profile (DS005), as a pure function of one sample: the controller samples
// every METAL.guardSampleMs, passes the running count of breaches, and stops the runner's process group when the
// verdict says so. The kernel's pressure level is 1 (normal), 2 (warning) or 4 (critical).
//
//   level 4                                         stop at once
//   level 2 (or unknown) with available < floor    a breach; the count of breaches in a row reaches METAL.guardBreaches: stop
//   anything else                                   the count starts again from zero
//
// A pressure level that cannot be read counts as elevated, so a runner that cannot be watched is not trusted. The
// available memory is read by the controller before this is called, and an unreadable one stops the runner there.

import { METAL, PRESSURE_CRITICAL, PRESSURE_NORMAL } from './profiles.mjs';

const GIB = 1024 * 1024 * 1024;

const gib = (bytes) => `${(bytes / GIB).toFixed(1)} GiB`;

/**
 * @param {{ pressureLevel: number|null, available: number, floorBytes: number, breaches?: number }} sample
 * @returns {{ breaches: number, stop: string|null }} the new count of breaches in a row, and the deployment's error when the runner must stop
 */
export function metalGuardVerdict({ pressureLevel, available, floorBytes, breaches = 0 }) {
    if (pressureLevel === PRESSURE_CRITICAL) {
        return { breaches: breaches + 1, stop: `stopped: macOS memory pressure is critical (level ${PRESSURE_CRITICAL}) with ${gib(available)} available` };
    }
    const elevated = pressureLevel !== PRESSURE_NORMAL;
    if (!(elevated && available < floorBytes)) return { breaches: 0, stop: null };
    const count = breaches + 1;
    if (count < METAL.guardBreaches) return { breaches: count, stop: null };
    const level = Number.isFinite(pressureLevel) ? `level ${pressureLevel}` : 'level unknown';
    return { breaches: count, stop: `stopped: macOS memory pressure (${level}) for ${count} samples in a row with ${gib(available)} available, below the ${gib(floorBytes)} floor` };
}
