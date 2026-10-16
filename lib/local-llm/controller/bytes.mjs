// Byte-count helpers the controller shares.

/** A known byte count: a safe integer >= 0. Zero is known. */
export function knownByte(value) {
    return Number.isSafeInteger(value) && value >= 0;
}
