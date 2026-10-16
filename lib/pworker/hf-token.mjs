// `pworker models hf-token`: stores the Hugging Face token the local-llm engine downloads gated or rate-limited weights with. The
// token lives in <home>/keys/huggingface.env as one line, `HF_TOKEN="..."`, mode 0600, and is never printed, logged or echoed.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

export const hfTokenFile = (home) => path.join(home, 'keys', 'huggingface.env');
const TOKEN_RE = /^[A-Za-z0-9._~+/=-]{1,512}$/;
const MAX_INPUT = 4096;

/** The token when it is a single printable-ASCII word without quotes, spaces or shell characters; otherwise a reason (never the token). */
export function checkToken(token) {
  if (typeof token !== 'string' || !token) return { ok: false, reason: 'the token is empty' };
  if (!TOKEN_RE.test(token)) return { ok: false, reason: 'the token must be 1-512 letters, digits and . _ ~ + / = - (no spaces, quotes or other characters)' };
  return { ok: true };
}

/**
 * Writes the token file: a temporary file in the same folder created 0600, synced, then renamed over the old one, so a replacement
 * never keeps the old file's mode and a reader never sees a half-written token. The keys folder is made 0700 when it is new.
 * Returns the path.
 */
export function writeHfToken(home, token, { fsApi = fs } = {}) {
  const verdict = checkToken(token);
  if (!verdict.ok) throw new Error(`Not stored: ${verdict.reason}.`);
  const file = hfTokenFile(home);
  const dir = path.dirname(file);
  fsApi.mkdirSync(dir, { recursive: true, mode: 0o700 });
  const temp = path.join(dir, `.huggingface.env.${process.pid}.${crypto.randomBytes(4).toString('hex')}.tmp`);
  const fd = fsApi.openSync(temp, 'wx', 0o600);
  try {
    fsApi.writeSync(fd, `HF_TOKEN=${JSON.stringify(token)}\n`);
    fsApi.fchmodSync(fd, 0o600);
    fsApi.fsyncSync(fd);
  } catch (error) {
    fsApi.closeSync(fd);
    try { fsApi.unlinkSync(temp); } catch { /* best effort */ }
    throw error;
  }
  fsApi.closeSync(fd);
  try {
    fsApi.renameSync(temp, file);
  } catch (error) {
    try { fsApi.unlinkSync(temp); } catch { /* best effort */ }
    throw error;
  }
  try {
    const dirFd = fsApi.openSync(dir, 'r');
    try { fsApi.fsyncSync(dirFd); } finally { fsApi.closeSync(dirFd); }
  } catch { /* a directory that cannot be synced still holds the renamed file */ }
  return file;
}

/**
 * The token from `input`: on a terminal, typed with no echo (Enter confirms, Ctrl-C cancels and gives null); otherwise the first
 * line of the stream, so a script can pipe it. The prompt is written to `output` only when a terminal is attached.
 */
export async function readTokenInput({ input, output }) {
  if (input.isTTY && typeof input.setRawMode === 'function') {
    output.write('Hugging Face token (input is hidden): ');
    return new Promise((resolve) => {
      let value = '';
      const finish = (answer) => {
        input.off('data', onData);
        try { input.setRawMode(false); } catch { /* closed */ }
        input.pause();
        output.write('\n');
        resolve(answer);
      };
      const onData = (chunk) => {
        for (const char of String(chunk)) {
          if (char === '\r' || char === '\n') return finish(value);
          if (char === '\u0003' || char === '\u0004') return finish(null);
          if (char === '\u007f' || char === '\b') value = value.slice(0, -1);
          else if (char >= ' ') value += char;
        }
      };
      input.setRawMode(true);
      input.setEncoding('utf8');
      input.resume();
      input.on('data', onData);
    });
  }
  let text = '';
  for await (const chunk of input) {
    text += chunk;
    if (text.length > MAX_INPUT || text.includes('\n')) break;
  }
  return text.split(/\r?\n/)[0].trim();
}
