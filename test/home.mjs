// Imported first by every test file: the tests never read or write the real Pworker home (~/.pworker), which may hold real API keys.
// PWORKER_HOME points at a fresh temporary folder and provider keys of the environment are removed, so no test can reach a real provider. HF_TOKEN is removed too, so no test can send a real Hugging Face token anywhere.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

if (!process.env.PWORKER_HOME || !path.resolve(process.env.PWORKER_HOME).startsWith(fs.realpathSync(os.tmpdir()))) {
  process.env.PWORKER_HOME = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'pworker-test-home-'));
}
for (const key of Object.keys(process.env)) if (/_API_KEY$|^PWORKER_TOKEN$|^PWORKER_URL$|^PWORKER_CONFIG$|_BASE_URL$|^HF_TOKEN$/.test(key)) delete process.env[key];
