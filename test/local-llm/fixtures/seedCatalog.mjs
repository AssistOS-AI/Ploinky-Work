// The seed catalog the engine tests load. The engine has no default catalog path, so every test passes this file explicitly.
export const SEED_CATALOG_FILE = new URL('./models.json', import.meta.url).pathname;
