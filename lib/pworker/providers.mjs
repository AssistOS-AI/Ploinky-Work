import { resolveUpstream } from '../settings.mjs';

/** Refuses to overwrite a provider that is not a plain endpoint (a managed local model or the local-llm engine). Every save path goes through it. */
export function assertCustomProviderAllowed(name, prior = {}) {
  if (prior?.start) throw new Error('A managed local provider is configured in its start block, not as a custom endpoint');
  if (prior?.kind === 'local-llm') throw new Error(`${name} is the local-llm engine of this machine, not an endpoint; manage its models with \`pworker models\``);
}

export function providerSpec(name, endpoint, key, rpm = 60, prior = {}) {
  if (!/^[a-z][a-z0-9_-]*$/.test(name)) throw new Error('Invalid provider name');
  assertCustomProviderAllowed(name, prior);
  const url = new URL(endpoint);
  if (!['https:', 'http:'].includes(url.protocol)) throw new Error('An HTTP(S) endpoint is required');
  if (url.username || url.password || url.search || url.hash) throw new Error('The endpoint must not contain credentials, a query, or a fragment');
  if (/\/chat\/completions\/?$/.test(url.pathname)) url.pathname = url.pathname.replace(/\/chat\/completions\/?$/, '');
  const baseUrl = url.href.replace(/\/+$/, '');
  const versioned = /\/v\d+$/.test(url.pathname.replace(/\/+$/, ''));
  const sameEndpoint = String(prior.baseUrl ?? '').replace(/\/+$/, '') === baseUrl;
  const limit = Number(rpm);
  if (!Number.isInteger(limit) || limit < 1) throw new Error('Requests per minute must be a positive integer');
  const keyVar = `PWORKER_${name.toUpperCase().replace(/[^A-Z0-9]/g, '_')}_API_KEY`;
  return {
    ...prior,
    baseUrl,
    envFile: key ? `${name}.env` : null,
    keyVar: key ? keyVar : null,
    noKey: !key,
    modelsPath: sameEndpoint ? (prior.modelsPath ?? (versioned ? '/models' : '/v1/models')) : (versioned ? '/models' : '/v1/models'),
    formats: sameEndpoint ? (prior.formats ?? { openai: versioned ? '/chat/completions' : '/v1/chat/completions' }) : { openai: versioned ? '/chat/completions' : '/v1/chat/completions' },
    limits: { ...(prior.limits ?? {}), maxPerMinute: limit },
  };
}

export async function fetchModelCatalog(provider, { key = null, fetchImpl = fetch, timeoutMs = 12_000 } = {}) {
  const url = String(provider.baseUrl).replace(/\/+$/, '') + (provider.modelsPath ?? '/v1/models');
  const response = await fetchImpl(url, { headers: { accept: 'application/json', ...(key ? { authorization: `Bearer ${key}` } : {}) }, signal: AbortSignal.timeout(timeoutMs) });
  if (!response.ok) throw new Error(`Model list request failed (${response.status})`);
  const body = await response.json();
  const models = [...new Set((Array.isArray(body.data) ? body.data : []).map((item) => item?.id).filter((id) => typeof id === 'string' && id))].sort();
  if (!models.length) throw new Error('The provider returned no selectable models');
  return models;
}

const million = 1_000_000;
const number = (value) => Number.isFinite(Number(value)) && Number(value) >= 0 ? Number(value) : null;

/**
 * Normalizes prices advertised by OpenAI-compatible model catalogues to USD per
 * million text tokens. OpenRouter reports USD per token; xAI reports cents per
 * 100 million tokens. A provider may supply `modelPricing[modelId]` when its
 * catalogue does not publish prices.
 */
export function modelPrices(model, provider = {}) {
  const configured = provider.modelPricing?.[model?.id] ?? provider.modelPricing?.[model?.name] ?? {};
  const price = model?.pricing ?? {};
  const fromPerToken = (value) => { const n = number(value); return n == null ? null : n * million; };
  const fromXai = (value) => { const n = number(value); return n == null ? null : n / 10_000; };
  const input = number(configured.inputUsdPerM ?? configured.input)
    ?? fromPerToken(price.prompt ?? price.input ?? price.input_per_token)
    ?? fromXai(model?.prompt_text_token_price);
  const output = number(configured.outputUsdPerM ?? configured.output)
    ?? fromPerToken(price.completion ?? price.output ?? price.output_per_token)
    ?? fromXai(model?.completion_text_token_price);
  const cachedInput = number(configured.cachedInputUsdPerM ?? configured.cachedInput)
    ?? fromPerToken(price.cache_read ?? price.cached_input)
    ?? fromXai(model?.cached_prompt_text_token_price);
  return { input, output, cachedInput };
}

function priceScore(model) {
  const { input, output } = model.prices;
  return input == null && output == null ? Infinity : (input ?? output ?? 0) + (output ?? input ?? 0);
}

/** Converts an upstream model list into stable, display-ready records. */
export function normalizeModelCatalog(providerName, provider, data = []) {
  const seen = new Set();
  return data.flatMap((raw, index) => {
    const id = raw?.id;
    if (typeof id !== 'string' || !id || seen.has(id)) return [];
    seen.add(id);
    return [{
      id,
      name: typeof raw.name === 'string' && raw.name ? raw.name : id,
      provider: providerName,
      prices: modelPrices(raw, provider),
      created: number(raw.created),
      contextLength: number(provider.modelLimits?.[id]?.contextTokens ?? raw.context_length ?? raw.context_window),
      maxOutputTokens: number(provider.modelLimits?.[id]?.maxOutputTokens ?? raw.max_output_tokens ?? raw.top_provider?.max_completion_tokens),
      modelLimits: provider.modelLimits?.[id] ?? null,
      quotaMultiplier: number(raw.quota_multiplier),
      billing: raw.x_billing === 'credit' ? 'credit' : 'plan_or_unknown',
      sourceIndex: index,
      raw,
    }];
  });
}

/** True when the provider catalog says this model produces text. Unknown catalog
 * formats remain selectable so ordinary OpenAI-compatible endpoints keep working. */
export function isTextOutputModel(model) {
  const output = model?.raw?.output_modalities ?? model?.raw?.architecture?.output_modalities;
  return !Array.isArray(output) || output.includes('text');
}

/** A model that previously returned an insufficient-credit response is never a
 * safe automatic choice. It remains visible in raw provider data for auditing. */
export function isModelEligibleForTier(model) {
  return model?.billing !== 'credit' && isTextOutputModel(model);
}

export function modelExclusionReason(model) {
  if (model?.billing === 'credit') return 'requires a credit balance';
  if (!isTextOutputModel(model)) return 'does not produce text';
  return null;
}

export function sortModelCatalog(models, order = 'recommended') {
  const copy = [...models];
  if (order === 'lowest-price') return copy.sort((a, b) => priceScore(a) - priceScore(b) || a.sourceIndex - b.sourceIndex || a.id.localeCompare(b.id));
  if (order === 'newest') return copy.sort((a, b) => (b.created ?? -Infinity) - (a.created ?? -Infinity) || a.sourceIndex - b.sourceIndex || a.id.localeCompare(b.id));
  return copy.sort((a, b) => a.sourceIndex - b.sourceIndex || a.id.localeCompare(b.id));
}

export function formatUsdPerM(value) {
  if (value == null) return 'price unavailable';
  if (value < 0.01) return `$${value.toFixed(4)}/M`;
  if (value < 1) return `$${value.toFixed(3)}/M`;
  return `$${value.toFixed(2)}/M`;
}

export function formatModelLabel(model) {
  const { input, output } = model.prices;
  const price = input == null && output == null ? 'price unavailable' : `in ${formatUsdPerM(input)} · out ${formatUsdPerM(output)}`;
  const plan = model.quotaMultiplier == null ? '' : ` · plan ${model.quotaMultiplier} credit${model.quotaMultiplier === 1 ? '' : 's'}/request`;
  const unavailable = modelExclusionReason(model);
  return `${model.provider} · ${model.name} · ${price}${plan}${unavailable ? ` · ${unavailable}` : ''}`;
}

export function connectedProviders(config, localStatus = {}, env = process.env) {
  return Object.entries(config.providers ?? {}).filter(([name]) => !name.startsWith('_')).flatMap(([name, provider]) => {
    const resolved = resolveUpstream(name, provider, env);
    if (provider.start || provider.kind === 'local-llm') return localStatus[name]?.running ? [name] : [];
    return resolved.key || provider.noKey ? [name] : [];
  });
}
