// Cleanup requests to ChatGPT through the account the Lunori plugin is signed in to.
// Usage: node chatgpt.mjs <lunori host dir>. Runs until stdin closes: one JSON request per line
// ({"system", "user", "model"?, "effort"?}; {"warmup": true} only resolves the model), one JSON
// reply per line ({"text"} or {"error"}). The model choice is cached: resolving it takes ~1.3 s.
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';

const host = process.argv[2];
const { resolveModel, access } = await import(pathToFileURL(host + '/account.mjs'));
const { consumeResponse } = await import(pathToFileURL(host + '/translator.mjs'));

// Lunori's catalog lists only gpt-5.6-luna and swaps other names for it, but the account also
// serves gpt-6-luna, so the catalog only supplies the account profile and the model is ours
const MODEL = 'gpt-6-luna';
const resolved = new Map();
async function pick(model, effort) {
  const key = `${model || ''}/${effort}`;
  if (!resolved.has(key)) {
    const selected = await resolveModel(undefined, 'standard', effort);
    resolved.set(key, { ...selected, model: model || MODEL, effort });
  }
  return resolved.get(key);
}

async function run(req) {
  const selected = await pick(req.model, req.effort || 'low');
  if (req.warmup) return { text: '', model: selected.model };
  const { token } = await access(selected.profileId);
  const response = await fetch('https://api.openai.com/v1/responses', {
    method: 'POST', redirect: 'error', signal: AbortSignal.timeout(20000),
    headers: { Authorization: 'Bearer ' + token, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: selected.model, instructions: req.system, input: [{ role: 'user', content: req.user }],
      reasoning: { effort: selected.effort }, store: false, stream: true,
    }),
  });
  return { text: (await consumeResponse(response)).text, model: selected.model };
}

for await (const line of createInterface({ input: process.stdin })) {
  if (!line.trim()) continue;
  let reply;
  try { reply = await run(JSON.parse(line)); } catch (error) {
    resolved.clear();
    reply = { error: String(error?.message || error) };
  }
  process.stdout.write(JSON.stringify(reply) + '\n');
}
