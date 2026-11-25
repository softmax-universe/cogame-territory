// The robust decide loop — the shared backbone for BOTH action decisions and
// message generation. Ask the model for one decision, parse its sampled JSON text, then `validate`
// (schema + legality). On a thrown rejection, re-prompt INCLUDING the reason and
// retry up to `maxAttempts`; on exhaustion fall back to `baseline()` so a flaky
// model never stalls a turn. Every attempt is reported via `recordAttempt` for
// the autopilot transcript surfaced in the UI.
//
// The catch here is legitimate control flow: catching a validation/parse
// rejection to re-prompt, catching a TRANSPORT failure (throttle, socket reset,
// 5xx, request timeout) to retry it the same way, falling back to baseline on
// exhaustion, and treating a terminal no-credentials transport error as an
// immediate baseline (re-prompting is futile — there is no model to reach). No
// failure escapes this function: the caller always gets a decision.
import type { ActAttempt, TextGeneration, GenerationPurpose } from "@cogweb/protocol";
import { isCredentialsUnavailable } from "./openrouter.js";
import type { OpenRouterLlmClient, LlmMessage } from "./openrouter.js";

/**
 * Parse a primary native JSON object reply, optionally wrapped in a single
 * fenced code block. Never scrapes braces out of surrounding prose, and rejects
 * ambiguous multi-block or competing-JSON replies so raw output is preserved.
 */
export function parseJsonAction(
  text: string,
): { kind: "parsed"; value: unknown } | { kind: "rejected"; reason: string } {
  const trimmed = text.trim();
  if (!trimmed) return { kind: "rejected", reason: "empty reply" };

  const parsePlainObject = (raw: string): unknown | null => {
    try {
      const value: unknown = JSON.parse(raw);
      return value !== null && typeof value === "object" && !Array.isArray(value) ? value : null;
    } catch {
      return null;
    }
  };

  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    const direct = parsePlainObject(trimmed);
    if (direct !== null) return { kind: "parsed", value: direct };
    return { kind: "rejected", reason: "invalid JSON object in reply" };
  }

  const fences = [...trimmed.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi)];
  if (fences.length > 1) {
    return { kind: "rejected", reason: "ambiguous reply: multiple fenced blocks" };
  }
  if (fences.length === 1) {
    const outside = trimmed.replace(fences[0]![0], "").trim();
    if (outside.includes("{") || outside.includes("}")) {
      return { kind: "rejected", reason: "ambiguous reply: JSON outside fenced block" };
    }
    const inner = fences[0]![1]!.trim();
    const parsed = parsePlainObject(inner);
    if (parsed !== null) return { kind: "parsed", value: parsed };
    return { kind: "rejected", reason: "invalid JSON object in fenced block" };
  }

  return { kind: "rejected", reason: "reply must be a JSON object or single fenced JSON block" };
}

export function extractJson(text: string): unknown {
  const parsed = parseJsonAction(text);
  if (parsed.kind === "rejected") throw new Error(parsed.reason);
  return parsed.value;
}

export interface RobustDecideOpts<Decision> {
  client: OpenRouterLlmClient;
  signal: AbortSignal;
  purpose: GenerationPurpose;
  /** System prompt: rules, strategy, output-format contract. */
  system: string;
  slot?: number;
  /** Render the user turn. Re-rendered each attempt with the prior rejection
   *  reason (null on the first try) so the retry re-states the error. */
  renderUser: (rejection: string | null) => string;
  /** Validate a candidate (schema + legality); returns the typed decision, or
   *  THROWS a human-readable reason suitable for re-prompting. */
  validate: (candidate: unknown) => Decision;
  /** An always-legal decision used when every attempt is exhausted. */
  baseline: () => Decision;
  /** Record one attempt (prompt, response, rejection-or-null) for the transcript. */
  recordAttempt: (attempt: ActAttempt) => void;
  /** Max model calls before falling back. Defaults to 3. */
  maxAttempts?: number;
  /** Mark a returned baseline as unsupervised fallback. */
  markFallback: () => void;
  model?: string;
}

/** Run the retry-then-fallback loop and return a validated decision. */
export async function robustDecide<Decision>(opts: RobustDecideOpts<Decision>): Promise<Decision> {
  const maxAttempts = Math.max(1, opts.maxAttempts ?? 3);
  let rejection: string | null = null;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    opts.signal.throwIfAborted();
    const generationId = globalThis.crypto.randomUUID();
    const prompt = opts.renderUser(rejection);
    const messages: LlmMessage[] = [{ role: "user", text: prompt }];
    let reply;
    let generation: TextGeneration | undefined;
    try {
      reply = await opts.client.complete({
        signal: opts.signal,
        purpose: opts.purpose,
        system: opts.system,
        messages,
        slot: opts.slot,
        model: opts.model,
        recordGeneration: (evidence) => {
          generation = evidence;
          opts.recordAttempt({
            generationId,
            purpose: opts.purpose,
            prompt,
            response: evidence.response,
            error: null,
            generation: evidence,
            platformCallId: evidence.platformCallId ?? null,
            providerRequestId: evidence.providerRequestId ?? null,
          });
        },
      });
    } catch (err) {
      // No credentials (offline cert): terminal and unrecoverable — retrying just
      // re-fails. Record it and play baseline now so the turn resolves instantly.
      if (isCredentialsUnavailable(err)) {
        opts.recordAttempt({
          generationId,
          purpose: opts.purpose,
          prompt,
          response: generation?.response ?? "",
          error: err instanceof Error ? err.message : String(err),
          generation,
          platformCallId: generation?.platformCallId ?? null,
          providerRequestId: generation?.providerRequestId ?? null,
        });
        opts.markFallback();
        return opts.baseline();
      }
      // ANY OTHER TRANSPORT FAILURE — throttle, socket reset, 5xx, request timeout
      // — is transient, so it is treated exactly like a rejected reply: record the
      // attempt and go round the loop once more; when the attempts are spent the
      // loop falls out to `baseline()` below. Rethrowing here (the old behaviour)
      // propagated out of the player process, so a single throttle cost the seat
      // the rest of the episode instead of degrading it to its scripted move.
      opts.recordAttempt({
        generationId,
        purpose: opts.purpose,
        prompt,
        response: generation?.response ?? "",
        error: err instanceof Error ? err.message : String(err),
        generation,
        platformCallId: generation?.platformCallId ?? null,
        providerRequestId: generation?.providerRequestId ?? null,
      });
      opts.signal.throwIfAborted();
      continue;
    }
    opts.signal.throwIfAborted();
    const response = reply.text;

    try {
      // Parse AND validate inside the same try: a parse failure (no JSON in the
      // reply) is a rejection to re-prompt exactly like a legality rejection.
      const candidate = extractJson(reply.text);
      const decision = opts.validate(candidate);
      opts.recordAttempt({
        generationId,
        purpose: opts.purpose,
        prompt,
        response,
        error: null,
        generation: reply.generation,
        parsedAction: decision,
        platformCallId: reply.platformCallId,
        providerRequestId: reply.generation.providerRequestId ?? null,
      });
      return decision;
    } catch (err) {
      rejection = err instanceof Error ? err.message : String(err);
      opts.recordAttempt({
        generationId,
        purpose: opts.purpose,
        prompt,
        response,
        error: rejection,
        generation: reply.generation,
        platformCallId: reply.platformCallId,
        providerRequestId: reply.generation.providerRequestId ?? null,
      });
    }
  }

  opts.markFallback();
  return opts.baseline();
}
