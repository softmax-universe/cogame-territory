import { realpathSync } from "node:fs";
// The ONE player entrypoint, shipped in the SAME image as the game and selected
// by env — `dist-server/game/player.js`, shimmed as `/bin/territory-player`:
//
//   PLAYER_SCRIPTED=homesteader | raider              that explicit scripted baseline
//   COWORLD_LLM_ENABLED=true / COWORLD_LLM_ENDPOINT /
//   COWORLD_LLM_MODEL / PLAYER_PROMPT / OPENROUTER_API_KEY   LLM policy (fails closed, no fallback)
//   none set                                          homesteader
//
// Native Coworld model selection uses canonical OpenRouter IDs.

import { argv, env } from "node:process";
import { fileURLToPath } from "node:url";

import { OpenRouterLlmClient, parseJsonAction } from "@cogweb/llm";
import type { TextGeneration } from "@cogweb/protocol";
import { runCoworldPlayer } from "@cogweb/coworld";
import type { PlayerDecideContext } from "@cogweb/coworld";

import { SubmissionSchema } from "../shared/engine/orders.js";
import { territoryModule } from "./game.js";
import type { TerritoryDecision, TerritorySeamState, TerritoryView } from "./game.js";
import type { TerritoryObservation } from "./redact.js";
import { renderPlayerMessages } from "./prompt.js";
import { scriptedDecide } from "./scripted.js";
import { promptSelectionSchema } from "./prompt-selection.js";

type Ctx = PlayerDecideContext<TerritorySeamState, TerritoryDecision, TerritoryView>;

/** Haiku 4.5. `maxTokens: 900` — 400 truncates ("cut off at max_tokens"). */
const DEFAULT_MODEL = "anthropic/claude-haiku-4.5";
const MAX_TOKENS = 900;

/** The LLM policy: one direct native model call per observation.
 *  Never substitutes a scripted or empty-hold action on backend, invalid-model,
 *  parse, ambiguity, or schema failure — preserves raw output and throws so the
 *  host records a terminal failure instead of default gameplay scores. */
export function makeLlmDecide(doctrine: string): (ctx: Ctx) => Promise<TerritoryDecision> {
  const model = env.COWORLD_LLM_MODEL?.trim() || DEFAULT_MODEL;
  const client = new OpenRouterLlmClient({
    model,
    maxTokens: MAX_TOKENS,
  });
  return async (ctx) => {
    const view = ctx.view as TerritoryObservation;
    const selection = promptSelectionSchema.parse(ctx.config);
    const messages = renderPlayerMessages(view, doctrine, ctx.reason, selection);
    const generationId = globalThis.crypto.randomUUID();
    const purpose = { kind: "learner" as const };
    const system = messages[0]!.content;
    const userPrompt = messages[1]!.content;
    let generation: TextGeneration | undefined;

    let reply;
    try {
      reply = await client.complete({
        purpose,
        signal: AbortSignal.any([ctx.signal, AbortSignal.timeout(client.timeoutMs)]),
        slot: ctx.playerSlot,
        system,
        messages: [{ role: "user", text: userPrompt }],
        recordGeneration: (evidence) => {
          generation = evidence;
          ctx.recordAttempt({
            generationId,
            purpose,
            prompt: userPrompt,
            response: evidence.response,
            error: null,
            generation: evidence,
            platformCallId: evidence.platformCallId ?? null,
            providerRequestId: evidence.providerRequestId ?? null,
          });
        },
      });
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      ctx.recordAttempt({
        generationId,
        purpose,
        prompt: userPrompt,
        response: generation?.response ?? "",
        error: reason,
        generation,
        platformCallId: generation?.platformCallId ?? null,
        providerRequestId: generation?.providerRequestId ?? null,
      });
      throw err;
    }

    const parsedJson = parseJsonAction(reply.text);
    if (parsedJson.kind === "rejected") {
      ctx.recordAttempt({
        generationId,
        purpose,
        prompt: userPrompt,
        response: reply.text,
        error: parsedJson.reason,
        generation: reply.generation,
        platformCallId: reply.platformCallId,
        providerRequestId: reply.generation.providerRequestId ?? null,
      });
      throw new Error(parsedJson.reason);
    }

    const validated = SubmissionSchema.safeParse(parsedJson.value);
    if (!validated.success) {
      const reason = validated.error.message;
      ctx.recordAttempt({
        generationId,
        purpose,
        prompt: userPrompt,
        response: reply.text,
        error: reason,
        generation: reply.generation,
        platformCallId: reply.platformCallId,
        providerRequestId: reply.generation.providerRequestId ?? null,
      });
      throw new Error(reason);
    }

    ctx.recordAttempt({
      generationId,
      purpose,
      prompt: userPrompt,
      response: reply.text,
      error: null,
      generation: reply.generation,
      parsedAction: validated.data,
      platformCallId: reply.platformCallId,
      providerRequestId: reply.generation.providerRequestId ?? null,
    });
    return validated.data;
  };
}

/** The scripted policy: a pure function of the view, no model, no network. */
export const makeScriptedDecide =
  (baseline: string) =>
  (ctx: Ctx): TerritoryDecision => {
    const decision = scriptedDecide(baseline, ctx.view as TerritoryObservation);
    // A real scripted decision still needs a selected attempt in the private
    // trajectory. No generation or provider identity is attached to controls.
    ctx.recordAttempt({
      prompt: "",
      response: JSON.stringify(decision),
      parsedAction: decision,
      error: null,
    });
    return decision;
  };

export function selectPlayerPolicy(
  envVars: Record<string, string | undefined> = env,
): { kind: "llm"; prompt: string } | { kind: "scripted"; baseline: string } {
  const scripted = envVars.PLAYER_SCRIPTED?.trim();
  if (scripted) return { kind: "scripted", baseline: scripted };
  const prompt = envVars.PLAYER_PROMPT?.trim();
  const endpoint = envVars.COWORLD_LLM_ENDPOINT?.trim();
  const enabled = envVars.COWORLD_LLM_ENABLED?.trim().toLowerCase() === "true";
  const model = envVars.COWORLD_LLM_MODEL?.trim();
  const apiKey = envVars.OPENROUTER_API_KEY?.trim();
  if (enabled || prompt || endpoint || model || apiKey) return { kind: "llm", prompt: prompt ?? "" };
  return { kind: "scripted", baseline: "homesteader" };
}

/** Choose the policy from the environment and play the slot to `final`. */
export function run(): Promise<number[]> {
  const selected = selectPlayerPolicy(env);
  if (selected.kind === "llm") {
    console.log(
      `[territory-player] LLM policy (model ${env.COWORLD_LLM_MODEL?.trim() || DEFAULT_MODEL}, fail-closed)`,
    );
    return runCoworldPlayer<TerritorySeamState, TerritoryDecision, TerritoryView>({
      module: territoryModule,
      decide: makeLlmDecide(selected.prompt),
    });
  }
  console.log(`[territory-player] scripted policy "${selected.baseline}"`);
  return runCoworldPlayer<TerritorySeamState, TerritoryDecision, TerritoryView>({
    module: territoryModule,
    decide: makeScriptedDecide(selected.baseline),
  });
}

// Run only when invoked as the entrypoint, not when imported by a test.
if (realpathSync(argv[1]!) === fileURLToPath(import.meta.url)) {
  await run();
}
