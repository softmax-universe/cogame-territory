import { realpathSync } from "node:fs";
// The ONE player entrypoint, shipped in the SAME image as the game and selected
// by env — `dist-server/game/player.js`, shimmed as `/bin/territory-player`:
//
//   PLAYER_PROMPT=<doctrine> or COWORLD_LLM_ENDPOINT  LLM policy (empty hold on invalid output)
//   PLAYER_SCRIPTED=homesteader | raider              that explicit scripted baseline
//   neither set (keyless CI smoke)                    homesteader
//
// Native Coworld model selection uses canonical OpenRouter IDs.

import { argv, env } from "node:process";
import { fileURLToPath } from "node:url";

import { OpenRouterLlmClient, robustDecide } from "@cogweb/llm";
import { runCoworldPlayer } from "@cogweb/coworld";
import type { PlayerDecideContext } from "@cogweb/coworld";

import { SubmissionSchema } from "../shared/engine/orders.js";
import { emptyHoldDecision, territoryModule } from "./game.js";
import type { TerritoryDecision, TerritorySeamState, TerritoryView } from "./game.js";
import type { TerritoryObservation } from "./redact.js";
import { renderPlayerMessages } from "./prompt.js";
import { scriptedDecide } from "./scripted.js";
import { promptSelectionSchema } from "./prompt-selection.js";

type Ctx = PlayerDecideContext<TerritorySeamState, TerritoryDecision, TerritoryView>;

/** Haiku 4.5. `maxTokens: 900` — 400 truncates ("cut off at max_tokens"). */
const DEFAULT_MODEL = "anthropic/claude-haiku.4.5";
const MAX_TOKENS = 900;
/** One retry, then an explicit empty hold (`fallback: true`). */
const MAX_ATTEMPTS = 2;

/** The LLM policy: doctrine folded into the system prompt; exhausted or
 *  unparseable model output executes an observable empty hold, never a scripted bot. */
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
    let usedFallback = true;
    const decision = await robustDecide<TerritoryDecision>({
      client,
      purpose: { kind: "learner" },
      signal: AbortSignal.any([ctx.signal, AbortSignal.timeout(client.timeoutMs)]),
      slot: ctx.playerSlot,
      system: messages[0]!.content,
      renderUser: (rejection) => {
        return messages[1]!.content + (rejection ? `\n\n${rejection}` : "");
      },
      validate: (candidate) => {
        const parsed = SubmissionSchema.parse(candidate);
        usedFallback = false;
        return parsed;
      },
      baseline: () => emptyHoldDecision(),
      recordAttempt: ctx.recordAttempt,
      markFallback: ctx.markFallback,
      maxAttempts: MAX_ATTEMPTS,
    });
    if (usedFallback) {
      console.log(
        JSON.stringify({ kind: "player_fallback", seat: ctx.seat, turn: ctx.turn, action: "empty_hold" }),
      );
    }
    return decision;
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
  if (prompt || endpoint) return { kind: "llm", prompt: prompt ?? "" };
  return { kind: "scripted", baseline: "homesteader" };
}

/** Choose the policy from the environment and play the slot to `final`. */
export function run(): Promise<number[]> {
  const selected = selectPlayerPolicy(env);
  if (selected.kind === "llm") {
    console.log(
      `[territory-player] LLM policy (model ${env.COWORLD_LLM_MODEL ?? DEFAULT_MODEL}, fallback empty_hold)`,
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
