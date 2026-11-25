import { createServer } from "node:http";
import { createHash, randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import { GamePlayerFailureSchema, runCoworldPlayer } from "@cogweb/coworld";
import { runCoworldGame } from "../coworld/server.js";
import { territoryModule } from "./game.js";
import { TrainingSession } from "./training-bridge.js";
import { makeLlmDecide, makeScriptedDecide, selectPlayerPolicy } from "./player.js";

afterEach(() => {
  vi.unstubAllEnvs();
});

it("completes the real nine-seat production loop with exact private attempts and redacted public replay", async () => {
  const directory = await mkdtemp(join(tmpdir(), "territory-training-"));
  vi.stubEnv("COGAME_RESULTS_URI", join(directory, "results.json"));
  vi.stubEnv("COGAME_SAVE_REPLAY_URI", join(directory, "replay.json"));
  vi.stubEnv("COGAME_SAVE_TRAJECTORY_URI", join(directory, "trajectory.jsonl"));
  const calls: Array<{ id: string; request: unknown; response: unknown }> = [];
  const seatCalls = new Map<string, number>();
  const inference = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      const id = randomUUID();
      const slot = String(req.headers["x-coworld-player-slot"]);
      const seen = seatCalls.get(slot) ?? 0;
      seatCalls.set(slot, seen + 1);
      const parsedBody = JSON.parse(body) as { messages: Array<{ role: string; content: string }> };
      const userText = parsedBody.messages.find((m) => m.role === "user")?.content ?? "";
      const reachLine = userText.match(/REACH[^\n]*\n\s+([^\n]+)/)?.[1] ?? "";
      const firstTile = reachLine.match(/(-?\d+,-?\d+)\(/)?.[1];
      const text =
        slot === "1" && seen === 0
          ? '{"orders":[{"type":"claim","tile":"0,0"}]}'
          : slot === "0" && seen === 0 && firstTile
            ? `Planning claim {tile} with non-action code:\n\`\`\`text\n${firstTile}: cost 4\n\`\`\`\n\`\`\`json\n{"orders":[{"type":"claim","tile":"${firstTile}"}]}\n\`\`\``
            : '{"orders":[],"messages":[]}';
      const response = {
        model: "fixture/territory",
        choices: [{ message: { content: text }, finish_reason: "stop" }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      };
      calls.push({ id, request: parsedBody, response });
      res.writeHead(200, { "content-type": "application/json", "X-Softmax-Llm-Call-Id": id });
      res.end(JSON.stringify(response));
    });
  });
  await new Promise<void>((resolve) => inference.listen(0, "127.0.0.1", resolve));
  const address = inference.address();
  if (typeof address === "string" || address === null)
    throw new Error("Fixture requires an IP listener");
  vi.stubEnv("COWORLD_LLM_ENDPOINT", `http://127.0.0.1:${address.port}`);
  vi.stubEnv("COWORLD_LLM_MODEL", "fixture/territory");
  const host = await runCoworldGame({
    host: "127.0.0.1",
    port: 0,
    config: {
      tokens: Array.from({ length: 9 }, (_, seat) => `fixture-${seat}`),
      players: Array.from({ length: 9 }, (_, seat) => ({ name: `private policy ${seat}` })),
      seed: 7,
      paceMs: 0,
      turns: 18,
    },
  });
  try {
    const decide = makeLlmDecide("private doctrine");
    const players = host.playerUrls.map((connect) =>
      runCoworldPlayer({ module: territoryModule, connect, decide }),
    );
    const [result] = await Promise.all([host.finished, Promise.all(players)]);
    expect(result.scores).toHaveLength(9);
    const trajectory = JSON.parse(await readFile(join(directory, "trajectory.jsonl"), "utf8"));
    expect(trajectory.episode).toMatchObject({
      status: "completed",
      game: "territory",
      seed_family: "7",
    });
    expect(trajectory.decisions).toHaveLength(162);
    expect(calls).toHaveLength(163);
    expect(trajectory.decisions[0].executed_action.orders).toHaveLength(1);
    expect(trajectory.decisions[0].executed_action.orders[0].type).toBe("claim");
    const byId = new Map(calls.map((call) => [call.id, call]));
    const bridge = new TrainingSession("7", 9, "private doctrine", {
      promptVariant: "canonical",
      promptSeed: 0,
    });
    for (const decision of trajectory.decisions) {
      expect(decision.visibility).toBe("private");
      expect(decision.action_status).toBe("accepted");
      expect(decision.attempts).toHaveLength(decision.decision_index === 1 ? 2 : 1);
      for (const attempt of decision.attempts) {
        expect(attempt.inference_mode).toBe("text_action");
        expect(attempt.request).toEqual(byId.get(attempt.platform_call_id)!.request);
        expect(attempt.raw_response).toBe(
          JSON.stringify(byId.get(attempt.platform_call_id)!.response),
        );
        expect(attempt.prompt).toEqual(attempt.request.messages);
        const observation = bridge.observation();
        if (observation.kind !== "decision") throw new Error("Bridge ended before hosted loop");
        const { prompt_template, ...view } = observation.semantic_view;
        expect(view).toEqual(decision.observation);
        expect(prompt_template).toEqual({
          bank: "territory-rules-v3",
          variant: "canonical",
          selection: "canonical",
          seed: 0,
          system_sha256: createHash("sha256")
            .update(observation.messages[0]!.content)
            .digest("hex"),
          user_sha256: createHash("sha256").update(observation.messages[1]!.content).digest("hex"),
        });
        expect(observation.messages).toEqual(attempt.prompt);
        const applied = bridge.step(observation.decision_id, attempt.response);
        expect(applied.kind).toBe(attempt.accepted ? "accepted" : "rejected");
        if (applied.kind === "accepted") expect(applied.action).toEqual(decision.executed_action);
      }
      const attempt = decision.attempts.at(-1);
      expect(attempt.parsed_action).toEqual(decision.executed_action);
      expect(JSON.stringify(attempt.prompt)).not.toContain("private policy");
    }
    expect(bridge.observation().kind).toBe("terminal");
    expect((await stat(join(directory, "trajectory.jsonl"))).mode & 0o777).toBe(0o600);
    const replay = JSON.parse(await readFile(join(directory, "replay.json"), "utf8"));
    const prompts = replay.frames.filter((frame: { type: string }) => frame.type === "actPrompt");
    expect(prompts).toHaveLength(162);
    for (const frame of prompts)
      for (const attempt of frame.actPrompt.attempts)
        expect(attempt).toEqual({
          prompt: "",
          response: "",
          error: attempt.error === null ? null : "Private player decision failed",
        });
    expect(JSON.stringify(replay)).not.toContain("private doctrine");
    expect(JSON.stringify(replay)).not.toContain("raw_response");
  } finally {
    await host.close();
    await new Promise<void>((resolve, reject) =>
      inference.close((error) => (error ? reject(error) : resolve())),
    );
  }
}, 60000);

it("consumes exhausted parser and engine retries with explicit engine fallbacks", () => {
  const parser = new TrainingSession("7", 9, "doctrine", {
    promptVariant: "canonical",
    promptSeed: 0,
  });
  expect(parser.step(0, "not JSON").kind).toBe("rejected");
  const consumed = parser.step(0, "not JSON");
  expect(consumed.kind).toBe("consumed_rejection");
  if (consumed.kind === "consumed_rejection") {
    expect(consumed.action).toEqual({ orders: [], messages: [], fallback: true });
    expect(consumed.observation.kind).toBe("decision");
    if (consumed.observation.kind === "decision") expect(consumed.observation.decision_id).toBe(1);
  }
  const engine = new TrainingSession("7", 9, "doctrine", {
    promptVariant: "canonical",
    promptSeed: 0,
  });
  const invalid = '{"orders":[{"type":"claim","tile":"0,0"}]}';
  expect(engine.step(0, invalid).kind).toBe("rejected");
  const engineConsumed = engine.step(0, invalid);
  expect(engineConsumed.kind).toBe("consumed_rejection");
  if (engineConsumed.kind === "consumed_rejection") {
    expect(engineConsumed.action).toEqual({ orders: [], messages: [], fallback: true });
  }
});

it("fails closed with player_failure.json and retains partial progress on ambiguity, invalid model, and backend failure", async () => {
  expect(selectPlayerPolicy({ PLAYER_SCRIPTED: "raider", COWORLD_LLM_ENDPOINT: "http://x" })).toEqual({
    kind: "scripted",
    baseline: "raider",
  });
  expect(selectPlayerPolicy({ COWORLD_LLM_ENABLED: "true" })).toEqual({
    kind: "llm",
    prompt: "",
  });
  expect(selectPlayerPolicy({ COWORLD_LLM_MODEL: "anthropic/claude-haiku-4.5" })).toEqual({
    kind: "llm",
    prompt: "",
  });
  expect(selectPlayerPolicy({ COWORLD_LLM_ENDPOINT: "http://x" })).toEqual({
    kind: "llm",
    prompt: "",
  });
  expect(selectPlayerPolicy({ PLAYER_PROMPT: "hold walls" })).toEqual({
    kind: "llm",
    prompt: "hold walls",
  });
  expect(selectPlayerPolicy({})).toEqual({ kind: "scripted", baseline: "homesteader" });

  for (const mode of ["ambiguous", "incomplete_json", "backend_503", "invalid_model_403"] as const) {
    const directory = await mkdtemp(join(tmpdir(), `territory-fail-${mode}-`));
    const resultsPath = join(directory, "results.json");
    const replayPath = join(directory, "replay.json");
    const trajectoryPath = join(directory, "trajectory.jsonl");
    const failurePath = join(directory, "player_failure.json");
    vi.stubEnv("COGAME_RESULTS_URI", resultsPath);
    vi.stubEnv("COGAME_SAVE_REPLAY_URI", replayPath);
    vi.stubEnv("COGAME_SAVE_TRAJECTORY_URI", trajectoryPath);
    vi.stubEnv("COGAME_PLAYER_FAILURE_URI", failurePath);

    let learnerCalls = 0;
    const rawAmbiguous = '```json\n{"orders":[]}\n```\nor\n```json\n{"orders":[]}\n```';
    const rawIncomplete = '```json\n{"orders":[{"type":"claim"';
    const inference = createServer((req, res) => {
      req.resume();
      req.on("end", () => {
        learnerCalls++;
        if (learnerCalls <= 2) {
          res.writeHead(200, { "content-type": "application/json", "X-Softmax-Llm-Call-Id": randomUUID() });
          res.end(
            JSON.stringify({
              model: "anthropic/claude-haiku-4.5",
              choices: [{ message: { content: '{"orders":[]}' }, finish_reason: "stop" }],
              usage: { prompt_tokens: 1, completion_tokens: 1 },
            }),
          );
          return;
        }
        if (mode === "backend_503") {
          res.writeHead(503, { "content-type": "application/json", "X-Softmax-Llm-Call-Id": randomUUID() });
          res.end(JSON.stringify({ error: { message: "upstream unavailable" } }));
          return;
        }
        if (mode === "invalid_model_403") {
          res.writeHead(403, { "content-type": "application/json", "X-Softmax-Llm-Call-Id": randomUUID() });
          res.end(JSON.stringify({ error: { message: "model not allowed" } }));
          return;
        }
        res.writeHead(200, { "content-type": "application/json", "X-Softmax-Llm-Call-Id": randomUUID() });
        res.end(
          JSON.stringify({
            model: "anthropic/claude-haiku-4.5",
            choices: [
              {
                message: { content: mode === "incomplete_json" ? rawIncomplete : rawAmbiguous },
                finish_reason: "stop",
              },
            ],
            usage: { prompt_tokens: 1, completion_tokens: 1 },
          }),
        );
      });
    });
    await new Promise<void>((resolve) => inference.listen(0, "127.0.0.1", resolve));
    const address = inference.address();
    if (typeof address === "string" || address === null)
      throw new Error("Fixture requires an IP listener");
    vi.stubEnv("COWORLD_LLM_ENDPOINT", `http://127.0.0.1:${address.port}`);
    vi.stubEnv("COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5");

    const host = await runCoworldGame({
      host: "127.0.0.1",
      port: 0,
      config: {
        tokens: Array.from({ length: 9 }, (_, seat) => `mixed-${seat}`),
        players: Array.from({ length: 9 }, (_, seat) => ({ name: seat === 0 ? "learner" : `bot-${seat}` })),
        seed: 13,
        paceMs: 0,
        turns: 6,
      },
    });
    try {
      const players = host.playerUrls.map((connect, seat) =>
        runCoworldPlayer({
          module: territoryModule,
          connect,
          decide:
            seat === 0
              ? makeLlmDecide("private doctrine")
              : makeScriptedDecide(seat % 2 === 0 ? "raider" : "homesteader"),
        }),
      );
      await expect(host.finished).rejects.toThrow(/seat 0 failed on turn 3/);
      await Promise.allSettled(players);
      expect(learnerCalls).toBe(3); // no retries on failure!
      expect(existsSync(resultsPath)).toBe(false); // no default gameplay scores!
      const failure = GamePlayerFailureSchema.parse(JSON.parse(await readFile(failurePath, "utf8")));
      expect(failure.failed_policy_index).toBe(0);
      expect(failure.kind).toBe(mode === "backend_503" ? "episode_inconclusive" : "player_error");
      const trajectory = JSON.parse(await readFile(trajectoryPath, "utf8"));
      expect(trajectory.episode.status).toBe(mode === "backend_503" ? "inconclusive" : "failed");
      expect(trajectory.episode.participant_outcomes).toBeNull();
      // Turns 1 & 2 (18 decisions) + turn 3 batch (9 decisions) preserved
      expect(trajectory.decisions.length).toBeGreaterThanOrEqual(19);
      const failedDecision = trajectory.decisions.find(
        (d: { seat: string; action_status: string }) => d.seat === "0" && d.action_status === "fallback",
      );
      expect(failedDecision).toBeDefined();
      if (mode === "ambiguous") {
        expect(failedDecision.attempts[0].response).toBe(rawAmbiguous);
      } else if (mode === "incomplete_json") {
        expect(failedDecision.attempts[0].response).toBe(rawIncomplete);
      } else if (mode === "backend_503") {
        expect(failedDecision.attempts[0].raw_response).toContain("upstream unavailable");
      } else {
        expect(failedDecision.attempts[0].raw_response).toContain("model not allowed");
      }
      const replay = JSON.parse(await readFile(replayPath, "utf8"));
      expect(replay.frames.length).toBeGreaterThan(2);
      expect(replay.failure).toMatchObject({ kind: failure.kind, failed_policy_index: 0 });
    } finally {
      await host.close();
      await new Promise<void>((resolve, reject) =>
        inference.close((error) => (error ? reject(error) : resolve())),
      );
    }
  }
}, 45000);

it("records complete selected scripted controls without model provenance", async () => {
  const directory = await mkdtemp(join(tmpdir(), "territory-scripted-training-"));
  vi.stubEnv("COGAME_RESULTS_URI", join(directory, "results.json"));
  vi.stubEnv("COGAME_SAVE_REPLAY_URI", join(directory, "replay.json"));
  vi.stubEnv("COGAME_SAVE_TRAJECTORY_URI", join(directory, "trajectory.jsonl"));
  const host = await runCoworldGame({
    host: "127.0.0.1",
    port: 0,
    config: {
      tokens: Array.from({ length: 9 }, (_, seat) => `scripted-${seat}`),
      players: Array.from({ length: 9 }, (_, seat) => ({ name: `control ${seat}` })),
      seed: 11,
      paceMs: 0,
      turns: 18,
    },
  });
  try {
    const players = host.playerUrls.map((connect, seat) =>
      runCoworldPlayer({
        module: territoryModule,
        connect,
        decide: makeScriptedDecide(seat < 5 ? "homesteader" : "raider"),
      }),
    );
    await Promise.all([host.finished, Promise.all(players)]);
    const trajectory = JSON.parse(await readFile(join(directory, "trajectory.jsonl"), "utf8"));
    expect(trajectory.episode.status).toBe("completed");
    expect(trajectory.decisions).toHaveLength(162);
    for (const decision of trajectory.decisions) {
      expect(decision.action_status).toBe("accepted");
      expect(decision.attempts).toHaveLength(1);
      const attempt = decision.attempts[0];
      expect(decision.selected_attempt_id).toBe(attempt.attempt_id);
      expect(attempt.accepted).toBe(true);
      expect(attempt.origin).toBe("unknown");
      expect(attempt.prompt).toBe("");
      expect(attempt.request).toBeNull();
      expect(attempt.model).toBeNull();
      expect(attempt.platform_call_id).toBeNull();
      expect(attempt.parsed_action).toEqual(decision.executed_action);
      expect(JSON.parse(attempt.response)).toEqual(decision.executed_action);
    }
  } finally {
    await host.close();
  }
}, 30000);
