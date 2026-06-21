/**
 * API client for the Connections backend.
 * Uses same origin when served from Node (e.g. http://localhost:3000).
 */
const getBase = () =>
  typeof window !== "undefined" ? window.location.origin : "";

export async function createRoom(mode = "basic", rounds = 1, templates = null) {
  const res = await fetch(`${getBase()}/api/room`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mode,
      rounds,
      // Only send templates when custom templates are provided (non-default)
      templates:
        Array.isArray(templates) && templates.length ? templates : null,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getRoom(roomId) {
  const res = await fetch(`${getBase()}/api/room/${roomId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function submitRoomResult(
  roomId,
  { playerName, score, timeSeconds, won, roundNumber },
) {
  const res = await fetch(`${getBase()}/api/room/${roomId}/result`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ playerName, score, timeSeconds, won, roundNumber }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getRoomLeaderboard(roomId) {
  const res = await fetch(`${getBase()}/api/room/${roomId}/leaderboard`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function sampleGame({
  mode = "basic",
  config = null,
  seed = null,
}) {
  const res = await fetch(`${getBase()}/api/game/sample`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, config, seed }),
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    const message =
      payload.reason ||
      payload.error ||
      `Sampler request failed (${res.status})`;
    throw new Error(message);
  }
  return payload;
}

export async function runLlmSolver({
  board,
  numCategories,
  model,
  maxGuesses,
}) {
  const res = await fetch(`${getBase()}/api/llm/solve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ board, numCategories, model, maxGuesses }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function runLlmGuess({
  remaining,
  numCategories,
  history,
  model,
}) {
  const res = await fetch(`${getBase()}/api/llm/guess`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ remaining, numCategories, history, model }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getLlmModels() {
  const res = await fetch(`${getBase()}/api/llm/models`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function isApiAvailable() {
  return true;
}
