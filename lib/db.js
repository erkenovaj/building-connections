import { createClient } from '@libsql/client';

let _client = null;

function getClient() {
  const url = process.env.TURSO_DATABASE_URL;
  const authToken = process.env.TURSO_AUTH_TOKEN;
  if (!url || !authToken) {
    throw new Error('Missing TURSO_DATABASE_URL or TURSO_AUTH_TOKEN');
  }
  if (!_client) {
    _client = createClient({ url, authToken });
  }
  return _client;
}

const SCHEMA_SQL = `
  CREATE TABLE IF NOT EXISTS rooms (
    id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    puzzle_seed TEXT NOT NULL,
    rounds INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    templates_json TEXT
  );
  CREATE TABLE IF NOT EXISTS room_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    score INTEGER NOT NULL,
    time_seconds INTEGER NOT NULL,
    won INTEGER NOT NULL,
    round_number INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (room_id) REFERENCES rooms(id)
  );
  CREATE TABLE IF NOT EXISTS solo_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT NOT NULL,
    score INTEGER NOT NULL,
    time_seconds INTEGER NOT NULL,
    mode TEXT NOT NULL,
    created_at INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_room_results_room ON room_results(room_id);
  CREATE INDEX IF NOT EXISTS idx_solo_results_score ON solo_results(score DESC);
  CREATE INDEX IF NOT EXISTS idx_solo_results_mode_score_time ON solo_results(mode, score, time_seconds);
`;

let schemaDone = false;

export async function getDb() {
  const client = getClient();
  if (!schemaDone) {
    for (const stmt of SCHEMA_SQL.split(';').map((s) => s.trim()).filter(Boolean)) {
      await client.execute(stmt);
    }
    try {
      await client.execute('ALTER TABLE rooms ADD COLUMN rounds INTEGER NOT NULL DEFAULT 1');
    } catch (_) {}
    try {
      await client.execute('ALTER TABLE room_results ADD COLUMN round_number INTEGER NOT NULL DEFAULT 1');
    } catch (_) {}
    try {
      await client.execute('ALTER TABLE rooms ADD COLUMN templates_json TEXT');
    } catch (_) {}
    schemaDone = true;
  }
  return client;
}

export function randomId(len = 8) {
  const chars = 'abcdefghjkmnpqrstuvwxyz23456789';
  let id = '';
  for (let i = 0; i < len; i++) id += chars[Math.floor(Math.random() * chars.length)];
  return id;
}
