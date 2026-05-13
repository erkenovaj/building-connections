import { getDb, randomId } from '../../lib/db.js';

function parseBody(req) {
  if (req.body == null) return {};
  return typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
}

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    return res.status(200).end();
  }
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  try {
    const body = parseBody(req);
    const mode = body.mode === 'easy'
      ? 'easy'
      : (body.mode === 'advanced' ? 'advanced' : 'basic');
    const rounds = Math.max(1, Math.min(10, Number(body.rounds) || 1));
    const templates = Array.isArray(body.templates) && body.templates.length ? JSON.stringify(body.templates) : null;
    const id = randomId();
    const puzzle_seed = id + '-' + Date.now();
    const created_at = Date.now();
    const db = await getDb();
    await db.execute({
      sql: 'INSERT INTO rooms (id, mode, puzzle_seed, rounds, created_at, templates_json) VALUES (?, ?, ?, ?, ?, ?)',
      args: [id, mode, puzzle_seed, rounds, created_at, templates],
    });
    const origin = req.headers.origin || (req.headers['x-forwarded-proto'] || 'https') + '://' + (req.headers['x-forwarded-host'] || req.headers.host || '');
    const joinLink = `${origin}?room=${id}`;
    res.status(200).json({ roomId: id, joinLink, seed: puzzle_seed, mode, rounds });
  } catch (e) {
    res.status(500).json({ error: String(e.message) });
  }
}
