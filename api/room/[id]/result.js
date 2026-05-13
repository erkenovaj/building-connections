import { getDb } from '../../../lib/db.js';

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
  const roomId = req.query.id;
  if (!roomId) return res.status(400).json({ error: 'Room id required' });
  try {
    const db = await getDb();
    const roomCheck = await db.execute({ sql: 'SELECT id FROM rooms WHERE id = ?', args: [roomId] });
    if (!roomCheck.rows[0]) return res.status(404).json({ error: 'Room not found' });
    const body = parseBody(req);
    const name = (body.playerName && String(body.playerName).trim()) || 'Anonymous';
    const round = Math.max(1, Number(body.roundNumber) || 1);
    const created_at = Date.now();
    await db.execute({
      sql: 'INSERT INTO room_results (room_id, player_name, score, time_seconds, won, round_number, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
      args: [
        roomId,
        name.substring(0, 50),
        Number(body.score) || 0,
        Number(body.timeSeconds) || 0,
        body.won ? 1 : 0,
        round,
        created_at,
      ],
    });
    res.status(200).json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: String(e.message) });
  }
}
