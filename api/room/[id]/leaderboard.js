import { getDb } from '../../../lib/db.js';

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed' });
  const roomId = req.query.id;
  if (!roomId) return res.status(400).json({ error: 'Room id required' });
  try {
    const db = await getDb();
    const roomCheck = await db.execute({ sql: 'SELECT id FROM rooms WHERE id = ?', args: [roomId] });
    if (!roomCheck.rows[0]) return res.status(404).json({ error: 'Room not found' });
    const roundNum = req.query.round != null ? Number(req.query.round) : null;
    const cap = 500;
    let result;
    if (roundNum != null && !Number.isNaN(roundNum)) {
      result = await db.execute({
        sql: 'SELECT player_name AS playerName, score, time_seconds AS timeSeconds, won, round_number AS roundNumber, created_at AS createdAt FROM room_results WHERE room_id = ? AND round_number = ? ORDER BY round_number ASC, score DESC, time_seconds ASC LIMIT ?',
        args: [roomId, roundNum, cap],
      });
    } else {
      result = await db.execute({
        sql: 'SELECT player_name AS playerName, score, time_seconds AS timeSeconds, won, round_number AS roundNumber, created_at AS createdAt FROM room_results WHERE room_id = ? ORDER BY round_number ASC, score DESC, time_seconds ASC LIMIT ?',
        args: [roomId, cap],
      });
    }
    res.setHeader('Cache-Control', 'public, max-age=10, s-maxage=10, stale-while-revalidate=30');
    res.status(200).json(result.rows);
  } catch (e) {
    res.status(500).json({ error: String(e.message) });
  }
}
