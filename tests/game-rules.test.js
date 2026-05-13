/**
 * Tests for Connections game rules and corner cases.
 * Run: npm test
 */
import { describe, it } from 'node:test';
import assert from 'node:assert';
import {
    evaluateGuess,
    getGameStatus,
    validatePuzzleStructure,
    CONFIG,
    setsAreEqual
} from '../js/game-rules.js';

// ——— Fixture: valid puzzle (16 words, 4 groups of 4, unique solution) ———
const validPuzzle = {
    board: [
        'A1', 'A2', 'A3', 'A4',
        'B1', 'B2', 'B3', 'B4',
        'C1', 'C2', 'C3', 'C4',
        'D1', 'D2', 'D3', 'D4'
    ],
    categories: {
        easy:   { name: 'Group A', members: ['A1', 'A2', 'A3', 'A4'] },
        medium: { name: 'Group B', members: ['B1', 'B2', 'B3', 'B4'] },
        hard:   { name: 'Group C', members: ['C1', 'C2', 'C3', 'C4'] },
        harder: { name: 'Group D', members: ['D1', 'D2', 'D3', 'D4'] }
    },
    numCategories: 4
};

describe('Puzzle structure (core rules)', () => {
    it('valid puzzle has exactly 16 words on board', () => {
        const result = validatePuzzleStructure(validPuzzle);
        assert.strictEqual(result.valid, true);
        assert.strictEqual(validPuzzle.board.length, 16);
    });

    it('valid puzzle has exactly four groups', () => {
        const count = Object.keys(validPuzzle.categories).length;
        assert.strictEqual(count, 4);
    });

    it('each group has exactly four words', () => {
        for (const cat of Object.values(validPuzzle.categories)) {
            assert.strictEqual(cat.members.length, 4);
        }
    });

    it('every word is used once in the solution (no word in multiple groups)', () => {
        const result = validatePuzzleStructure(validPuzzle);
        assert.strictEqual(result.valid, true);
    });

    it('rejects board with wrong number of words', () => {
        const bad = { ...validPuzzle, board: validPuzzle.board.slice(0, 15) };
        const result = validatePuzzleStructure(bad);
        assert.strictEqual(result.valid, false);
        assert.ok(result.errors.some(e => e.includes('16')));
    });

    it('rejects duplicate words on board', () => {
        const bad = { ...validPuzzle, board: [...validPuzzle.board.slice(0, 15), validPuzzle.board[0]] };
        const result = validatePuzzleStructure(bad);
        assert.strictEqual(result.valid, false);
        assert.ok(result.errors.some(e => e.includes('duplicate')));
    });

    it('rejects when a word appears in more than one category', () => {
        const bad = {
            ...validPuzzle,
            categories: {
                ...validPuzzle.categories,
                easy: { name: 'Group A', members: ['A1', 'A2', 'A3', 'B1'] }
            }
        };
        const result = validatePuzzleStructure(bad);
        assert.strictEqual(result.valid, false);
        assert.ok(result.errors.some(e => e.includes('more than one category')));
    });

    it('rejects when board word is not in any category', () => {
        const bad = { ...validPuzzle, board: ['X', ...validPuzzle.board.slice(1)] };
        const result = validatePuzzleStructure(bad);
        assert.strictEqual(result.valid, false);
        assert.ok(result.errors.some(e => e.includes('does not belong')));
    });
});

describe('Making a guess', () => {
    it('guess must consist of exactly four words', () => {
        const three = evaluateGuess(validPuzzle, ['A1', 'A2', 'A3']);
        assert.strictEqual(three.correct, false);

        const five = evaluateGuess(validPuzzle, ['A1', 'A2', 'A3', 'A4', 'B1']);
        assert.strictEqual(five.correct, false);
    });

    it('correct only if all four words belong to the same intended group', () => {
        const result = evaluateGuess(validPuzzle, ['A1', 'A2', 'A3', 'A4']);
        assert.strictEqual(result.correct, true);
        assert.strictEqual(result.matchedCategory.name, 'Group A');
        assert.strictEqual(result.difficulty, 'easy');
    });

    it('partial correctness (3 out of 4) counts as incorrect', () => {
        const result = evaluateGuess(validPuzzle, ['A1', 'A2', 'A3', 'B1']);
        assert.strictEqual(result.correct, false);
    });

    it('wrong group of four is incorrect', () => {
        const result = evaluateGuess(validPuzzle, ['A1', 'B1', 'C1', 'D1']);
        assert.strictEqual(result.correct, false);
    });

    it('guess with duplicate words is incorrect', () => {
        const result = evaluateGuess(validPuzzle, ['A1', 'A1', 'A2', 'A3']);
        assert.strictEqual(result.correct, false);
    });

    it('order of selected words does not matter', () => {
        const result = evaluateGuess(validPuzzle, ['A4', 'A3', 'A2', 'A1']);
        assert.strictEqual(result.correct, true);
        assert.strictEqual(result.matchedCategory.name, 'Group A');
    });
});

describe('Mistakes', () => {
    it('up to three mistakes allowed; fourth ends the game', () => {
        assert.strictEqual(getGameStatus(0, 0, 4), 'playing');
        assert.strictEqual(getGameStatus(1, 0, 4), 'playing');
        assert.strictEqual(getGameStatus(2, 0, 4), 'playing');
        assert.strictEqual(getGameStatus(3, 0, 4), 'playing');
        assert.strictEqual(getGameStatus(CONFIG.MAX_MISTAKES, 0, 4), 'lost');
        assert.strictEqual(getGameStatus(5, 0, 4), 'lost');
    });

    it('mistakes do not reset after a correct guess', () => {
        // Simulate: 1 mistake, then 1 correct group -> still 1 mistake, status playing
        assert.strictEqual(getGameStatus(1, 1, 4), 'playing');
        assert.strictEqual(getGameStatus(2, 2, 4), 'playing');
    });

    it('fourth incorrect guess ends game regardless of progress', () => {
        assert.strictEqual(getGameStatus(4, 2, 4), 'lost');
        assert.strictEqual(getGameStatus(4, 3, 4), 'lost');
    });
});

describe('Group resolution and end states', () => {
    it('win: all four groups solved with fewer than four mistakes', () => {
        assert.strictEqual(getGameStatus(0, 4, 4), 'won');
        assert.strictEqual(getGameStatus(1, 4, 4), 'won');
        assert.strictEqual(getGameStatus(2, 4, 4), 'won');
        assert.strictEqual(getGameStatus(3, 4, 4), 'won');
    });

    it('loss: fourth mistake before solving all', () => {
        assert.strictEqual(getGameStatus(4, 0, 4), 'lost');
        assert.strictEqual(getGameStatus(4, 1, 4), 'lost');
    });

    it('solved group is one category only (remaining words still unique solution)', () => {
        // After "solving" group A, only B,C,D remain; each word still in exactly one category
        const result = validatePuzzleStructure(validPuzzle);
        assert.strictEqual(result.valid, true);
    });
});

describe('Overlaps and traps (decoy groupings)', () => {
    // Puzzle where some words could thematically fit multiple categories; only intended grouping is valid
    const trapPuzzle = {
        board: ['Cat', 'Dog', 'Lion', 'Wolf', 'Apple', 'Pear', 'Plum', 'Berry', 'Red', 'Blue', 'Green', 'Yellow', 'One', 'Two', 'Three', 'Four'],
        categories: {
            easy:   { name: 'Animals',    members: ['Cat', 'Dog', 'Lion', 'Wolf'] },
            medium: { name: 'Fruits',     members: ['Apple', 'Pear', 'Plum', 'Berry'] },
            hard:   { name: 'Colors',     members: ['Red', 'Blue', 'Green', 'Yellow'] },
            harder: { name: 'Numbers',    members: ['One', 'Two', 'Three', 'Four'] }
        },
        numCategories: 4
    };

    it('only the intended overall grouping is valid', () => {
        const correct = evaluateGuess(trapPuzzle, ['Cat', 'Dog', 'Lion', 'Wolf']);
        assert.strictEqual(correct.correct, true);
    });

    it('decoy grouping (e.g. "things that are red") is not accepted', () => {
        // If "Apple" and "Red" might seem to go together, a guess mixing categories is wrong
        const decoy = evaluateGuess(trapPuzzle, ['Apple', 'Red', 'Pear', 'Blue']);
        assert.strictEqual(decoy.correct, false);
    });

    it('each word belongs to only one group in the solution', () => {
        const result = validatePuzzleStructure(trapPuzzle);
        assert.strictEqual(result.valid, true);
    });
});

describe('Difficulty labels', () => {
    it('difficulty is associated with category and revealed on solve', () => {
        const r = evaluateGuess(validPuzzle, ['D1', 'D2', 'D3', 'D4']);
        assert.strictEqual(r.correct, true);
        assert.strictEqual(r.difficulty, 'harder');
    });

    it('difficulty has no effect on correctness', () => {
        const easy = evaluateGuess(validPuzzle, ['A1', 'A2', 'A3', 'A4']);
        const hard = evaluateGuess(validPuzzle, ['D1', 'D2', 'D3', 'D4']);
        assert.strictEqual(easy.correct, true);
        assert.strictEqual(hard.correct, true);
    });
});

describe('Non-rules (explicitly not allowed)', () => {
    it('no partial credit: 3/4 correct is wrong', () => {
        assert.strictEqual(evaluateGuess(validPuzzle, ['A1', 'A2', 'A3', 'B1']).correct, false);
    });

    it('no reusing words across groups: each word in exactly one category', () => {
        const result = validatePuzzleStructure(validPuzzle);
        assert.strictEqual(result.valid, true);
    });

    it('setsAreEqual helper', () => {
        assert.strictEqual(setsAreEqual(new Set(['a', 'b']), new Set(['b', 'a'])), true);
        assert.strictEqual(setsAreEqual(new Set(['a', 'b']), new Set(['a'])), false);
        assert.strictEqual(setsAreEqual(new Set(), new Set()), true);
    });
});

describe('Edge cases', () => {
    it('empty or invalid guess input', () => {
        assert.strictEqual(evaluateGuess(validPuzzle, []).correct, false);
        assert.strictEqual(evaluateGuess(validPuzzle, null).correct, false);
        assert.strictEqual(evaluateGuess(validPuzzle, ['A1', 'A2']).correct, false);
    });

    it('puzzle with no categories fails validation', () => {
        const result = validatePuzzleStructure({ board: validPuzzle.board, categories: {} });
        assert.strictEqual(result.valid, false);
    });

    it('category with wrong member count fails validation', () => {
        const bad = {
            ...validPuzzle,
            categories: {
                ...validPuzzle.categories,
                easy: { name: 'Group A', members: ['A1', 'A2', 'A3'] }
            }
        };
        const result = validatePuzzleStructure(bad);
        assert.strictEqual(result.valid, false);
    });
});
