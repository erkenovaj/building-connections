/**
 * Game rules and validation logic.
 * These functions are used by both the game UI and tests.
 */
import { CONFIG } from './config.js';

/**
 * Check if two sets contain the same elements.
 */
export function setsAreEqual(setA, setB) {
    return setA.size === setB.size && [...setA].every(item => setB.has(item));
}

/**
 * Evaluate a guess against a puzzle.
 * @param {Object} puzzle - Puzzle object with board and categories
 * @param {Array<string>} selectedWords - Array of 4 selected words
 * @returns {Object} - {correct: boolean, matchedCategory: Object|null, difficulty: string|null}
 */
export function evaluateGuess(puzzle, selectedWords) {
    // Invalid input
    if (!selectedWords || !Array.isArray(selectedWords) || selectedWords.length !== CONFIG.CONCEPTS_PER_GROUP) {
        return { correct: false, matchedCategory: null, difficulty: null };
    }

    // Check for duplicates
    if (new Set(selectedWords).size !== selectedWords.length) {
        return { correct: false, matchedCategory: null, difficulty: null };
    }

    const selectedSet = new Set(selectedWords);

    // Check each category to see if the selected words match
    for (const [difficulty, category] of Object.entries(puzzle.categories)) {
        const categorySet = new Set(category.members);
        if (setsAreEqual(categorySet, selectedSet)) {
            return { correct: true, matchedCategory: category, difficulty };
        }
    }

    return { correct: false, matchedCategory: null, difficulty: null };
}

/**
 * Get the current game status based on mistakes and solved categories.
 * @param {number} mistakes - Number of mistakes made
 * @param {number} solvedCategories - Number of categories solved
 * @param {number} numCategories - Total number of categories in the puzzle
 * @returns {string} - 'playing', 'won', or 'lost'
 */
export function getGameStatus(mistakes, solvedCategories, numCategories) {
    // Win condition: all categories solved with fewer than max mistakes
    if (solvedCategories === numCategories && mistakes < CONFIG.MAX_MISTAKES) {
        return 'won';
    }

    // Loss condition: reached max mistakes
    if (mistakes >= CONFIG.MAX_MISTAKES) {
        return 'lost';
    }

    // Otherwise still playing
    return 'playing';
}

/**
 * Validate that a puzzle structure is correct.
 * @param {Object} puzzle - Puzzle object to validate
 * @returns {Object} - {valid: boolean, errors: string[]}
 */
export function validatePuzzleStructure(puzzle) {
    const errors = [];

    // Check puzzle has required structure
    if (!puzzle || typeof puzzle !== 'object') {
        return { valid: false, errors: ['Puzzle must be an object'] };
    }

    if (!Array.isArray(puzzle.board)) {
        return { valid: false, errors: ['Puzzle must have a board array'] };
    }

    if (!puzzle.categories || typeof puzzle.categories !== 'object') {
        return { valid: false, errors: ['Puzzle must have a categories object'] };
    }

    // Check board has exactly 16 words
    if (puzzle.board.length !== 16) {
        errors.push(`Board must have exactly 16 words, found ${puzzle.board.length}`);
    }

    // Check for duplicate words on board
    const boardSet = new Set(puzzle.board);
    if (boardSet.size !== puzzle.board.length) {
        errors.push('Board contains duplicate words');
    }

    // Check categories
    const categoryKeys = Object.keys(puzzle.categories);
    if (categoryKeys.length === 0) {
        errors.push('Puzzle must have at least one category');
    }

    // Check each category has exactly 4 members
    const allCategoryMembers = [];
    for (const [difficulty, category] of Object.entries(puzzle.categories)) {
        if (!category.members || !Array.isArray(category.members)) {
            errors.push(`Category ${difficulty} must have a members array`);
            continue;
        }

        if (category.members.length !== CONFIG.CONCEPTS_PER_GROUP) {
            errors.push(`Category ${difficulty} must have exactly ${CONFIG.CONCEPTS_PER_GROUP} members, found ${category.members.length}`);
        }

        allCategoryMembers.push(...category.members);
    }

    // Check no word appears in multiple categories
    const memberCounts = {};
    for (const member of allCategoryMembers) {
        memberCounts[member] = (memberCounts[member] || 0) + 1;
        if (memberCounts[member] > 1) {
            errors.push(`Word "${member}" appears in more than one category`);
        }
    }

    // Check every board word belongs to exactly one category
    for (const word of puzzle.board) {
        if (!allCategoryMembers.includes(word)) {
            errors.push(`Board word "${word}" does not belong to any category`);
        }
    }

    // Check every category member appears on the board
    for (const member of allCategoryMembers) {
        if (!puzzle.board.includes(member)) {
            errors.push(`Category member "${member}" does not appear on the board`);
        }
    }

    return {
        valid: errors.length === 0,
        errors
    };
}

// Re-export CONFIG for convenience
export { CONFIG };
