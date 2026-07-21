# Connections Difficulty Estimation Prompt

You are estimating how difficult a solved Connections puzzle will be for human
players.

## Game Rules

Connections is a 16-item word puzzle. The player sees all 16 items shuffled on a
board and must partition them into four hidden categories of four items each.
Each category has a title that explains the relationship among its four items.
The category titles are not shown to the player while solving. Players can make
wrong guesses, and a puzzle is harder when plausible alternate groupings,
ambiguous word senses, specialized knowledge, wordplay, or trap items make the
true partition harder to find.

The four official categories are commonly ordered from easiest to hardest:

1. Yellow
2. Green
3. Blue
4. Purple

This order is useful evidence, but the final difficulty score should reflect
the whole puzzle, not only the hardest category.

## Input Format

You will receive the board order and the solved categories:

```text
Board order:
0: ITEM; 1: ITEM; ...; 15: ITEM

Solved categories:
1. CATEGORY TITLE: ITEM, ITEM, ITEM, ITEM
2. CATEGORY TITLE: ITEM, ITEM, ITEM, ITEM
3. CATEGORY TITLE: ITEM, ITEM, ITEM, ITEM
4. CATEGORY TITLE: ITEM, ITEM, ITEM, ITEM
```

## Task

Estimate the scalar human difficulty of the puzzle on the historical expert
scale, where lower scores are easier and higher scores are harder. In this
dataset the observed expert scores range from 0.5 to 4.6, with most puzzles
near 2.0 to 3.5.

Prefer numeric accuracy over explanation style. If the evidence conflicts, give
the score that best approximates observed human/expert difficulty.

## Evidence To Consider

- How obvious each category relationship is from the items alone.
- Whether items have multiple common meanings that invite wrong groups.
- Whether many items could plausibly belong to several categories.
- Whether categories rely on specific culture, brands, idioms, spelling,
  homophones, hidden words, fill-in-the-blank constructions, or other wordplay.
- Whether the hardest category is discoverable after easier categories are
  removed.
- Whether the board contains repeated lexical patterns that help or mislead.
- Whether the puzzle has image cards; use the image alt text as the item.

## Output Format

Return strict JSON only:

```json
{
  "difficulty": 2.7,
  "explanation": "A short evidence-based explanation of the score."
}
```

The `difficulty` value must be a real scalar. The explanation should be faithful
to the provided categories and items, concise, and should not invent facts.

