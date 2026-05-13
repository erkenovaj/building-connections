// Game configuration and constants
const CONFIG = {
    MAX_MISTAKES: 4,
    CONCEPTS_PER_GROUP: 4,
    TOTAL_CATEGORIES_NORMAL: 4,
    TOTAL_CATEGORIES_ADVANCED: 3,
    DOG_UPDATE_INTERVAL: 10000,
    TOOLTIP_DELAY: 300
};

// Concept definitions: populated from category-templates-new.json (and any custom templates)
const CONCEPT_DEFINITIONS = {};

// Category templates for puzzle generation - flat list of term entries
// Each entry: { name: string, description: string, tags: string[] }
let CATEGORY_TEMPLATES = [];

// Load item-based config from JSON file and derive concept definitions and categories.
async function loadGameConfig() {
    try {
        const url = new URL('../configs/category-templates-new.json', import.meta.url).href;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load category templates: ${response.statusText}`);
        }
        const data = await response.json();
        if (!Array.isArray(data) || data.length === 0) {
            throw new Error('Category templates JSON must be a non-empty array.');
        }
        CATEGORY_TEMPLATES = data;
        // Populate concept definitions from description field
        data.forEach(entry => {
            if (entry && typeof entry.name === 'string' && typeof entry.description === 'string') {
                CONCEPT_DEFINITIONS[entry.name] = entry.description;
            }
        });
    } catch (error) {
        // If loading fails, leave CATEGORY_TEMPLATES empty; PuzzleGenerator will surface the error.
    }
}

// Initialize config-derived data immediately at module load.
await loadGameConfig();

export { CONFIG, CONCEPT_DEFINITIONS, CATEGORY_TEMPLATES };