import TooltipManager from "./tooltip.js";
import DogAnimations from "./dog-animations.js";
import Sounds from "./sounds.js";
import { CONFIG, CONCEPT_DEFINITIONS, CATEGORY_TEMPLATES } from "./config.js";
import * as api from "./api.js";
import { evaluateGuess, remainingTerms } from "./game-rules.js";

const GUIDE_SEEN_KEY = "ai_connections_guide_seen";
const LANG_STORAGE_KEY = "ai_connections_lang";

const UI_STRINGS = {
  headerTitle: {
    en: "Connections",
    ru: "Connections",
  },
  headerDescBasic: {
    en: "Create four groups of four that share a common theme. The dog is watching...",
    ru: "Соберите четыре группы по четыре термина с общей темой. Пёсик наблюдает...",
  },
  headerDescAdvanced: {
    en: "Create three groups of four that share a common theme. Four decoys included. The dog is watching...",
    ru: "Соберите три группы по четыре термина с общей темой. Добавлены четыре «обманки». Пёсик наблюдает...",
  },
  welcomeTitle: {
    en: "Connections",
    ru: "Connections",
  },
  welcomeDesc: {
    en: "Create groups of four that share a common theme. Choose how you want to play:",
    ru: "Соберите группы по четыре термина с общей темой. Выберите, как вы хотите играть:",
  },
  soloPanelTitle: {
    en: "Solo game",
    ru: "Одиночная игра",
  },
  roomPanelTitle: {
    en: "Room (multiplayer)",
    ru: "Комната (мультиплеер)",
  },
  modeLabel: {
    en: "Mode:",
    ru: "Режим:",
  },
  modeEasy: {
    en: "Easy (4 categories)",
    ru: "Лёгкий (4 категории)",
  },
  modeBasic: {
    en: "Basic (4 categories)",
    ru: "Обычный (4 категории)",
  },
  modeAdvanced: {
    en: "Advanced (3 categories)",
    ru: "Продвинутый (3 категории)",
  },
  startSolo: {
    en: "Start solo game",
    ru: "Начать одиночную игру",
  },
  createRoom: {
    en: "Create room",
    ru: "Создать комнату",
  },
  backHome: {
    en: "Back to home",
    ru: "На главный экран",
  },
  joinLabel: {
    en: "Already have a room link?",
    ru: "Уже есть ссылка на комнату?",
  },
  joinButton: {
    en: "Join",
    ru: "Присоединиться",
  },
  joinHint: {
    en: "Solo: play on your own. Room: same puzzle for everyone, shared room leaderboard.",
    ru: "Solo: play on your own. Room: same puzzle for everyone, shared room leaderboard.",
  },
  templatesLabel: {
    en: "Category templates (optional):",
    ru: "Категории (необязательно, из JSON):",
  },
  templatesApply: {
    en: "Use pasted JSON",
    ru: "Использовать JSON из поля",
  },
  linkPlaceholder: {
    en: "https://...?room=abc12345",
    ru: "https://...?room=abc12345",
  },
  defsOn: {
    en: "📖 DEFS: ON",
    ru: "📖 ОПРЕД: ВКЛ",
  },
  defsOff: {
    en: "🚫 DEFS: OFF",
    ru: "🚫 ОПРЕД: ВЫКЛ",
  },
  submit: {
    en: "Submit",
    ru: "Проверить",
  },
  deselectAll: {
    en: "Deselect All",
    ru: "Снять выделение",
  },
  newGame: {
    en: "New Game",
    ru: "Новая игра",
  },
  dictionaryButton: {
    en: "Dictionary",
    ru: "Словарь",
  },
  guideButton: {
    en: "Guide",
    ru: "Правила",
  },
  leaderboardButton: {
    en: "Leaderboard",
    ru: "Таблица лидеров",
  },
  winTitle: {
    en: "🎉 Puzzle Complete!",
    ru: "🎉 Головоломка решена!",
  },
  winDogLine: {
    en: "You've successfully grouped all the AI safety concepts! The dog is very proud! 🐕",
    ru: "Вы успешно собрали все группы по безопасности ИИ! Пёс вами очень гордится! 🐕",
  },
  winNameLabel: {
    en: "Name for leaderboard (optional):",
    ru: "Имя для таблицы лидеров (необязательно):",
  },
  winSave: {
    en: "Save & Play Again",
    ru: "Сохранить и сыграть ещё раз",
  },
  winNext: {
    en: "Next game",
    ru: "Следующая игра",
  },
  loseTitle: {
    en: "Game Over",
    ru: "Игра окончена",
  },
  loseButton: {
    en: "Try Again",
    ru: "Попробовать ещё раз",
  },
  roomJoinTitle: {
    en: "Join room",
    ru: "Вход в комнату",
  },
  roomJoinText: {
    en: "Enter your name to appear on the room leaderboard:",
    ru: "Введите имя, которое будет показано в таблице лидеров комнаты:",
  },
  roomJoinLabel: {
    en: "Your name",
    ru: "Ваше имя",
  },
  roomJoinStart: {
    en: "Start",
    ru: "Начать",
  },
  dictionaryTitle: {
    en: "AI Safety Dictionary",
    ru: "Словарь по безопасности ИИ",
  },
  dictionaryIntro: {
    en: "Definitions of all concepts in the game:",
    ru: "Определения всех понятий, которые встречаются в игре:",
  },
  dictionaryClose: {
    en: "Close",
    ru: "Закрыть",
  },
  guideTitle: {
    en: "How to Play Connections",
    ru: "Как играть в Connections",
  },
  guideStart: {
    en: "Start Playing",
    ru: "Начать игру",
  },
  leaderboardTitle: {
    en: "🏆 Leaderboard",
    ru: "🏆 Таблица лидеров",
  },
  leaderboardClose: {
    en: "Close",
    ru: "Закрыть",
  },
  copyLink: {
    en: "Copy link",
    ru: "Копировать ссылку",
  },
  copied: {
    en: "Copied!",
    ru: "Скопировано!",
  },
  lobbyLeaderboard: {
    en: "Leaderboard",
    ru: "Таблица лидеров",
  },
  downloadResults: {
    en: "Download results",
    ru: "Скачать результаты",
  },
  hintLabel: {
    en: "💡 Hint (%d)",
    ru: "💡 Подсказка (%d)",
  },
  roundOf: {
    en: "Round %1 of %2",
    ru: "Раунд %1 из %2",
  },
  scoreEmpty: {
    en: "Score: —",
    ru: "Очки: —",
  },
  scoreWithNumber: {
    en: "Score: %d",
    ru: "Очки: %d",
  },
  categoryN: {
    en: "Category %d",
    ru: "Категория %d",
  },
  loadingText: {
    en: "Loading…",
    ru: "Загрузка…",
  },
  guideSections: [
    {
      title: { en: "🎯 Objective", ru: "🎯 Цель" },
      body: {
        en: "<p><strong>Basic:</strong> Find 4 groups of 4 concepts. <strong>Advanced:</strong> Find 3 groups of 4 (plus decoys). Each group corresponds to a category with increasing difficulty.</p>",
        ru: "<p><strong>Обычный:</strong> найдите 4 группы по 4 понятия. <strong>Продвинутый:</strong> найдите 3 группы по 4 (плюс отвлекающие). Каждая группа — категория с возрастающей сложностью.</p>",
      },
    },
    {
      title: { en: "🕹️ Gameplay", ru: "🕹️ Как играть" },
      body: {
        en: "<ul><li><strong>Select 4 tiles</strong> that you think belong to the same category</li><li><strong>Click Submit</strong> to check if your grouping is correct</li><li><strong>You have 4 attempts</strong> - use them wisely!</li></ul>",
        ru: "<ul><li><strong>Выберите 4 плитки</strong>, которые, по вашему мнению, относятся к одной категории</li><li><strong>Нажмите «Проверить»</strong>, чтобы проверить группировку</li><li><strong>У вас 4 попытки</strong> — используйте их с умом!</li></ul>",
      },
    },
    {
      title: { en: "⏱️ Timer & Score", ru: "⏱️ Таймер и очки" },
      body: {
        en: "<ul><li>The <strong>timer</strong> runs from the start of each game until you win or lose.</li><li><strong>Score</strong> is based on correct categories, mistakes, and time. Higher is better!</li><li>Room games use a shared room leaderboard.</li></ul>",
        ru: "<ul><li><strong>Таймер</strong> идёт с начала игры до победы или поражения.</li><li><strong>Очки</strong> зависят от угаданных категорий, ошибок и времени. Чем больше — тем лучше!</li><li>Введите имя при победе, чтобы попасть в <strong>таблицу лидеров</strong>.</li></ul>",
      },
    },
    {
      title: {
        en: "📱 On Phones & Tablets",
        ru: "📱 На телефонах и планшетах",
      },
      body: {
        en: "<ul><li><strong>Tap a tile</strong> once to see its definition (when hints are ON). Tap again to select it for your group.</li><li>Use the Dictionary button for all definitions.</li></ul>",
        ru: "<ul><li><strong>Нажмите на плитку</strong> один раз, чтобы увидеть определение (при включённых подсказках). Нажмите снова, чтобы выбрать её в группу.</li><li>Кнопка «Словарь» — все определения.</li></ul>",
      },
    },
    {
      title: { en: "🐕 The AI Safety Dog", ru: "🐕 Пёс безопасности ИИ" },
      body: {
        en: "<p>Your canine companion reacts to your progress:</p><ul><li>🐕 <strong>Happy</strong> - When you're doing well</li><li>😟 <strong>Worried</strong> - When you make mistakes</li><li>🎉 <strong>Celebrating</strong> - When you solve categories</li><li>😴 <strong>Sleeping</strong> - When the game ends</li></ul>",
        ru: "<p>Ваш пёс реагирует на ваш прогресс:</p><ul><li>🐕 <strong>Доволен</strong> — когда вы справляетесь хорошо</li><li>😟 <strong>Беспокоится</strong> — когда вы ошибаетесь</li><li>🎉 <strong>Празднует</strong> — когда вы отгадываете категории</li><li>😴 <strong>Спит</strong> — когда игра закончена</li></ul>",
      },
    },
    {
      title: { en: "💡 Tips", ru: "💡 Советы" },
      body: {
        en: "<ul><li>Hover (or tap on mobile) over concepts to see their definitions</li><li>Use the Dictionary for quick reference</li><li>Look for both technical and thematic connections</li><li>Think about both the meaning and context of each term</li></ul>",
        ru: "<ul><li>Наведите курсор (или нажмите на мобильном) на понятие, чтобы увидеть определение</li><li>Словарь — для быстрой справки</li><li>Ищите и технические, и тематические связи</li><li>Учитывайте и значение, и контекст каждого термина</li></ul>",
      },
    },
  ],
};

function parseUrlConfig() {
  const params = new URLSearchParams(window.location.search);
  const room = params.get("room");
  const modeParam = params.get("mode");
  return {
    mode:
      modeParam === "normal"
        ? "basic"
        : modeParam === "easy" ||
            modeParam === "basic" ||
            modeParam === "advanced"
          ? modeParam
          : null,
    showGuide: params.get("guide") !== "0",
    roomId: room || null,
  };
}

function isTouchDevice() {
  return "ontouchstart" in window || navigator.maxTouchPoints > 0;
}

class AISafetyGame {
  constructor() {
    this.currentPuzzle = null;
    this.selectedConcepts = [];
    this.mistakes = 0;
    this.solvedCategories = 0;
    this.hintsEnabled = true;
    this.hintsRemaining = 4;
    this.hintRevealedConcepts = new Set();
    this.gameMode = "basic";
    this.tooltipManager = new TooltipManager();
    this.dogAnimations = new DogAnimations();
    this.timerStart = null;
    this.timerInterval = null;
    this.elapsedSeconds = 0;
    this.currentScore = 0;
    this.lastTappedForTooltip = null;
    this.lastGuessTimestamp = null;
    this.conceptCategoryMap = {};
    this.regime = "solo";
    this.roomId = null;
    this.puzzleSeed = null;
    this.baseSeed = null;
    this.joinLink = null;
    this.playerName = "";
    this.customTemplates = null;
    this.currentRound = 1;
    this.totalRounds = 1;
    this.dictionaryEntries = [];
    this.llmBenchmark = null;
    this.isLlmRunning = false;
    this.agentModel = null;
    this.pendingAgentRun = false;
    this.language = localStorage.getItem(LANG_STORAGE_KEY) || "en";
    this._savingWinResult = false;
    this._savingRoundResult = false;
    this.init();
  }

  getActiveLanguageCode() {
    return this.language === "ru" ? "ru" : "en";
  }

  normalizeTemplateEntry(entry, lang) {
    if (!entry) return null;
    const pickText = (val) => {
      if (val == null) return null;
      if (typeof val === "string") return val;
      if (typeof val === "object") {
        if (val[lang]) return val[lang];
        if (val.en) return val.en;
        const keys = Object.keys(val);
        if (keys.length > 0) return val[keys[0]];
      }
      return null;
    };
    const name = pickText(entry.name);
    const description = pickText(entry.description);
    let tags = [];
    const tagsRaw = entry.tags;
    if (Array.isArray(tagsRaw)) {
      tags = tagsRaw.map((t) => String(t));
    } else if (tagsRaw && typeof tagsRaw === "object") {
      const perLang =
        tagsRaw[lang] || tagsRaw.en || tagsRaw[Object.keys(tagsRaw)[0]];
      if (Array.isArray(perLang)) {
        tags = perLang.map((t) => String(t));
      }
    }
    if (!name || !description || !tags.length) return null;
    return { name, description, tags };
  }

  getTemplatesForLanguage(lang) {
    const source =
      this.customTemplates &&
      Array.isArray(this.customTemplates) &&
      this.customTemplates.length
        ? this.customTemplates
        : Array.isArray(CATEGORY_TEMPLATES) && CATEGORY_TEMPLATES.length
          ? CATEGORY_TEMPLATES
          : null;
    if (!source) return null;
    const normalized = source
      .map((entry) => this.normalizeTemplateEntry(entry, lang))
      .filter(Boolean);
    return normalized.length ? normalized : null;
  }

  refreshConceptDefinitionsForLanguage(lang) {
    Object.keys(CONCEPT_DEFINITIONS).forEach(
      (k) => delete CONCEPT_DEFINITIONS[k],
    );
    const templates = this.getTemplatesForLanguage(lang);
    if (templates) {
      templates.forEach((entry) => {
        CONCEPT_DEFINITIONS[entry.name] = entry.description;
      });
    }
  }

  /**
   * Build maps from old-language concept names and category tag names to new-language.
   * Used when switching language mid-game so the same logical puzzle is shown in the other language.
   */
  buildLanguageMaps(oldLang, newLang) {
    const oldToNew = new Map();
    const oldTagToNewTag = new Map();
    const source =
      this.customTemplates &&
      Array.isArray(this.customTemplates) &&
      this.customTemplates.length
        ? this.customTemplates
        : Array.isArray(CATEGORY_TEMPLATES) && CATEGORY_TEMPLATES.length
          ? CATEGORY_TEMPLATES
          : null;
    if (!source) return { oldToNew, oldTagToNewTag };
    for (const entry of source) {
      const oldE = this.normalizeTemplateEntry(entry, oldLang);
      const newE = this.normalizeTemplateEntry(entry, newLang);
      if (oldE && newE) {
        oldToNew.set(oldE.name, newE.name);
        for (let i = 0; i < oldE.tags.length; i++) {
          if (newE.tags[i]) oldTagToNewTag.set(oldE.tags[i], newE.tags[i]);
        }
      }
    }
    return { oldToNew, oldTagToNewTag };
  }

  /**
   * Translate current puzzle and all game state to the new language (same game, same mistakes/solves).
   * @param {string} oldLang - language the puzzle is currently in
   * @param {string} newLang - language to switch to
   */
  translateCurrentGameToLanguage(oldLang, newLang) {
    if (!this.currentPuzzle) return;
    this.tooltipManager.forceHide();
    const { oldToNew, oldTagToNewTag } = this.buildLanguageMaps(
      oldLang,
      newLang,
    );
    const mapConcept = (c) => oldToNew.get(c) ?? c;
    const mapTag = (t) => oldTagToNewTag.get(t) ?? t;

    this.currentPuzzle.board = this.currentPuzzle.board.map(mapConcept);
    for (const difficulty of Object.keys(this.currentPuzzle.categories)) {
      const cat = this.currentPuzzle.categories[difficulty];
      cat.name = mapTag(cat.name);
      cat.members = cat.members.map(mapConcept);
    }
    const newConceptCategoryMap = {};
    for (const oldConcept of Object.keys(this.conceptCategoryMap)) {
      const oldCat = this.conceptCategoryMap[oldConcept];
      newConceptCategoryMap[mapConcept(oldConcept)] = mapTag(oldCat);
    }
    this.conceptCategoryMap = newConceptCategoryMap;
    this.selectedConcepts = this.selectedConcepts.map(mapConcept);
    this.hintRevealedConcepts = new Set(
      [...this.hintRevealedConcepts].map(mapConcept),
    );

    this.refreshConceptDefinitionsForLanguage(newLang);
    this.updateDictionaryEntries();
    this.createGameBoard();
    this.updateSubmitButton();
    this.updateHintRevealedOnBoard();
    document.querySelectorAll(".category-slot.filled").forEach((slot) => {
      const difficulty = slot.dataset.difficulty;
      const category = this.currentPuzzle.categories[difficulty];
      if (category) {
        slot.querySelector(".category-name").textContent = category.name;
        slot.querySelector(".category-concepts").textContent =
          category.members.join(", ");
      }
    });
  }

  showLoading() {
    const el = document.getElementById("app-loading");
    if (el) el.classList.add("visible");
  }

  hideLoading() {
    const el = document.getElementById("app-loading");
    if (el) el.classList.remove("visible");
  }

  init() {
    const urlConfig = parseUrlConfig();
    if (urlConfig.mode !== null) this.gameMode = urlConfig.mode;
    this.bindEvents();
    this.applyLanguage();
    // Warm up the backend / database so creating a room feels faster.
    this.prewarmBackend();
    this.updateDictionaryEntries();
    this.updateHintsToggle();
    this.updateHintButton();
    this.updateModeToggle();
    this.updateHeaderDesc();
    this.updateTimerDisplay(0);
    this.updateScoreDisplay(null);
    this.loadLlmModels();

    if (urlConfig.roomId) {
      this.showLoading();
      api
        .getRoom(urlConfig.roomId)
        .then((data) => {
          this.hideLoading();
          this.showGameWithRoom(data);
        })
        .catch(() => {
          this.hideLoading();
          this.showWelcome();
        });
      return;
    }
    this.showWelcome();
  }

  prewarmBackend() {
    if (this._backendPrewarmed) return;
    this._backendPrewarmed = true;
  }

  showWelcome() {
    document.getElementById("welcome-screen").style.display = "flex";
    document.getElementById("game-container").style.display = "none";
    this.bindWelcomeEvents();
  }

  bindWelcomeEvents() {
    document.getElementById("welcome-solo").onclick = () => this.startSolo();
    document.getElementById("welcome-lobby-create").onclick = () =>
      this.createRoomWithOptions();
    document.getElementById("welcome-join-btn").onclick = () =>
      this.joinRoomFromInput();
    document.getElementById("welcome-room-link").onkeydown = (e) => {
      if (e.key === "Enter") this.joinRoomFromInput();
    };

    document.querySelectorAll(".welcome-mode-btn").forEach((btn) => {
      btn.onclick = (e) => {
        const mode = e.target.dataset.mode;
        document
          .querySelectorAll(".welcome-mode-btn")
          .forEach((b) => b.classList.remove("active"));
        e.target.classList.add("active");
        this.gameMode = mode;
      };
    });

    document.getElementById("solo-templates-btn").onclick = () =>
      document.getElementById("solo-templates-file").click();
    document.getElementById("solo-templates-file").onchange = (e) =>
      this.handleTemplateFile(e);

    const roundsSlider = document.getElementById("lobby-rounds");
    const roundsValue = document.getElementById("lobby-rounds-value");
    roundsSlider.oninput = (e) => {
      roundsValue.textContent = e.target.value;
    };

    const applyBtn = document.getElementById("templates-apply-btn");
    if (applyBtn) {
      applyBtn.onclick = () => {
        const textarea = document.getElementById("templates-textarea");
        if (!textarea) return;
        const raw = textarea.value && textarea.value.trim();
        if (!raw) {
          alert("Paste JSON into the box before applying.");
          return;
        }
        try {
          const json = JSON.parse(raw);
          if (!Array.isArray(json) || json.length === 0) {
            throw new Error("Invalid format");
          }
          json.forEach((entry) => {
            if (!entry) throw new Error("Invalid format");
            const hasSimple =
              typeof entry.name === "string" &&
              typeof entry.description === "string" &&
              Array.isArray(entry.tags) &&
              entry.tags.length > 0;
            const hasMultilang =
              entry.name &&
              typeof entry.name === "object" &&
              entry.description &&
              typeof entry.description === "object" &&
              entry.tags &&
              (Array.isArray(entry.tags) || typeof entry.tags === "object");
            if (!hasSimple && !hasMultilang) {
              throw new Error("Invalid format");
            }
          });
          this.customTemplates = json;
          const lang = this.getActiveLanguageCode();
          this.refreshConceptDefinitionsForLanguage(lang);
          this.updateDictionaryEntries();
          document.getElementById("solo-templates-name").textContent =
            "Pasted JSON";
        } catch (err) {
          alert(
            'Invalid JSON. Expected an array of entries like { "name": "Term", "description": "...", "tags": ["Category name"] }.',
          );
        }
      };
    }
  }

  handleTemplateFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target.result);
        if (!Array.isArray(json) || json.length === 0)
          throw new Error("Invalid format");
        json.forEach((entry) => {
          if (!entry) throw new Error("Invalid format");
          const hasSimple =
            typeof entry.name === "string" &&
            typeof entry.description === "string" &&
            Array.isArray(entry.tags) &&
            entry.tags.length > 0;
          const hasMultilang =
            entry.name &&
            typeof entry.name === "object" &&
            entry.description &&
            typeof entry.description === "object" &&
            entry.tags &&
            (Array.isArray(entry.tags) || typeof entry.tags === "object");
          if (!hasSimple && !hasMultilang) {
            throw new Error("Invalid format");
          }
        });
        // Accept: flat or multilingual list; keep original, normalize per language later
        this.customTemplates = json;
        const lang = this.getActiveLanguageCode();
        this.refreshConceptDefinitionsForLanguage(lang);
        this.updateDictionaryEntries();
        document.getElementById("solo-templates-name").textContent = file.name;
      } catch (err) {
        alert(
          'Invalid JSON file. Expected an array of entries like { \"name\": \"Term\", \"description\": \"...\", \"tags\": [\"Category name\"] }.',
        );
        event.target.value = "";
      }
    };
    reader.readAsText(file);
  }

  joinRoomFromInput() {
    const input = document.getElementById("welcome-room-link");
    const raw = (input && input.value && input.value.trim()) || "";
    const match =
      raw.match(/room=([a-z0-9]+)/i) ||
      (raw.length >= 6 && !raw.includes(" ") ? [{}, raw] : null);
    const roomId = match ? (match[1] || match[0]).toLowerCase() : null;
    if (roomId) {
      window.location.search = "?room=" + roomId;
    } else {
      alert(
        "Paste a full room link (e.g. https://...?room=abc12345) or enter the room code.",
      );
    }
  }

  createRoomWithOptions() {
    const mode = this.gameMode || "basic";
    const roundsInput = document.getElementById("lobby-rounds");
    if (!roundsInput) {
      alert("Error: Could not find rounds input");
      return;
    }
    const rounds = Math.max(
      1,
      Math.min(10, parseInt(roundsInput.value, 10) || 1),
    );
    const templates = this.customTemplates;
    this.showLoading();
    api
      .createRoom(mode, rounds, templates)
      .then((data) => {
        this.hideLoading();
        sessionStorage.setItem(`room_rounds_${data.roomId}`, String(rounds));
        if (templates) {
          sessionStorage.setItem(
            `room_templates_${data.roomId}`,
            JSON.stringify(templates),
          );
        }
        window.location.search = "?room=" + data.roomId;
      })
      .catch((err) => {
        this.hideLoading();
        alert(
          "Could not create room. Is the server running? " +
            (err.message || ""),
        );
      });
  }

  startAgentGame() {
    const select = document.getElementById("agent-model-input");
    this.agentModel =
      (select && select.value && select.value.trim()) || "llama3.2:3b";
    this.pendingAgentRun = true;
    this.startSolo();
  }

  startSolo() {
    const active = document.querySelector(".welcome-mode-btn.active");
    this.gameMode = active ? active.dataset.mode : this.gameMode || "basic";
    const templates = this.customTemplates;
    if (templates) {
      this.customTemplates = templates;
    }
    document.getElementById("welcome-screen").style.display = "none";
    document.getElementById("game-container").style.display = "block";
    this.regime = "solo";
    this.roomId = null;
    this.puzzleSeed = null;
    this.joinLink = null;
    this.currentRound = 1;
    this.totalRounds = 1;
    document.getElementById("lobby-bar").style.display = "none";
    document.querySelector(".header").classList.remove("has-lobby-bar");
    const lbBtn = document.querySelector(".leaderboard-btn");
    if (lbBtn) lbBtn.style.display = "none";
    const modeToggle = document.querySelector(".mode-toggle");
    if (modeToggle) modeToggle.style.display = "flex";
    document.querySelectorAll(".mode-btn").forEach((btn) => {
      btn.disabled = false;
    });
    if (document.getElementById("leaderboard-download-btn"))
      document.getElementById("leaderboard-download-btn").style.display =
        "none";
    this.updateModeToggle();
    this.updateHeaderDesc();
    this.updateRoundDisplay();
    if (!localStorage.getItem(GUIDE_SEEN_KEY))
      setTimeout(() => this.showGuide(), 100);
    this.showLoading();
    requestAnimationFrame(async () => {
      try {
        await this.startNewGame();
      } finally {
        this.hideLoading();
        if (this.pendingAgentRun) {
          this.pendingAgentRun = false;
          this.runLlmAgent();
        }
      }
    });
  }

  showGameWithRoom(roomData) {
    document.getElementById("welcome-screen").style.display = "none";
    document.getElementById("game-container").style.display = "block";
    this.regime = "lobby";
    this.roomId = roomData.roomId;
    // Prefer locally stored rounds for the room (creator’s choice), fall back to server value
    const storedRounds = sessionStorage.getItem(`room_rounds_${this.roomId}`);
    if (storedRounds !== null && storedRounds !== undefined) {
      this.totalRounds = Number(storedRounds) || 1;
    } else {
      this.totalRounds =
        roomData.rounds !== undefined && roomData.rounds !== null
          ? Number(roomData.rounds)
          : 1;
    }
    this.baseSeed = roomData.seed || `${this.roomId}-${Date.now()}`;
    const storedRound = localStorage.getItem(`room_round_${this.roomId}`);
    this.currentRound = storedRound ? parseInt(storedRound, 10) : 1;
    if (this.currentRound > this.totalRounds) {
      this.currentRound = this.totalRounds;
    }
    this.puzzleSeed = this.generateRoundSeed(this.currentRound);
    this.joinLink = window.location.href.split("?")[0] + "?room=" + this.roomId;
    this.gameMode =
      roomData.mode === "easy"
        ? "easy"
        : roomData.mode === "advanced"
          ? "advanced"
          : "basic";
    const storedTemplates = sessionStorage.getItem(
      `room_templates_${this.roomId}`,
    );
    if (storedTemplates) {
      try {
        this.customTemplates = JSON.parse(storedTemplates);
      } catch {}
    } else if (roomData.templates) {
      this.customTemplates = roomData.templates;
    }
    const lang = this.getActiveLanguageCode();
    this.refreshConceptDefinitionsForLanguage(lang);
    this.updateModeToggle();
    this.updateHeaderDesc();
    document.getElementById("lobby-bar").style.display = "flex";
    document.querySelector(".header").classList.add("has-lobby-bar");
    const linkEl = document.getElementById("lobby-link");
    if (linkEl) linkEl.value = this.joinLink;
    const modeToggle = document.querySelector(".mode-toggle");
    if (modeToggle) modeToggle.style.display = "none";
    document.querySelector(".leaderboard-btn").style.display = "none";
    document.getElementById("lobby-copy-btn").onclick = () =>
      this.copyRoomLink();
    document.getElementById("lobby-leaderboard-btn").onclick = () =>
      this.showLeaderboard();
    document.getElementById("lobby-download-btn").onclick = () =>
      this.downloadRoomResults();
    this.updateRoundDisplay();
    this.showRoomNameModal();
  }

  generateRoundSeed(round) {
    return `${this.baseSeed}-round-${round}`;
  }

  showRoomNameModal() {
    const modal = document.getElementById("room-name-modal");
    const input = document.getElementById("room-player-name");
    const startBtn = document.getElementById("room-name-start-btn");
    if (!modal || !input) return;
    input.value = "";
    modal.style.display = "flex";
    input.focus();
    const start = () => {
      this.playerName = (input.value && input.value.trim()) || "Anonymous";
      modal.style.display = "none";
      if (!localStorage.getItem(GUIDE_SEEN_KEY))
        setTimeout(() => this.showGuide(), 100);
      this.showLoading();
      requestAnimationFrame(async () => {
        try {
          await this.startNewGame();
        } finally {
          this.hideLoading();
        }
      });
    };
    startBtn.onclick = start;
    input.onkeydown = (e) => {
      if (e.key === "Enter") start();
    };
  }

  copyRoomLink() {
    if (!this.joinLink) return;
    navigator.clipboard.writeText(this.joinLink).then(() => {
      const btn = document.getElementById("lobby-copy-btn");
      const lang = this.language === "ru" ? "ru" : "en";
      const orig = UI_STRINGS.copyLink[lang];
      btn.textContent = UI_STRINGS.copied[lang];
      setTimeout(() => {
        btn.textContent = orig;
      }, 1500);
    });
  }

  async downloadRoomResults() {
    if (this.regime !== "lobby" || !this.roomId) return;
    try {
      const rows = await api.getRoomLeaderboard(this.roomId);
      const headers = ["Rank", "Name", "Score", "Time (s)", "Won", "Date"];
      const lines = [headers.join(",")];
      rows.forEach((r, i) => {
        const date = r.createdAt ? new Date(r.createdAt).toISOString() : "";
        lines.push(
          [
            i + 1,
            `"${(r.playerName || "").replace(/"/g, '""')}"`,
            r.score,
            r.timeSeconds,
            r.won ? "Yes" : "No",
            date,
          ].join(","),
        );
      });
      const blob = new Blob([lines.join("\n")], {
        type: "text/csv;charset=utf-8",
      });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `room-${this.roomId}-results.csv`;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e) {
      alert("Could not download: " + (e.message || ""));
    }
  }

  bindEvents() {
    document
      .getElementById("submit-btn")
      .addEventListener("click", () => this.submitGuess());
    document
      .getElementById("deselect-btn")
      .addEventListener("click", () => this.deselectAll());
    document
      .getElementById("new-game-btn")
      .addEventListener("click", () => this.startNewGame());
    const llmRunBtn = document.getElementById("llm-run-btn");
    if (llmRunBtn)
      llmRunBtn.addEventListener("click", () => this.runLlmBenchmark());
    const llmRefreshBtn = document.getElementById("llm-refresh-btn");
    if (llmRefreshBtn)
      llmRefreshBtn.addEventListener("click", () => this.loadLlmModels());
    const llmAgentBtn = document.getElementById("llm-agent-btn");
    if (llmAgentBtn)
      llmAgentBtn.addEventListener("click", () => this.runLlmAgent());
    const welcomeAgentBtn = document.getElementById("welcome-agent");
    if (welcomeAgentBtn)
      welcomeAgentBtn.addEventListener("click", () => this.startAgentGame());
    document
      .querySelector(".dictionary-btn")
      .addEventListener("click", () => this.showDictionary());
    document
      .querySelector(".guide-btn")
      .addEventListener("click", () => this.showGuide());
    document
      .querySelector(".hints-toggle-btn")
      .addEventListener("click", () => this.toggleHints());
    const hintBtn = document.querySelector(".category-hints-toggle-btn");
    if (hintBtn) {
      hintBtn.addEventListener("click", () => this.useHint());
    }
    document
      .querySelector(".leaderboard-btn")
      .addEventListener("click", () => this.showLeaderboard());
    document.querySelector(".guide-close-btn").addEventListener("click", () => {
      localStorage.setItem(GUIDE_SEEN_KEY, "1");
      document.getElementById("guide-modal").style.display = "none";
    });
    document
      .getElementById("win-save-btn")
      .addEventListener("click", () => this.closeWinAndSave());
    document
      .getElementById("win-next-btn")
      .addEventListener("click", () => this.nextRound());
    document
      .getElementById("leaderboard-close-btn")
      .addEventListener(
        "click",
        () =>
          (document.getElementById("leaderboard-modal").style.display = "none"),
      );
    document.getElementById("home-btn").addEventListener("click", () => {
      window.location = window.location.pathname || "/";
    });
    const dlBtn = document.getElementById("leaderboard-download-btn");
    if (dlBtn)
      dlBtn.addEventListener("click", () => this.downloadRoomResults());

    const langToggle = document.getElementById("lang-toggle");
    if (langToggle) {
      langToggle.addEventListener("click", () => this.toggleLanguage());
    }

    document.querySelectorAll(".mode-btn").forEach((btn) => {
      btn.addEventListener("click", (e) =>
        this.setGameMode(e.target.dataset.mode),
      );
    });

    document.addEventListener("click", (e) => {
      if (isTouchDevice()) return;
      if (!e.target.closest(".concept-tooltip")) {
        this.tooltipManager.forceHide();
      }
    });

    this.setupModalCloseOnBackdrop();
  }

  setupModalCloseOnBackdrop() {
    const modalIds = [
      "room-name-modal",
      "win-modal",
      "lose-modal",
      "dictionary-modal",
      "guide-modal",
      "leaderboard-modal",
    ];
    modalIds.forEach((modalId) => {
      const modal = document.getElementById(modalId);
      if (modal) {
        modal.addEventListener("click", (e) => {
          if (e.target === modal) {
            modal.style.display = "none";
          }
        });
        const content = modal.querySelector(".modal-content");
        if (content) {
          content.addEventListener("click", (e) => {
            e.stopPropagation();
          });
        }
      }
    });
  }

  setGameMode(mode) {
    if (this.gameMode === mode) return;
    this.gameMode = mode;
    this.updateModeToggle();
    this.updateHeaderDesc();
    this.startNewGame();
  }

  getEffectiveMode() {
    if (this.gameMode === "easy") return "easy";
    if (this.gameMode === "advanced") return "advanced";
    return "basic";
  }

  updateModeToggle() {
    document.querySelectorAll(".mode-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.mode === this.gameMode);
    });
    this.updateCategorySlotsVisibility();
  }

  updateHeaderDesc() {
    const p = document.querySelector(".header-desc");
    if (!p) return;
    if (this.getEffectiveMode() === "advanced") {
      p.textContent = UI_STRINGS.headerDescAdvanced[this.language];
    } else {
      p.textContent = UI_STRINGS.headerDescBasic[this.language];
    }
  }

  updateCategorySlotsVisibility() {
    const fourth = document.querySelector(".category-slot-4th");
    if (fourth)
      fourth.classList.toggle("hidden", this.getEffectiveMode() === "advanced");
    const slots = document.getElementById("category-slots");
    const effective = this.getEffectiveMode();
    slots.classList.toggle(
      "slots-4",
      effective === "basic" || effective === "easy",
    );
    slots.classList.toggle("slots-3", effective === "advanced");
  }

  toggleHints() {
    this.hintsEnabled = !this.hintsEnabled;
    this.updateHintsToggle();

    // If hints are disabled, hide any currently visible tooltip
    if (!this.hintsEnabled) {
      this.tooltipManager.forceHide();
    }

    // Recreate the game board to apply the new hints setting
    this.createGameBoard();
  }

  updateHintsToggle() {
    const hintsBtn = document.querySelector(".hints-toggle-btn");
    if (!hintsBtn) return;
    if (this.hintsEnabled) {
      hintsBtn.classList.add("on");
      hintsBtn.classList.remove("off");
      hintsBtn.innerHTML = UI_STRINGS.defsOn[this.language];
    } else {
      hintsBtn.classList.add("off");
      hintsBtn.classList.remove("on");
      hintsBtn.innerHTML = UI_STRINGS.defsOff[this.language];
    }
  }

  useHint() {
    if (this.hintsRemaining <= 0 || !this.currentPuzzle) return;
    const solvedConcepts = this.getSolvedConcepts();
    const unrevealed = this.currentPuzzle.board.filter(
      (concept) =>
        !this.hintRevealedConcepts.has(concept) && !solvedConcepts.has(concept),
    );
    if (unrevealed.length === 0) return;

    const chosen = unrevealed[Math.floor(Math.random() * unrevealed.length)];
    this.hintRevealedConcepts.add(chosen);
    this.hintsRemaining--;
    this.currentScore = Math.max(0, Math.round(this.currentScore - 200));
    this.updateScoreDisplay(this.currentScore);
    this.updateHintButton();
    this.updateHintRevealedOnBoard();
  }

  getSolvedConcepts() {
    const solved = new Set();
    const slots = document.querySelectorAll(".category-slot.filled");
    slots.forEach((slot) => {
      const difficulty = slot.dataset.difficulty;
      const category = this.currentPuzzle?.categories[difficulty];
      if (category) category.members.forEach((c) => solved.add(c));
    });
    return solved;
  }

  updateHintButton() {
    const btn = document.querySelector(".category-hints-toggle-btn");
    if (!btn) return;
    const lang = this.language === "ru" ? "ru" : "en";
    btn.textContent = UI_STRINGS.hintLabel[lang].replace(
      "%d",
      String(this.hintsRemaining),
    );
    btn.disabled = this.hintsRemaining <= 0;
    btn.classList.toggle("on", this.hintsRemaining > 0);
    btn.classList.toggle("off", this.hintsRemaining <= 0);
  }

  updateHintRevealedOnBoard() {
    const cards = document.querySelectorAll(".concept-card");
    cards.forEach((card) => {
      const concept = card.dataset.concept;
      const categoryName = this.conceptCategoryMap[concept];
      if (this.hintRevealedConcepts.has(concept) && categoryName) {
        card.dataset.hint = categoryName;
      }
    });
  }

  async startNewGame() {
    this.closeAllModals();
    this.stopTimer();
    this.timerStart = Date.now();
    this.elapsedSeconds = 0;
    this.currentScore = 0;
    this.llmBenchmark = null;
    this.updateTimerDisplay(0);
    this.updateScoreDisplay(null);
    this.resetLlmBenchmarkPanel();
    if (this.regime === "lobby" && this.puzzleSeed === null) {
      this.puzzleSeed = this.generateRoundSeed(this.currentRound);
    }
    const seed = this.regime === "lobby" ? this.puzzleSeed : null;
    const lang = this.getActiveLanguageCode();
    const templates = this.getTemplatesForLanguage(lang);
    // Sampling is server-side so the same CP-SAT implementation serves
    // solo games, room seeds, and external callers.
    let sampledPuzzle = null;
    let lastError = null;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const response = await api.sampleGame({
          mode: this.getEffectiveMode(),
          config: templates,
          seed,
        });
        sampledPuzzle = response.game;
        if (
          sampledPuzzle &&
          Array.isArray(sampledPuzzle.board) &&
          sampledPuzzle.board.length === 16 &&
          sampledPuzzle.categories &&
          typeof sampledPuzzle.categories === "object"
        ) {
          break;
        }
      } catch (err) {
        lastError = err;
      }
      await new Promise((resolve) => setTimeout(resolve, 180));
    }

    if (
      !sampledPuzzle ||
      !Array.isArray(sampledPuzzle.board) ||
      sampledPuzzle.board.length !== 16
    ) {
      console.error(
        "Server sampler failed to produce a valid puzzle.",
        lastError,
      );
      const details = lastError?.message
        ? `\nDetails: ${lastError.message}`
        : "";
      alert(
        `Could not generate a uniquely solvable puzzle. Please try again.${details}`,
      );
      return;
    }

    this.currentPuzzle = sampledPuzzle;
    this.updateDictionaryEntries();
    this.selectedConcepts = [];
    this.mistakes = 0;
    this.solvedCategories = 0;
    this.hintsRemaining = 4;
    this.hintRevealedConcepts = new Set();
    this.lastTappedForTooltip = null;
    this.lastGuessTimestamp = null;

    // Build a quick lookup from concept to its category name (if any)
    this.conceptCategoryMap = {};
    for (const category of Object.values(this.currentPuzzle.categories)) {
      category.members.forEach((concept) => {
        this.conceptCategoryMap[concept] = category.name;
      });
    }

    this.updateCategorySlotsVisibility();
    this.updateMistakesDisplay();
    this.updateHintButton();
    this.resetCategorySlots();
    this.createGameBoard();
    this.updateSubmitButton();
    this.startTimer();
    this.updateRoundDisplay();

    this.dogAnimations.updateDogMood("happy");
  }

  updateRoundDisplay() {
    const roundEl = document.getElementById("round-display");
    if (!roundEl) return;
    if (this.regime === "lobby" && this.totalRounds) {
      const total = Number(this.totalRounds) || 1;
      const lang = this.language === "ru" ? "ru" : "en";
      roundEl.style.display = "block";
      roundEl.textContent = UI_STRINGS.roundOf[lang]
        .replace("%1", String(this.currentRound))
        .replace("%2", String(total));
    } else {
      roundEl.style.display = "none";
    }
  }

  startTimer() {
    this.timerStart = Date.now();
    this.timerInterval = setInterval(() => {
      this.elapsedSeconds = Math.floor((Date.now() - this.timerStart) / 1000);
      this.updateTimerDisplay(this.elapsedSeconds);
    }, 1000);
  }

  stopTimer() {
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
    if (this.timerStart) {
      this.elapsedSeconds = Math.floor((Date.now() - this.timerStart) / 1000);
    }
  }

  updateTimerDisplay(seconds) {
    const el = document.getElementById("timer-display");
    if (!el) return;
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    el.textContent = `${m}:${s.toString().padStart(2, "0")}`;
  }

  updateScoreDisplay(score) {
    const el = document.getElementById("score-display");
    if (!el) return;
    const lang = this.language === "ru" ? "ru" : "en";
    el.textContent =
      score === null
        ? UI_STRINGS.scoreEmpty[lang]
        : UI_STRINGS.scoreWithNumber[lang].replace("%d", String(score));
    this.updateLlmHumanScore();
  }

  computeScore(won) {
    // Legacy helper retained for compatibility; scoring is now applied
    // incrementally inside submitGuess based on time between guesses.
    return Math.max(0, Math.round(this.currentScore));
  }

  createGameBoard() {
    const gameBoard = document.getElementById("game-board");
    const solvedConcepts = this.getSolvedConcepts();
    gameBoard.innerHTML = "";

    this.currentPuzzle.board.forEach((concept) => {
      const card = document.createElement("div");
      card.className = "concept-card";
      card.textContent = concept;
      card.dataset.concept = concept;

      const categoryName = this.conceptCategoryMap[concept];
      if (this.hintRevealedConcepts.has(concept) && categoryName) {
        card.dataset.hint = categoryName;
      }
      if (this.selectedConcepts.includes(concept)) {
        card.classList.add("selected");
      }
      if (solvedConcepts.has(concept)) {
        card.classList.add("correct");
      }

      // Hover tooltips when hints are enabled. On touch devices, hide() is a no-op,
      // so the tooltip will not immediately disappear on tap.
      if (this.hintsEnabled) {
        card.addEventListener("mouseenter", () => {
          if (
            !card.classList.contains("correct") &&
            !card.classList.contains("incorrect")
          ) {
            this.tooltipManager.show(concept, card);
          }
        });
        card.addEventListener("mouseleave", () => this.tooltipManager.hide());
      }

      card.addEventListener("click", (e) =>
        this.handleCardClick(concept, card, e),
      );
      gameBoard.appendChild(card);
    });
  }

  handleCardClick(concept, cardElement, e) {
    if (cardElement.classList.contains("correct")) return;

    const isTouch =
      isTouchDevice() ||
      (e && (e.pointerType === "touch" || (e.touches && e.touches.length > 0)));

    if (
      isTouch &&
      this.hintsEnabled &&
      !this.selectedConcepts.includes(concept)
    ) {
      if (this.selectedConcepts.length === CONFIG.CONCEPTS_PER_GROUP - 1) {
        this.tooltipManager.forceHide();
        this.lastTappedForTooltip = null;
        this.toggleConcept(concept, cardElement);
        return;
      }
      if (this.lastTappedForTooltip === cardElement) {
        const now = Date.now();
        if (
          this._lastTooltipTapTime != null &&
          now - this._lastTooltipTapTime < 300
        ) {
          return;
        }
        this.lastTappedForTooltip = null;
        this._lastTooltipTapTime = null;
        this.tooltipManager.forceHide();
        this.toggleConcept(concept, cardElement);
      } else {
        this.lastTappedForTooltip = cardElement;
        this._lastTooltipTapTime = Date.now();
        this.tooltipManager.show(concept, cardElement);
        this.tooltipManager.makeSticky();
        return;
      }
    } else {
      this.toggleConcept(concept, cardElement);
    }
  }

  toggleConcept(concept, cardElement) {
    if (cardElement.classList.contains("correct")) return;
    this.tooltipManager.forceHide();
    this.lastTappedForTooltip = null;

    const incorrectCards = Array.from(
      document.querySelectorAll(".concept-card.incorrect:not(.correct)"),
    );
    if (incorrectCards.length > 0) {
      this.fadeIncorrectTiles(incorrectCards);
    }

    const index = this.selectedConcepts.indexOf(concept);

    if (index > -1) {
      this.selectedConcepts.splice(index, 1);
      cardElement.classList.remove("selected");
      if (
        cardElement.classList.contains("incorrect") &&
        !cardElement.classList.contains("correct")
      ) {
        cardElement.classList.remove("incorrect");
      }
    } else {
      if (this.selectedConcepts.length < CONFIG.CONCEPTS_PER_GROUP) {
        this.selectedConcepts.push(concept);
        cardElement.classList.add("selected");
        if (cardElement.classList.contains("incorrect")) {
          cardElement.classList.remove("incorrect");
        }
      }
    }

    this.updateSubmitButton();

    if (this.selectedConcepts.length === CONFIG.CONCEPTS_PER_GROUP) {
      this.dogAnimations.updateDogMood("excited");
    } else {
      this.dogAnimations.updateDogMood("happy");
    }
  }

  fadeIncorrectTiles(cards) {
    cards.forEach((card) => {
      card.classList.add("incorrect-fading");
    });
    setTimeout(() => {
      cards.forEach((card) => {
        card.classList.remove("incorrect", "incorrect-fading");
      });
    }, 400);
  }

  updateSubmitButton() {
    const submitBtn = document.getElementById("submit-btn");
    submitBtn.disabled =
      this.selectedConcepts.length !== CONFIG.CONCEPTS_PER_GROUP;
  }

  submitGuess() {
    if (this.selectedConcepts.length !== CONFIG.CONCEPTS_PER_GROUP) return;

    // Time since previous guess (or since game start for the first guess)
    const now = Date.now();
    let secondsSinceLastGuess;
    if (this.lastGuessTimestamp) {
      secondsSinceLastGuess = (now - this.lastGuessTimestamp) / 1000;
    } else if (this.timerStart) {
      secondsSinceLastGuess = (now - this.timerStart) / 1000;
    } else {
      secondsSinceLastGuess = 0;
    }
    // Avoid division by zero and negative/NaN values
    if (!Number.isFinite(secondsSinceLastGuess) || secondsSinceLastGuess <= 0) {
      secondsSinceLastGuess = 0.1;
    }
    this.lastGuessTimestamp = now;

    // Keep elapsed time in sync with scoring
    this.elapsedSeconds = Math.floor((now - this.timerStart) / 1000);
    this.updateTimerDisplay(this.elapsedSeconds);

    // Dog gets worried during submission
    this.dogAnimations.updateDogMood("worried");

    let matchedCategory = null;
    let categoryDifficulty = null;

    for (const [difficulty, category] of Object.entries(
      this.currentPuzzle.categories,
    )) {
      const categorySet = new Set(category.members);
      const selectedSet = new Set(this.selectedConcepts);

      if (this.setsAreEqual(categorySet, selectedSet)) {
        matchedCategory = category;
        categoryDifficulty = difficulty;
        break;
      }
    }

    if (matchedCategory) {
      Sounds.correct();
      this.markGroupAsCorrect(this.selectedConcepts);
      this.fillCategorySlot(matchedCategory, categoryDifficulty);
      this.solvedCategories++;

      // Scoring for a correct category: base + speed bonus, always at least 100
      const rawPoints = 12000 / secondsSinceLastGuess - this.elapsedSeconds;
      const pointsForThisCategory = Math.max(100, Math.round(rawPoints));
      this.currentScore += pointsForThisCategory;
      this.currentScore = Math.max(0, Math.round(this.currentScore));
      this.updateScoreDisplay(this.currentScore);

      this.dogAnimations.updateDogMood("celebrating");

      if (this.solvedCategories === this.currentPuzzle.numCategories) {
        this.stopTimer();
        setTimeout(() => {
          Sounds.win();
          this.showWinModal();
          this.dogAnimations.updateDogMood("happy");
        }, 2000);
      } else {
        setTimeout(() => {
          this.dogAnimations.updateDogMood("happy");
        }, 2000);
      }
    } else {
      Sounds.wrong();
      this.mistakes++;
      this.updateMistakesDisplay();
      this.markGroupAsIncorrect(this.selectedConcepts);

      // Scoring for a mistake:
      // - (seconds since previous guess)
      this.currentScore -= secondsSinceLastGuess;
      this.currentScore = Math.max(0, Math.round(this.currentScore));
      this.updateScoreDisplay(this.currentScore);

      // Dog gets progressively more sad with each mistake
      if (this.mistakes === 1) {
        this.dogAnimations.updateDogMood("worried");
        this.dogAnimations.spawnBones();
      }
      if (this.mistakes === 2) {
        this.dogAnimations.updateDogMood("concerned");
        this.dogAnimations.spawnBones();
      }
      if (this.mistakes === 3) {
        this.dogAnimations.updateDogMood("sad");
        this.dogAnimations.spawnBones();
      }

      if (this.mistakes === CONFIG.MAX_MISTAKES) {
        this.stopTimer();
        this.saveResult(false);
        setTimeout(() => {
          Sounds.lose();
          this.showLoseModal();
          this.dogAnimations.updateDogMood("sad");
        }, 1500);
      } else {
        // Return to concerned after a brief period
        setTimeout(() => {
          if (this.mistakes > 0) {
            this.dogAnimations.updateDogMood("concerned");
          }
        }, 1000);
      }
    }

    this.selectedConcepts = [];
    this.updateSubmitButton();
  }

  markGroupAsCorrect(concepts) {
    concepts.forEach((concept) => {
      const card = this.findCardByConcept(concept);
      if (card) {
        card.classList.add("correct");
        card.classList.remove("selected", "incorrect");
      }
    });
  }

  markGroupAsIncorrect(concepts) {
    concepts.forEach((concept) => {
      const card = this.findCardByConcept(concept);
      if (card && !card.classList.contains("correct")) {
        card.classList.add("incorrect");
        card.classList.remove("selected");
        card.classList.add("shake");
        setTimeout(() => card.classList.remove("shake"), 500);
      }
    });
  }

  fillCategorySlot(category, difficulty) {
    const slot = document.querySelector(
      `.category-slot[data-difficulty="${difficulty}"]`,
    );
    if (slot) {
      slot.classList.add("filled");
      slot.querySelector(".category-name").textContent = category.name;
      slot.querySelector(".category-concepts").textContent =
        category.members.join(", ");
    }
  }

  findCardByConcept(concept) {
    const cards = document.querySelectorAll(".concept-card");
    for (const card of cards) {
      if (card.textContent === concept) {
        return card;
      }
    }
    return null;
  }

  setsAreEqual(setA, setB) {
    return setA.size === setB.size && [...setA].every((item) => setB.has(item));
  }

  updateMistakesDisplay() {
    const mistakeDots = document.querySelectorAll(".mistake-dot");
    mistakeDots.forEach((dot, index) => {
      dot.classList.toggle("used", index < this.mistakes);
    });
  }

  deselectAll() {
    this.selectedConcepts = [];
    const cards = document.querySelectorAll(".concept-card");
    cards.forEach((card) => {
      card.classList.remove("selected");
      // Remove incorrect styling when deselecting all
      if (
        card.classList.contains("incorrect") &&
        !card.classList.contains("correct")
      ) {
        card.classList.remove("incorrect");
      }
    });
    this.updateSubmitButton();

    // Dog becomes HAPPY when deselecting
    this.dogAnimations.updateDogMood("happy");
  }

  resetLlmBenchmarkPanel() {
    const status = document.getElementById("llm-status-pill");
    const trace = document.getElementById("llm-trace");
    const runBtn = document.getElementById("llm-run-btn");
    if (status) {
      status.textContent = "Idle";
      status.className = "llm-status-pill";
    }
    if (trace) trace.innerHTML = "<p>No model run yet.</p>";
    if (runBtn) runBtn.disabled = false;
    this.setText("llm-score-value", "—");
    this.setText("llm-match-value", "—");
    this.setText("llm-delta-value", "—");
    this.updateLlmHumanScore();
  }

  updateLlmHumanScore() {
    const humanScore = this.currentPuzzle
      ? String(this.currentScore || 0)
      : "—";
    this.setText("llm-human-score-value", humanScore);
    if (this.llmBenchmark && Number.isFinite(this.llmBenchmark.score)) {
      this.setText(
        "llm-delta-value",
        String(this.llmBenchmark.score - (this.currentScore || 0)),
      );
    }
  }

  setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  setLlmStatus(text, state = "") {
    const status = document.getElementById("llm-status-pill");
    if (!status) return;
    status.textContent = text;
    status.className = state ? `llm-status-pill ${state}` : "llm-status-pill";
  }

  async loadLlmModels() {
    const refreshBtn = document.getElementById("llm-refresh-btn");
    const input = document.getElementById("llm-model-input");
    if (!input) return;
    if (refreshBtn) refreshBtn.disabled = true;
    try {
      const data = await api.getLlmModels();
      const models = Array.isArray(data.models)
        ? [
            ...new Set(
              data.models
                .map((model) => String(model || "").trim())
                .filter(Boolean),
            ),
          ]
        : [];
      const currentValue = input.value || data.defaultModel || "llama3.2:3b";
      input.innerHTML = "";
      if (currentValue && !models.includes(currentValue)) {
        const currentOption = document.createElement("option");
        currentOption.value = currentValue;
        currentOption.textContent = currentValue;
        input.appendChild(currentOption);
      }
      models.forEach((model) => {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        input.appendChild(option);
      });
      if (models.includes(currentValue)) {
        input.value = currentValue;
      } else if (data.defaultModel && models.includes(data.defaultModel)) {
        input.value = data.defaultModel;
      }
      const agentInput = document.getElementById("agent-model-input");
      if (agentInput) {
        agentInput.innerHTML = input.innerHTML;
        agentInput.value = input.value;
      }
      if (models.length > 0) {
        this.setLlmStatus(`${models.length} models`, "ready");
      }
    } catch (_) {
      // Keep manual model entry available if the package is not installed locally yet.
      this.setLlmStatus("Models unavailable", "error");
    } finally {
      if (refreshBtn) refreshBtn.disabled = false;
    }
  }

  escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  scoreModelGuesses(guesses) {
    const solvedDifficulties = new Set();
    const rows = [];
    let mistakes = 0;
    let solved = 0;
    let score = 0;
    let elapsed = 0;
    const secondsPerGuess = 8;

    for (const guess of guesses) {
      if (
        mistakes >= CONFIG.MAX_MISTAKES ||
        solved >= this.currentPuzzle.numCategories
      )
        break;
      elapsed += secondsPerGuess;
      const guessSet = new Set(guess.items);
      let matchedDifficulty = null;
      let matchedCategory = null;

      for (const [difficulty, category] of Object.entries(
        this.currentPuzzle.categories,
      )) {
        if (solvedDifficulties.has(difficulty)) continue;
        if (this.setsAreEqual(new Set(category.members), guessSet)) {
          matchedDifficulty = difficulty;
          matchedCategory = category;
          break;
        }
      }

      if (matchedCategory) {
        solvedDifficulties.add(matchedDifficulty);
        solved++;
        const rawPoints = 12000 / secondsPerGuess - elapsed;
        score = Math.max(
          0,
          Math.round(score + Math.max(100, Math.round(rawPoints))),
        );
        rows.push({
          ...guess,
          correct: true,
          matchedName: matchedCategory.name,
          difficulty: matchedDifficulty,
        });
      } else {
        mistakes++;
        score = Math.max(0, Math.round(score - secondsPerGuess));
        rows.push({
          ...guess,
          correct: false,
          matchedName: null,
          difficulty: null,
        });
      }
    }

    return {
      score,
      solved,
      mistakes,
      elapsed,
      won:
        solved === this.currentPuzzle.numCategories &&
        mistakes < CONFIG.MAX_MISTAKES,
      rows,
    };
  }

  renderLlmBenchmark(result, scored) {
    const trace = document.getElementById("llm-trace");
    if (!trace) return;
    const rows = scored.rows
      .map((row, index) => {
        const label = this.escapeHtml(row.label || `Guess ${index + 1}`);
        const verdict = row.correct ? `Matched ${row.matchedName}` : "Miss";
        const items = this.escapeHtml(row.items.join(", "));
        return `
                <div class="llm-guess ${row.correct ? "correct" : "incorrect"}">
                    <div class="llm-guess-title">${label}<span>${this.escapeHtml(verdict)}</span></div>
                    <div>${items}</div>
                </div>
            `;
      })
      .join("");
    const cadence =
      '<p class="mono-label">Scored with the live category matcher at 8 seconds per model guess.</p>';
    trace.innerHTML =
      rows || "<p>The model returned no valid four-term guesses.</p>";
    trace.insertAdjacentHTML(
      "beforeend",
      cadence + this.renderLlmReasoning(result),
    );
  }

  renderLlmReasoning(result) {
    const reasoning = result.reasoning || result.notes || result.raw || "";
    if (!reasoning) return "";
    return `
            <details class="llm-reasoning-panel" open>
                <summary>Reasoning / verification trace</summary>
                <pre>${this.escapeHtml(reasoning)}</pre>
            </details>
        `;
  }

  renderAgentTrace(rows, mistakes, thinking) {
    const trace = document.getElementById("llm-trace");
    if (!trace) return;
    const lines = rows.map((row, index) => {
      const label = this.escapeHtml(row.label || `Guess ${index + 1}`);
      const verdict = row.correct
        ? `Matched ${this.escapeHtml(row.matchedName)}`
        : "Miss";
      const items = this.escapeHtml((row.items || []).join(", "));
      return `
        <div class="llm-guess ${row.correct ? "correct" : "incorrect"}">
          <div class="llm-guess-title">${label}<span>${verdict}</span></div>
          <div>${items}</div>
        </div>
      `;
    });
    if (thinking) {
      lines.push(
        '<div class="llm-guess"><div class="llm-guess-title">thinking…</div></div>',
      );
    }
    const header = `<p class="mono-label">Mistakes ${mistakes}/${CONFIG.MAX_MISTAKES}</p>`;
    trace.innerHTML = header + (lines.join("") || "<p>Agent is starting…</p>");
  }

  async runLlmAgent() {
    if (!this.currentPuzzle || this.isLlmRunning) return;
    const panel = document.getElementById("llm-bench-panel");
    if (panel) panel.open = true;
    const agentBtn = document.getElementById("llm-agent-btn");
    const runBtn = document.getElementById("llm-run-btn");
    const modelInput = document.getElementById("llm-model-input");
    const model =
      this.agentModel ||
      (modelInput && modelInput.value && modelInput.value.trim()) ||
      "llama3.2:3b";

    this.isLlmRunning = true;
    if (agentBtn) agentBtn.disabled = true;
    if (runBtn) runBtn.disabled = true;

    const numCategories = this.currentPuzzle.numCategories;
    const solvedDifficulties = new Set();
    const history = [];
    const rows = [];
    let mistakes = 0;
    let solved = 0;
    let score = 0;
    let elapsed = 0;
    const secondsPerGuess = 8;

    try {
      while (mistakes < CONFIG.MAX_MISTAKES && solved < numCategories) {
        this.setLlmStatus(`solving · turn ${rows.length + 1}`, "running");
        this.renderAgentTrace(rows, mistakes, true);

        const remaining = remainingTerms(
          this.currentPuzzle,
          solvedDifficulties,
        );
        let guess = null;
        try {
          const res = await api.runLlmGuess({
            remaining,
            numCategories,
            history,
            model,
          });
          guess = res && res.guess ? res.guess : null;
        } catch (_) {
          guess = null;
        }
        elapsed += secondsPerGuess;

        if (!guess || !Array.isArray(guess.items) || guess.items.length !== 4) {
          mistakes++;
          score = Math.max(0, Math.round(score - secondsPerGuess));
          rows.push({
            label: "Unusable output",
            items: guess && Array.isArray(guess.items) ? guess.items : [],
            correct: false,
            matchedName: null,
          });
          this.renderAgentTrace(rows, mistakes, false);
          continue;
        }

        const verdict = evaluateGuess(this.currentPuzzle, guess.items);
        if (verdict.correct) {
          solvedDifficulties.add(verdict.difficulty);
          solved++;
          const rawPoints = 12000 / secondsPerGuess - elapsed;
          score = Math.max(
            0,
            Math.round(score + Math.max(100, Math.round(rawPoints))),
          );
          rows.push({
            label: guess.label || verdict.matchedCategory.name,
            items: guess.items,
            correct: true,
            matchedName: verdict.matchedCategory.name,
          });
        } else {
          mistakes++;
          history.push({ items: guess.items });
          score = Math.max(0, Math.round(score - secondsPerGuess));
          rows.push({
            label: guess.label || "Guess",
            items: guess.items,
            correct: false,
            matchedName: null,
          });
        }
        this.renderAgentTrace(rows, mistakes, false);
      }

      const won = solved === numCategories && mistakes < CONFIG.MAX_MISTAKES;
      this.llmBenchmark = { score, solved, mistakes, won, rows };
      this.setText("llm-score-value", String(score));
      this.setText("llm-match-value", `${solved}/${numCategories}`);
      this.updateLlmHumanScore();
      this.setLlmStatus(
        won ? "Solved" : "Out of mistakes",
        won ? "ready" : "error",
      );
    } finally {
      this.isLlmRunning = false;
      if (agentBtn) agentBtn.disabled = false;
      if (runBtn) runBtn.disabled = false;
      this.agentModel = null;
    }
  }

  async runLlmBenchmark() {
    if (!this.currentPuzzle || this.isLlmRunning) return;
    const runBtn = document.getElementById("llm-run-btn");
    const modelInput = document.getElementById("llm-model-input");
    const model =
      (modelInput && modelInput.value && modelInput.value.trim()) ||
      "llama3.2:3b";
    this.isLlmRunning = true;
    if (runBtn) runBtn.disabled = true;
    this.setLlmStatus("Running", "running");
    const trace = document.getElementById("llm-trace");
    if (trace)
      trace.innerHTML = "<p>Sending the current board to the model solver…</p>";
    const panel = document.getElementById("llm-bench-panel");
    if (panel) panel.open = true;

    try {
      const result = await api.runLlmSolver({
        board: this.currentPuzzle.board,
        numCategories: this.currentPuzzle.numCategories,
        model,
        maxGuesses: Math.max(
          6,
          this.currentPuzzle.numCategories + CONFIG.MAX_MISTAKES,
        ),
      });
      const scored = this.scoreModelGuesses(result.guesses || []);
      this.llmBenchmark = scored;
      this.setText("llm-score-value", String(scored.score));
      this.setText(
        "llm-match-value",
        `${scored.solved}/${this.currentPuzzle.numCategories}`,
      );
      this.updateLlmHumanScore();
      this.renderLlmBenchmark(result, scored);
      this.setLlmStatus(scored.won ? "Solved" : "Scored", "ready");
    } catch (error) {
      let message = error.message || "Model solver failed.";
      try {
        const parsed = JSON.parse(message);
        message = parsed.error || parsed.setup || message;
      } catch (_) {}
      if (trace) trace.innerHTML = `<p>${this.escapeHtml(message)}</p>`;
      this.setLlmStatus("Error", "error");
    } finally {
      this.isLlmRunning = false;
      if (runBtn) runBtn.disabled = false;
    }
  }

  showDictionary() {
    const dictionaryModal = document.getElementById("dictionary-modal");
    const conceptList = document.getElementById("dictionary-list");

    conceptList.innerHTML = "";

    const entries =
      Array.isArray(this.dictionaryEntries) && this.dictionaryEntries.length > 0
        ? this.dictionaryEntries
        : Object.entries(CONCEPT_DEFINITIONS).map(([name, description]) => ({
            name,
            description,
          }));

    entries.forEach(({ name: concept, description: definition }) => {
      const item = document.createElement("div");
      item.className = "concept-item";

      const name = document.createElement("div");
      name.className = "concept-name";
      name.textContent = concept;

      const desc = document.createElement("div");
      desc.className = "concept-description";
      desc.textContent = definition;

      item.appendChild(name);
      item.appendChild(desc);
      conceptList.appendChild(item);
    });

    dictionaryModal.style.display = "flex";
  }

  updateDictionaryEntries() {
    const lang = this.getActiveLanguageCode();
    const templates = this.getTemplatesForLanguage(lang);
    if (templates) {
      this.dictionaryEntries = templates.map((entry) => ({
        name: entry.name,
        description: entry.description,
      }));
    } else {
      this.dictionaryEntries = Object.entries(CONCEPT_DEFINITIONS).map(
        ([name, description]) => ({ name, description }),
      );
    }
  }

  showGuide() {
    document.getElementById("guide-modal").style.display = "flex";
  }

  showWinModal() {
    const statsEl = document.getElementById("win-stats");
    const promptEl = document.getElementById("win-name-prompt");
    const roundInfoEl = document.getElementById("win-round-info");
    const saveBtn = document.getElementById("win-save-btn");
    const nextBtn = document.getElementById("win-next-btn");
    if (statsEl)
      statsEl.textContent = `Time: ${Math.floor(this.elapsedSeconds / 60)}:${(this.elapsedSeconds % 60).toString().padStart(2, "0")} — Score: ${this.currentScore}`;
    if (promptEl) promptEl.style.display = "none";
    if (this.regime !== "lobby")
      document.getElementById("player-name").value = "";

    const isLobby = this.regime === "lobby";
    const total = this.totalRounds ? Number(this.totalRounds) : 1;

    if (isLobby) {
      // Lobby mode: always show round info.
      if (roundInfoEl) {
        roundInfoEl.style.display = "block";
        roundInfoEl.textContent = `Round ${this.currentRound} of ${total}`;
      }
      // Hide save button in lobby; use the secondary button for next/close.
      if (saveBtn) saveBtn.style.display = "none";
      if (nextBtn) {
        nextBtn.style.display = "inline-block";
        nextBtn.textContent =
          total > 1 && this.currentRound < total ? "Next game" : "Close";
      }
    } else {
      // Solo play: no round info, show save button as before
      if (roundInfoEl) roundInfoEl.style.display = "none";
      if (saveBtn) {
        saveBtn.style.display = "inline-block";
        saveBtn.textContent = "Save & Play Again";
      }
      if (nextBtn) nextBtn.style.display = "none";
    }
    document.getElementById("win-modal").style.display = "flex";
  }

  async closeWinAndSave() {
    if (this.regime === "solo" && this._savingWinResult) return;
    const nameInput = document.getElementById("player-name");
    const baseName =
      (nameInput && nameInput.value && nameInput.value.trim()) || "Anonymous";
    const name =
      this.regime === "lobby" ? this.playerName || baseName : baseName;
    const saveBtn = document.getElementById("win-save-btn");
    if (this.regime === "solo") {
      this._savingWinResult = true;
      if (saveBtn) saveBtn.disabled = true;
    }
    try {
      if (this.regime === "lobby") {
        const nameWithRound = `${name} (R${this.currentRound})`;
        await api.submitRoomResult(this.roomId, {
          playerName: nameWithRound,
          score: this.currentScore,
          timeSeconds: this.elapsedSeconds,
          won: true,
          roundNumber: this.currentRound,
        });
      }
    } finally {
      if (this.regime === "solo") {
        this._savingWinResult = false;
        if (saveBtn) saveBtn.disabled = false;
      }
    }
    document.getElementById("win-modal").style.display = "none";
    document.getElementById("win-name-prompt").style.display = "none";
    this.startNewGame();
  }

  async nextRound() {
    if (this._savingRoundResult) return;
    this._savingRoundResult = true;
    const nextBtn = document.getElementById("win-next-btn");
    if (nextBtn) nextBtn.disabled = true;
    const name = this.playerName || "Anonymous";
    const nameWithRound = `${name} (R${this.currentRound})`;
    try {
      await api.submitRoomResult(this.roomId, {
        playerName: nameWithRound,
        score: this.currentScore,
        timeSeconds: this.elapsedSeconds,
        won: true,
        roundNumber: this.currentRound,
      });
    } catch (e) {
      console.error("Failed to submit round result:", e);
    } finally {
      this._savingRoundResult = false;
      if (nextBtn) nextBtn.disabled = false;
    }
    if (this.currentRound < this.totalRounds) {
      // Advance to next round and start a new puzzle.
      this.currentRound++;
      localStorage.setItem(
        `room_round_${this.roomId}`,
        this.currentRound.toString(),
      );
      this.puzzleSeed = this.generateRoundSeed(this.currentRound);
      document.getElementById("win-modal").style.display = "none";
      this.updateRoundDisplay();
      this.startNewGame();
    } else {
      // Final round: just close the modal, stay on the finished puzzle.
      document.getElementById("win-modal").style.display = "none";
    }
  }

  async getLeaderboard() {
    if (this.regime === "lobby" && this.roomId) {
      try {
        return await api.getRoomLeaderboard(this.roomId);
      } catch {
        return [];
      }
    }
    return [];
  }

  saveResult(won) {
    const submit = () => {
      if (this.regime === "lobby" && this.roomId) {
        const baseName = this.playerName || "Anonymous";
        const nameWithRound = `${baseName} (R${this.currentRound})`;
        api
          .submitRoomResult(this.roomId, {
            playerName: nameWithRound,
            score: this.currentScore,
            timeSeconds: this.elapsedSeconds,
            won: false,
            roundNumber: this.currentRound,
          })
          .catch(() => {});
      }
    };
    submit();
    try {
      const key = "ai_connections_recent";
      const entry = {
        won,
        score: this.currentScore,
        time: this.elapsedSeconds,
        mode: this.gameMode,
        date: new Date().toISOString(),
      };
      let recent = [];
      try {
        const raw = localStorage.getItem(key);
        if (raw) recent = JSON.parse(raw);
      } catch {}
      recent.unshift(entry);
      recent = recent.slice(0, 10);
      localStorage.setItem(key, JSON.stringify(recent));
    } catch (_) {}
  }

  async showLeaderboard() {
    if (this.regime !== "lobby" || !this.roomId) return;
    const container = document.getElementById("leaderboard-list");
    const hintEl = document.getElementById("leaderboard-hint");
    const downloadBtn = document.getElementById("leaderboard-download-btn");
    if (!container) return;
    container.innerHTML = "<p>Loading…</p>";
    const list = await this.getLeaderboard();
    container.innerHTML = "";
    if (hintEl) {
      hintEl.textContent = "Room results (same puzzle)";
    }
    if (downloadBtn) {
      downloadBtn.style.display =
        this.regime === "lobby" ? "inline-block" : "none";
    }
    if (list.length === 0) {
      container.innerHTML =
        '<p class="leaderboard-hint">No scores yet. Win a game and enter your name to appear here!</p>';
    } else {
      list.forEach((entry, i) => {
        const div = document.createElement("div");
        div.className = "leaderboard-item" + (i < 3 ? ` rank-${i + 1}` : "");
        const time = entry.timeSeconds != null ? entry.timeSeconds : entry.time;
        const name = entry.playerName != null ? entry.playerName : entry.name;
        const timeStr = `${Math.floor(time / 60)}:${(time % 60).toString().padStart(2, "0")}`;
        const wonStr = entry.won !== undefined ? (entry.won ? " ✓" : "") : "";
        // Lobby player names include the round label, e.g. "Alice (R2)".
        div.innerHTML = `<span><strong>#${i + 1}</strong> ${name}</span><span>${entry.score} pts · ${timeStr}${wonStr}</span>`;
        container.appendChild(div);
      });
    }
    document.getElementById("leaderboard-modal").style.display = "flex";
  }

  showLoseModal() {
    // Reveal the correct groups on the main board
    this.revealSolutionOnBoard();

    const solutionContainer = document.getElementById("solution-container");
    solutionContainer.innerHTML = "";

    for (const [difficulty, category] of Object.entries(
      this.currentPuzzle.categories,
    )) {
      const categoryDiv = document.createElement("div");
      categoryDiv.className = `solution-category ${difficulty}`;

      const badge = document.createElement("span");
      badge.className = `difficulty-badge ${difficulty}-badge`;
      badge.textContent = difficulty.toUpperCase();

      const name = document.createElement("div");
      name.className = "category-name";
      name.textContent = category.name;

      const concepts = document.createElement("div");
      concepts.className = "category-concepts";
      concepts.textContent = category.members.join(", ");

      categoryDiv.appendChild(badge);
      categoryDiv.appendChild(name);
      categoryDiv.appendChild(concepts);
      solutionContainer.appendChild(categoryDiv);
    }

    document.getElementById("lose-modal").style.display = "flex";
    // Dog stays sad for lose modal
  }

  revealSolutionOnBoard() {
    if (!this.currentPuzzle) return;
    const boardEl = document.getElementById("game-board");
    if (!boardEl) return;

    boardEl.innerHTML = "";

    const used = new Set();

    // Render each solved category as a row of four green cards
    for (const category of Object.values(this.currentPuzzle.categories)) {
      category.members.forEach((concept) => {
        used.add(concept);
        const card = document.createElement("div");
        card.className = "concept-card correct";
        card.textContent = concept;
        boardEl.appendChild(card);
      });
    }

    // Any leftover concepts (decoys) go at the end
    const leftovers = this.currentPuzzle.board.filter((c) => !used.has(c));
    leftovers.forEach((concept) => {
      const card = document.createElement("div");
      card.className = "concept-card";
      card.textContent = concept;
      boardEl.appendChild(card);
    });
  }

  closeAllModals() {
    document.getElementById("win-modal").style.display = "none";
    document.getElementById("lose-modal").style.display = "none";
    document.getElementById("dictionary-modal").style.display = "none";
    document.getElementById("guide-modal").style.display = "none";
    this.tooltipManager.forceHide();
  }

  resetCategorySlots() {
    const lang = this.language === "ru" ? "ru" : "en";
    const slots = document.querySelectorAll(".category-slot");
    slots.forEach((slot, index) => {
      slot.classList.remove("filled", "incorrect");
      slot.querySelector(".category-name").textContent = UI_STRINGS.categoryN[
        lang
      ].replace("%d", String(index + 1));
      slot.querySelector(".category-concepts").textContent = "";
    });
  }

  applyLanguage() {
    const lang = this.language === "ru" ? "ru" : "en";
    this.language = lang;
    document.documentElement.lang = lang;

    const langToggle = document.getElementById("lang-toggle");
    if (langToggle) {
      langToggle.textContent = lang === "ru" ? "RU / EN" : "EN / RU";
    }

    const headerTitle = document.querySelector(".header h1");
    if (headerTitle) headerTitle.textContent = UI_STRINGS.headerTitle[lang];

    const welcomeTitle = document.querySelector(".welcome-card h1");
    if (welcomeTitle) welcomeTitle.textContent = UI_STRINGS.welcomeTitle[lang];

    const welcomeDesc = document.querySelector(".welcome-desc");
    if (welcomeDesc) welcomeDesc.textContent = UI_STRINGS.welcomeDesc[lang];

    const soloPanelTitle = document.querySelector(
      ".solo-panel .welcome-panel-title",
    );
    if (soloPanelTitle)
      soloPanelTitle.textContent = UI_STRINGS.soloPanelTitle[lang];

    const roomPanelTitle = document.querySelector(
      ".room-panel .welcome-panel-title",
    );
    if (roomPanelTitle)
      roomPanelTitle.textContent = UI_STRINGS.roomPanelTitle[lang];

    document
      .querySelectorAll(".welcome-option-group label")
      .forEach((label) => {
        const txt = label.textContent.trim();
        if (txt.startsWith("Mode") || txt.startsWith("Режим")) {
          label.textContent = UI_STRINGS.modeLabel[lang];
        }
        if (txt.startsWith("Category templates")) {
          label.textContent = UI_STRINGS.templatesLabel[lang];
        }
      });

    document.querySelectorAll(".welcome-mode-btn").forEach((btn) => {
      const mode = btn.dataset.mode;
      if (mode === "easy") btn.textContent = UI_STRINGS.modeEasy[lang];
      if (mode === "basic") btn.textContent = UI_STRINGS.modeBasic[lang];
      if (mode === "advanced") btn.textContent = UI_STRINGS.modeAdvanced[lang];
    });

    document.querySelectorAll(".mode-btn").forEach((btn) => {
      const mode = btn.dataset.mode;
      if (mode === "easy") btn.textContent = UI_STRINGS.modeEasy[lang];
      if (mode === "basic") btn.textContent = UI_STRINGS.modeBasic[lang];
      if (mode === "advanced") btn.textContent = UI_STRINGS.modeAdvanced[lang];
    });

    const soloBtn = document.getElementById("welcome-solo");
    if (soloBtn) soloBtn.textContent = UI_STRINGS.startSolo[lang];

    const lobbyCreateBtn = document.getElementById("welcome-lobby-create");
    if (lobbyCreateBtn)
      lobbyCreateBtn.textContent = UI_STRINGS.createRoom[lang];

    const homeBtn = document.getElementById("home-btn");
    if (homeBtn) homeBtn.textContent = UI_STRINGS.backHome[lang];

    const submitBtn = document.getElementById("submit-btn");
    if (submitBtn) submitBtn.textContent = UI_STRINGS.submit[lang];

    const deselectBtn = document.getElementById("deselect-btn");
    if (deselectBtn) deselectBtn.textContent = UI_STRINGS.deselectAll[lang];

    const newGameBtn = document.getElementById("new-game-btn");
    if (newGameBtn) newGameBtn.textContent = UI_STRINGS.newGame[lang];

    const dictBtn = document.querySelector(".dictionary-btn");
    if (dictBtn) dictBtn.textContent = UI_STRINGS.dictionaryButton[lang];

    const guideBtn = document.querySelector(".guide-btn");
    if (guideBtn) guideBtn.textContent = UI_STRINGS.guideButton[lang];

    const lbBtn = document.querySelector(".leaderboard-btn");
    if (lbBtn) lbBtn.textContent = UI_STRINGS.leaderboardButton[lang];

    const joinLabel = document.querySelector(".welcome-join label");
    if (joinLabel) joinLabel.textContent = UI_STRINGS.joinLabel[lang];

    const joinBtn = document.getElementById("welcome-join-btn");
    if (joinBtn) joinBtn.textContent = UI_STRINGS.joinButton[lang];

    const joinHint = document.querySelector(".welcome-join .welcome-hint");
    if (joinHint) joinHint.textContent = UI_STRINGS.joinHint[lang];

    const linkInput = document.getElementById("welcome-room-link");
    if (linkInput) linkInput.placeholder = UI_STRINGS.linkPlaceholder[lang];

    const templatesApplyBtn = document.getElementById("templates-apply-btn");
    if (templatesApplyBtn)
      templatesApplyBtn.textContent = UI_STRINGS.templatesApply[lang];

    // Win modal
    const winTitle = document.querySelector("#win-modal h2");
    if (winTitle) winTitle.textContent = UI_STRINGS.winTitle[lang];
    const winDogLine = document.querySelector("#win-modal p:nth-of-type(2)");
    if (winDogLine) winDogLine.textContent = UI_STRINGS.winDogLine[lang];
    const winNameLabel = document.querySelector("#win-name-prompt label");
    if (winNameLabel) winNameLabel.textContent = UI_STRINGS.winNameLabel[lang];
    const winSaveBtn = document.getElementById("win-save-btn");
    if (winSaveBtn) winSaveBtn.textContent = UI_STRINGS.winSave[lang];
    const winNextBtn = document.getElementById("win-next-btn");
    if (winNextBtn) winNextBtn.textContent = UI_STRINGS.winNext[lang];

    // Lose modal
    const loseTitle = document.querySelector("#lose-modal h2");
    if (loseTitle) loseTitle.textContent = UI_STRINGS.loseTitle[lang];
    const loseBtn = document.querySelector("#lose-modal .new-game-btn");
    if (loseBtn) loseBtn.textContent = UI_STRINGS.loseButton[lang];

    // Room-name modal
    const roomJoinTitle = document.querySelector("#room-name-modal h2");
    if (roomJoinTitle)
      roomJoinTitle.textContent = UI_STRINGS.roomJoinTitle[lang];
    const roomJoinText = document.querySelector("#room-name-modal p");
    if (roomJoinText) roomJoinText.textContent = UI_STRINGS.roomJoinText[lang];
    const roomJoinLabel = document.querySelector("#room-name-modal label");
    if (roomJoinLabel)
      roomJoinLabel.textContent = UI_STRINGS.roomJoinLabel[lang];
    const roomJoinStart = document.getElementById("room-name-start-btn");
    if (roomJoinStart)
      roomJoinStart.textContent = UI_STRINGS.roomJoinStart[lang];

    // Dictionary modal
    const dictTitle = document.querySelector("#dictionary-modal h2");
    if (dictTitle) dictTitle.textContent = UI_STRINGS.dictionaryTitle[lang];
    const dictIntro = document.querySelector("#dictionary-modal p");
    if (dictIntro) dictIntro.textContent = UI_STRINGS.dictionaryIntro[lang];
    const dictClose = document.querySelector("#dictionary-modal .new-game-btn");
    if (dictClose) dictClose.textContent = UI_STRINGS.dictionaryClose[lang];

    // Guide modal
    const guideTitle = document.querySelector("#guide-modal h2");
    if (guideTitle) guideTitle.textContent = UI_STRINGS.guideTitle[lang];
    const guideStart = document.querySelector("#guide-modal .guide-close-btn");
    if (guideStart) guideStart.textContent = UI_STRINGS.guideStart[lang];
    const guideSections = document.querySelectorAll(
      "#guide-modal .guide-section",
    );
    UI_STRINGS.guideSections.forEach((sectionData, i) => {
      const section = guideSections[i];
      if (!section || !sectionData) return;
      const h3 = section.querySelector("h3");
      if (h3) h3.textContent = sectionData.title[lang];
      const afterH3 = Array.from(section.children).slice(1);
      afterH3.forEach((c) => c.remove());
      section.insertAdjacentHTML("beforeend", sectionData.body[lang]);
    });

    // Leaderboard modal
    const lbTitle = document.querySelector("#leaderboard-modal h2");
    if (lbTitle) lbTitle.textContent = UI_STRINGS.leaderboardTitle[lang];
    const lbClose = document.getElementById("leaderboard-close-btn");
    if (lbClose) lbClose.textContent = UI_STRINGS.leaderboardClose[lang];

    const loadingTextEl = document.querySelector(".app-loading-text");
    if (loadingTextEl) loadingTextEl.textContent = UI_STRINGS.loadingText[lang];

    // Lobby bar (puzzle page)
    const copyBtn = document.getElementById("lobby-copy-btn");
    if (copyBtn) copyBtn.textContent = UI_STRINGS.copyLink[lang];
    const lobbyLbBtn = document.getElementById("lobby-leaderboard-btn");
    if (lobbyLbBtn) lobbyLbBtn.textContent = UI_STRINGS.lobbyLeaderboard[lang];
    const lobbyDownloadBtn = document.getElementById("lobby-download-btn");
    if (lobbyDownloadBtn)
      lobbyDownloadBtn.textContent = UI_STRINGS.downloadResults[lang];
    const lbDownloadBtn = document.getElementById("leaderboard-download-btn");
    if (lbDownloadBtn)
      lbDownloadBtn.textContent = UI_STRINGS.downloadResults[lang];

    // Templates / concepts: refresh definitions for active language
    this.refreshConceptDefinitionsForLanguage(lang);
    this.updateDictionaryEntries();

    // Puzzle page: score, hint button, round, category placeholders
    this.updateScoreDisplay(this.currentPuzzle ? this.currentScore : null);
    this.updateHintButton();
    this.updateRoundDisplay();
    document.querySelectorAll(".category-slot").forEach((slot, index) => {
      if (!slot.classList.contains("filled")) {
        const nameEl = slot.querySelector(".category-name");
        if (nameEl)
          nameEl.textContent = UI_STRINGS.categoryN[lang].replace(
            "%d",
            String(index + 1),
          );
      }
    });
    this.updateHeaderDesc();
    this.updateHintsToggle();
    localStorage.setItem(LANG_STORAGE_KEY, lang);
  }

  toggleLanguage() {
    const oldLang = this.language;
    this.language = this.language === "en" ? "ru" : "en";
    const newLang = this.language;
    this.applyLanguage();
    this.translateCurrentGameToLanguage(oldLang, newLang);
  }
}

// Initialize game when DOM is ready (modules run deferred; DOMContentLoaded may have already fired)
function initGame() {
  new AISafetyGame();
}
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initGame);
} else {
  initGame();
}

export default AISafetyGame;
