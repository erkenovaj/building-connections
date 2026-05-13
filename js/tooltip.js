import { CONCEPT_DEFINITIONS } from './config.js';

function isTouchEnv() {
    return typeof window !== 'undefined' && ('ontouchstart' in window || navigator.maxTouchPoints > 0);
}

class TooltipManager {
    constructor() {
        this.tooltip = document.getElementById('concept-tooltip');
        this.currentConcept = null;
        this.isSticky = false;
        this.init();
    }

    init() {
        // Create tooltip element if it doesn't exist
        if (!this.tooltip) {
            this.tooltip = document.createElement('div');
            this.tooltip.id = 'concept-tooltip';
            this.tooltip.className = 'concept-tooltip';
            document.body.appendChild(this.tooltip);
        }

        this.tooltip.addEventListener('mouseenter', () => this.makeSticky());
        this.tooltip.addEventListener('mouseleave', () => this.unstick());
    }

    show(concept, element) {
        const conceptData = CONCEPT_DEFINITIONS[concept];
        if (!conceptData) return;

        this.tooltip.innerHTML = `
            <div class="tooltip-title">${concept}</div>
            <div class="tooltip-definition">${conceptData}</div>
        `;

        this.positionTooltip(element);
        this.tooltip.classList.add('show');
        this.currentConcept = concept;
        // On touch devices, keep tooltip visible until explicitly hidden
        if (isTouchEnv()) {
            this.isSticky = true;
            this.tooltip.classList.add('sticky');
        }
    }

    positionTooltip(element) {
        const rect = element.getBoundingClientRect();

        // Temporarily show tooltip off-screen to measure its size accurately
        this.tooltip.style.left = '-9999px';
        this.tooltip.style.top = '-9999px';
        this.tooltip.classList.add('show');
        const tooltipRect = this.tooltip.getBoundingClientRect();

        const scrollX = window.scrollX || window.pageXOffset || 0;
        const scrollY = window.scrollY || window.pageYOffset || 0;

        // Center horizontally above the target cell
        let left = rect.left + scrollX + (rect.width / 2) - (tooltipRect.width / 2);
        let top = rect.top + scrollY - tooltipRect.height - 8;

        // Clamp horizontally within viewport
        const margin = 10;
        if (left < margin) left = margin;
        if (left + tooltipRect.width > scrollX + window.innerWidth - margin) {
            left = scrollX + window.innerWidth - margin - tooltipRect.width;
        }

        // If there isn't enough space above, place directly below the cell
        if (top < scrollY + margin) {
            top = rect.bottom + scrollY + 8;
        }

        this.tooltip.style.left = left + 'px';
        this.tooltip.style.top = top + 'px';
    }

    hide() {
        if (isTouchEnv()) {
            return;
        }
        if (this.isSticky) return;
        this.tooltip.classList.remove('show', 'sticky');
        this.currentConcept = null;
    }

    makeSticky() {
        this.isSticky = true;
        this.tooltip.classList.add('sticky');
    }

    unstick() {
        if (isTouchEnv()) return;
        this.isSticky = false;
        this.tooltip.classList.remove('sticky');
    }

    // Force hide tooltip (for game events)
    forceHide() {
        this.isSticky = false;
        this.tooltip.classList.remove('show', 'sticky');
        this.currentConcept = null;
    }
}

export default TooltipManager;