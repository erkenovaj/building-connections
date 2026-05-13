class DogAnimations {
    constructor() {
        this.dog = null;
        this.dogContainer = null;
        this.currentMood = 'happy';
        this.init();
    }

    init() {
        this.createDog();
        this.startRandomAnimations();
    }

    createDog() {
        // Remove existing dog if any
        const existingDog = document.querySelector('.dog-container');
        if (existingDog) {
            existingDog.remove();
        }

        const dogContainer = document.createElement('div');
        dogContainer.className = 'dog-container';
        dogContainer.setAttribute('aria-hidden', 'true');
        this.dogContainer = dogContainer;
        
        this.dog = document.createElement('img');
        this.dog.className = 'dog';
        this.dog.alt = 'AI Safety Dog';
        
        dogContainer.appendChild(this.dog);
        document.body.appendChild(dogContainer);
        
        this.updateDogMood('happy');
        this.positionDogProminently();
    }

    updateDogMood(mood) {
        if (!this.dog) {
            this.createDog(); // Recreate if missing
            return;
        }
        
        console.log(`Updating dog mood to: ${mood}`);
        
        this.currentMood = mood;
        
        // Force remove all animation classes first
        this.dog.className = 'dog';
        
        // Clear any previous size overrides
        this.dog.style.width = '';
        this.dog.style.height = '';
        this.dog.style.animation = '';
        
        // Small delay to ensure CSS reset
        setTimeout(() => {
            if (!this.dog) return;
            
            this.dog.className = `dog ${mood}`;
            
            // Update dog image based on mood
            const imagePath = mood === 'sad' || mood === 'worried' || mood === 'concerned' 
                ? 'images/sad_dog.png' 
                : 'images/happy_dog.png';
            
            this.dog.src = imagePath;
            
            // Add error handling for image loading
            this.dog.onerror = () => {
                console.error(`Failed to load dog image: ${imagePath}`);
                if (this.dog && this.dog.tagName === 'IMG' && !imagePath.includes('happy_dog.png')) {
                    this.dog.src = 'images/happy_dog.png';
                }
            };
            
            // Make sure dog is always visible
            this.positionDogProminently();
            
            // Trigger mood-specific animations
            this.triggerMoodAnimation(mood);
        }, 10);
    }

    positionDogProminently() {
        if (!this.dog || !this.dogContainer) return;

        this.dogContainer.style.right = '18px';
        this.dogContainer.style.bottom = '18px';
        this.dogContainer.style.zIndex = '60';
        this.dog.style.left = '';
        this.dog.style.top = '';
    }

    triggerMoodAnimation(mood) {
        if (!this.dog) return;

        // Use setTimeout to ensure CSS class is applied before animation
        setTimeout(() => {
            if (!this.dog) return;

            switch(mood) {
                case 'happy':
                    this.createPawTrail();
                    break;
                case 'worried':
                case 'concerned':
                case 'sad':
                    this.spawnBones();
                    break;
                case 'celebrating':
                    this.createConfetti();
                    break;
            }
        }, 50);
    }

    createPawTrail() {
        // Disabled - paws removed
        return;
    }

    spawnBones() {
        // Disabled - bones removed
        return;
    }

    createConfetti() {
        const emojis = ['🎉', '🎊', '⭐', '✨', '🎈'];
        for (let i = 0; i < 12; i++) {
            setTimeout(() => {
                const confetti = document.createElement('div');
                confetti.className = 'confetti';
                confetti.textContent = emojis[Math.floor(Math.random() * emojis.length)];
                confetti.style.left = Math.random() * 100 + 'vw';
                confetti.style.fontSize = '36px';
                confetti.style.zIndex = '9998';
                document.body.appendChild(confetti);
                
                setTimeout(() => {
                    if (confetti.parentElement) {
                        confetti.remove();
                    }
                }, 3000);
            }, i * 200);
        }
    }

    floatAcrossScreen() {
        if (!this.dog) return;
        
        this.dog.style.animation = 'floatAcross 8s ease-in-out';
        setTimeout(() => {
            if (this.dog) {
                this.dog.style.animation = '';
                this.positionDogProminently();
            }
        }, 8000);
    }

    startRandomAnimations() {
        setInterval(() => {
            if (this.currentMood === 'happy' && Math.random() < 0.2) {
                this.floatAcrossScreen();
            }
        }, 15000);
    }
}

export default DogAnimations;
