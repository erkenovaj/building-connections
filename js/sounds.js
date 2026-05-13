/**
 * Simple sound effects using Web Audio API (no external files).
 * Browsers require user interaction before playing; first game action unlocks audio.
 */

let audioContext = null;

function getContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioContext;
}

function playTone(frequency, duration, type = 'sine', volume = 0.15) {
    try {
        const ctx = getContext();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = frequency;
        osc.type = type;
        gain.gain.setValueAtTime(volume, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + duration);
    } catch (_) {}
}

function playCorrect() {
    playTone(523, 0.12);
    setTimeout(() => playTone(659, 0.12), 80);
    setTimeout(() => playTone(784, 0.2), 160);
}

function playWrong() {
    playTone(200, 0.15, 'sawtooth');
    setTimeout(() => playTone(150, 0.25, 'sawtooth'), 100);
}

function playWin() {
    [523, 659, 784, 1047].forEach((f, i) => {
        setTimeout(() => playTone(f, 0.2), i * 120);
    });
}

function playLose() {
    playTone(392, 0.2);
    setTimeout(() => playTone(330, 0.25), 150);
    setTimeout(() => playTone(262, 0.35), 300);
}

function playShuffle() {
    playTone(440, 0.08);
    setTimeout(() => playTone(554, 0.08), 60);
}

const Sounds = {
    correct: playCorrect,
    wrong: playWrong,
    win: playWin,
    lose: playLose,
    shuffle: playShuffle
};

export default Sounds;
