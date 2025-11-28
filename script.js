// Global variables to store model weights
let modelWeights = null;

// DOM Elements
const els = {
    streak: document.getElementById('in-streak'),
    delay: document.getElementById('in-delay'),
    utility: document.getElementById('in-utility'),
    linkedin: document.getElementById('in-linkedin'),
    valStreak: document.getElementById('val-streak'),
    valDelay: document.getElementById('val-delay'),
    valUtility: document.getElementById('val-utility'),
    score: document.getElementById('score-display'),
    risk: document.getElementById('risk-display'),
    status: document.getElementById('model-status')
};

// 1. Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
    initCharts();
    await loadModel();
    updateSimulation(); // Run first calculation
});

// 2. Load Model from JSON
async function loadModel() {
    try {
        const response = await fetch('model.json');
        if (!response.ok) throw new Error("Failed to load model.json");
        
        const data = await response.json();
        modelWeights = data.weights;
        
        els.status.textContent = "Model Loaded (Linear Regression)";
        els.status.classList.remove('loading');
        els.status.classList.add('ready');
        
        // Add event listeners only after model loads
        [els.streak, els.delay, els.utility, els.linkedin].forEach(el => {
            el.addEventListener('input', updateSimulation);
        });
        
    } catch (error) {
        console.error(error);
        els.status.textContent = "Error Loading Model";
        // Fallback weights if JSON fails (for demo reliability)
        modelWeights = {
            intercept: 500,
            coefficients: [10, -12, 100, 50] // [streak, delay, utility, linkedin]
        };
    }
}

// 3. The Inference Engine (Running in Browser)
function updateSimulation() {
    // Update UI Labels
    els.valStreak.textContent = els.streak.value;
    els.valDelay.textContent = els.delay.value;
    els.valUtility.textContent = els.utility.value + '%';

    if (!modelWeights) return;

    // A. Extract Features
    // Note: Must match the order used in Python training!
    const features = [
        parseInt(els.streak.value),         // Feature 0: Streak
        parseInt(els.delay.value),          // Feature 1: Delay
        parseInt(els.utility.value) / 100,  // Feature 2: Utility (0-1 float)
        els.linkedin.checked ? 1 : 0        // Feature 3: LinkedIn (Binary)
    ];

    // B. Calculate Dot Product (Intercept + W1*X1 + W2*X2...)
    let score = modelWeights.intercept;
    features.forEach((val, index) => {
        score += val * modelWeights.coefficients[index];
    });

    // C. Clamp Score (300 - 900)
    score = Math.max(300, Math.min(900, Math.round(score)));

    // D. Update UI
    els.score.textContent = score;
    updateRiskUI(score);
}

function updateRiskUI(score) {
    let riskText = "Medium Risk";
    let color = "#f59e0b"; // Yellow
    let bg = "rgba(245, 158, 11, 0.2)";

    if (score >= 750) {
        riskText = "Low Risk";
        color = "#22c55e"; // Green
        bg = "rgba(34, 197, 94, 0.2)";
    } else if (score < 600) {
        riskText = "High Risk";
        color = "#ef4444"; // Red
        bg = "rgba(239, 68, 68, 0.2)";
    }

    els.risk.textContent = riskText;
    els.risk.style.color = color;
    els.risk.style.backgroundColor = bg;
    document.querySelector('.score-circle').style.borderColor = color;
}

// 4. Initialize Charts (Chart.js)
function initCharts() {
    // Chart 1: Streak vs Score
    new Chart(document.getElementById('streakChart'), {
        type: 'line',
        data: {
            labels: ['0 mo', '6 mo', '12 mo', '18 mo', '24 mo'],
            datasets: [{
                label: 'TrustScore Projection',
                data: [500, 560, 620, 680, 740],
                borderColor: '#2563eb',
                tension: 0.4,
                fill: true,
                backgroundColor: 'rgba(37, 99, 235, 0.1)'
            }]
        },
        options: { responsive: true, plugins: { legend: { display: false } } }
    });

    // Chart 2: Risk Distribution
    new Chart(document.getElementById('riskChart'), {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
            datasets: [{
                data: [45, 35, 20],
                backgroundColor: ['#22c55e', '#f59e0b', '#ef4444'],
                borderWidth: 0
            }]
        },
        options: { responsive: true }
    });
}