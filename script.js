let currentModelData = null;
let charts = {}; // Store chart instances

// DOM Elements
const els = {
    modelSelect: document.getElementById('model-selector'),
    streak: document.getElementById('in-streak'),
    delay: document.getElementById('in-delay'),
    utility: document.getElementById('in-utility'),
    linkedin: document.getElementById('in-linkedin'),
    valStreak: document.getElementById('val-streak'),
    valDelay: document.getElementById('val-delay'),
    valUtility: document.getElementById('val-utility'),
    score: document.getElementById('score-display'),
    risk: document.getElementById('risk-display'),
    status: document.getElementById('model-status'),
    scoreCircle: document.querySelector('.score-circle')
};

// --- 1. INITIALIZATION ---
document.addEventListener('DOMContentLoaded', async () => {
    initCharts();
    
    // Attempt to load the default selected model
    await loadModel(els.modelSelect.value);

    // Event Listeners
    els.modelSelect.addEventListener('change', (e) => loadModel(e.target.value));
    
    // Real-time inference on input change
    [els.streak, els.delay, els.utility, els.linkedin].forEach(el => {
        el.addEventListener('input', runInference);
    });
});

// --- 2. MODEL LOADING ---
async function loadModel(filename) {
    els.status.textContent = `Fetching ${filename}...`;
    els.score.textContent = "...";
    els.status.style.color = "#94a3b8"; // Reset color

    try {
        // Security check for local file opening
        if (window.location.protocol === 'file:') {
            throw new Error("CORS_BLOCK");
        }

        const response = await fetch(filename);
        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
        
        currentModelData = await response.json();
        
        const typeLabel = currentModelData.type.toUpperCase();
        els.status.textContent = `✔ Loaded: ${typeLabel} Architecture`;
        els.status.style.color = "#4ade80"; // Green text

        runInference(); // Run first prediction immediately

    } catch (error) {
        console.warn("Model load failed:", error);
        handleLoadError(error, filename);
    }
}

function handleLoadError(error, filename) {
    let msg = "Error loading model.";
    
    if (error.message === "CORS_BLOCK") {
        msg = "⚠️ Demo Mode: Local file access blocked. Using fallback logic.";
    } else {
        msg = `⚠️ Demo Mode: '${filename}' not found. Using fallback logic.`;
    }
    
    els.status.textContent = msg;
    els.status.style.color = "#facc15"; // Yellow warning

    // Set a Dummy Model so the UI still works for the user
    // This ensures the simulator is usable even if they messed up the JSON upload
    if (filename.includes("linear")) {
        currentModelData = { type: 'linear', intercept: 500, coefficients: [10, -12, 100, 50] };
    } else {
        // Dummy tree-like structure for demo purposes
        currentModelData = { type: 'demo_fallback' };
    }
    runInference();
}

// --- 3. INFERENCE ENGINE ---
function runInference() {
    if (!currentModelData) return;

    // Update UI Labels
    els.valStreak.textContent = `${els.streak.value} mo`;
    els.valDelay.textContent = `${els.delay.value} days`;
    els.valUtility.textContent = `${els.utility.value}%`;

    // 1. Prepare Feature Vector [streak, delay, utility, linkedin]
    // MUST match the order used in Python training
    const features = [
        parseInt(els.streak.value),
        parseInt(els.delay.value),
        parseInt(els.utility.value) / 100, // Normalize 0-100 to 0.0-1.0
        els.linkedin.checked ? 1 : 0
    ];

    let prediction = 500; // Default baseline

    // 2. Select Logic based on Model Type
    if (currentModelData.type === 'linear') {
        prediction = currentModelData.intercept;
        features.forEach((val, i) => {
            prediction += val * currentModelData.coefficients[i];
        });
    } 
    else if (currentModelData.type === 'forest') {
        // Random Forest: Average of all tree predictions
        let sum = 0;
        currentModelData.trees.forEach(tree => {
            sum += traverseTree(tree, features);
        });
        prediction = sum / currentModelData.n_estimators;
    } 
    else if (currentModelData.type === 'gbm') {
        // XGBoost/GBM: Base Score + LearningRate * Sum(TreeResiduals)
        let treeSum = 0;
        currentModelData.trees.forEach(tree => {
            treeSum += traverseTree(tree, features);
        });
        prediction = currentModelData.init_score + (currentModelData.learning_rate * treeSum);
    }
    else if (currentModelData.type === 'demo_fallback') {
        // Simple hardcoded logic if JSON is missing
        prediction = 500 + (features[0]*10) - (features[1]*10) + (features[2]*100);
    }

    // 3. Post-Process
    prediction = Math.round(Math.max(300, Math.min(900, prediction)));
    
    // 4. Update UI
    els.score.textContent = prediction;
    updateRiskVisuals(prediction);
    updateCharts(prediction);
}

// --- 4. TREE TRAVERSAL (For Forest & GBM) ---
function traverseTree(node, features) {
    // Base case: Leaf node (has 'value' property)
    if (node.value !== undefined) {
        return node.value;
    }

    // Recursive step: Compare feature against threshold
    const featureVal = features[node.feature_index];
    if (featureVal <= node.threshold) {
        return traverseTree(node.left, features);
    } else {
        return traverseTree(node.right, features);
    }
}

// --- 5. UI UPDATES ---
function updateRiskVisuals(score) {
    let color, text, bg;

    if (score >= 750) {
        color = "#22c55e"; text = "Low Risk (Approved)"; bg = "rgba(34, 197, 94, 0.1)";
    } else if (score >= 600) {
        color = "#f59e0b"; text = "Medium Risk (Review)"; bg = "rgba(245, 158, 11, 0.1)";
    } else {
        color = "#ef4444"; text = "High Risk (Reject)"; bg = "rgba(239, 68, 68, 0.1)";
    }

    els.risk.textContent = text;
    els.risk.style.color = color;
    els.scoreCircle.style.borderColor = color;
    els.scoreCircle.style.boxShadow = `0 0 30px ${color}40`; // Soft glow
}

function initCharts() {
    // 1. Streak Chart
    const ctxStreak = document.getElementById('streakChart');
    charts.streak = new Chart(ctxStreak, {
        type: 'line',
        data: {
            labels: ['0', '6', '12', '18', '24'],
            datasets: [{
                label: 'Projected Score',
                data: [500, 550, 600, 650, 700],
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { min: 300, max: 900 } }
        }
    });

    // 2. Risk Pie Chart
    const ctxRisk = document.getElementById('riskChart');
    charts.risk = new Chart(ctxRisk, {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
            datasets: [{
                data: [45, 35, 20],
                backgroundColor: ['#22c55e', '#f59e0b', '#ef4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function updateCharts(currentScore) {
    // Dynamically update the line chart based on current streak
    // This is a visual effect to make the dashboard feel alive
    if (charts.streak) {
        const base = currentScore - (parseInt(els.streak.value) * 10);
        const newData = [0, 6, 12, 18, 24].map(m => Math.min(900, base + (m * 10)));
        charts.streak.data.datasets[0].data = newData;
        charts.streak.update('none'); // 'none' for smooth animation
    }
}