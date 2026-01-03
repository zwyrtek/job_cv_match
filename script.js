// Job-CV Match - Frontend JavaScript
// Connects to Flask API backend for neural network predictions
// NOW WITH TECHNICAL AND SOFT SKILLS CATEGORIZATION

const API_URL = 'http://localhost:5000';  // vede do firemniho prostredi proto run na local pro pouziti mimo company

const jobTextarea = document.getElementById('jobText');
const cvTextarea = document.getElementById('cvText');
const evaluateBtn = document.getElementById('evaluateBtn');
const loadingText = document.getElementById('loadingText');
const resultSection = document.getElementById('result');

// Event listener for evaluate button
evaluateBtn.addEventListener('click', async () => {
    const jobDescription = jobTextarea.value.trim();
    const candidateCV = cvTextarea.value.trim();

    // Validation
    if (!jobDescription) {
        showError('Please enter a Job Description');
        return;
    }

    if (!candidateCV) {
        showError('Please enter a Candidate CV');
        return;
    }

    // Show loading state
    evaluateBtn.disabled = true;
    loadingText.style.display = 'inline';
    resultSection.innerHTML = '<div class="placeholder">Analyzing with neural network...</div>';

    try {
        // Call API
        const response = await fetch(`${API_URL}/api/evaluate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                jobDescription: jobDescription,
                candidateCV: candidateCV
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'API request failed');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        showError(`Failed to evaluate match: ${error.message}. Make sure the API server is running on ${API_URL}`);
    } finally {
        // Hide loading state
        evaluateBtn.disabled = false;
        loadingText.style.display = 'none';
    }
});

function displayResults(data) {
    const {
        score,
        scoreCategory,
        scoreColor,
        yearsExperience,
        matchedTechnical,
        matchedSoft,
        missingTechnical,
        missingSoft,
        explanation
    } = data;

    // Determine progress bar width (scale 1-10 to 10-100%)
    const progressWidth = Math.max(10, (score / 10) * 100);

    // Build HTML
    let html = `
        <div class="score-section">
            <div class="score">${score}<span style="font-size: 1rem; color: var(--muted);">/10</span></div>
            <span class="badge">${scoreCategory}</span>
        </div>
        
        <div class="progress">
            <span style="width: ${progressWidth}%; transition: width 0.5s ease;"></span>
        </div>
        
        <p class="expl">${explanation}</p>
    `;

    // Count total skills
    const totalMatched = (matchedTechnical?.length || 0) + (matchedSoft?.length || 0);
    const totalMissing = (missingTechnical?.length || 0) + (missingSoft?.length || 0);

    // Show skills grid if we have skills
    if (totalMatched > 0 || totalMissing > 0) {
        html += '<div class="grid">';

        // MATCHED TECHNICAL SKILLS
        if (matchedTechnical && matchedTechnical.length > 0) {
            html += `
                <div class="card">
                    <h4>✓ Technical Skills Match (${matchedTechnical.length})</h4>
                    <div class="tags">
                        ${matchedTechnical.map(skill => `<span class="tag ok">${escapeHtml(skill)}</span>`).join('')}
                    </div>
                </div>
            `;
        }

        // MATCHED SOFT SKILLS
        if (matchedSoft && matchedSoft.length > 0) {
            html += `
                <div class="card">
                    <h4>✓ Soft Skills Match (${matchedSoft.length})</h4>
                    <div class="tags">
                        ${matchedSoft.map(skill => `<span class="tag ok">${escapeHtml(skill)}</span>`).join('')}
                    </div>
                </div>
            `;
        }

        // MISSING TECHNICAL SKILLS
        if (missingTechnical && missingTechnical.length > 0) {
            html += `
                <div class="card">
                    <h4>⚠ Missing Technical Skills (${missingTechnical.length})</h4>
                    <div class="tags">
                        ${missingTechnical.map(skill => `<span class="tag gap">${escapeHtml(skill)}</span>`).join('')}
                    </div>
                </div>
            `;
        }

        // MISSING SOFT SKILLS
        if (missingSoft && missingSoft.length > 0) {
            html += `
                <div class="card">
                    <h4>⚠ Missing Soft Skills (${missingSoft.length})</h4>
                    <div class="tags">
                        ${missingSoft.map(skill => `<span class="tag gap">${escapeHtml(skill)}</span>`).join('')}
                    </div>
                </div>
            `;
        }

        html += '</div>';
    }

    resultSection.innerHTML = html;

    // Smooth scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
    resultSection.innerHTML = `
        <div class="placeholder" style="color: var(--bad);">
            ⚠ ${escapeHtml(message)}
        </div>
    `;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto-expand textareas as user types
function autoExpand(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

jobTextarea.addEventListener('input', function() {
    autoExpand(this);
});

cvTextarea.addEventListener('input', function() {
    autoExpand(this);
});

// Check API health on page load
window.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch(`${API_URL}/api/health`);
        if (response.ok) {
            console.log('✓ API connection successful');
        } else {
            console.warn('⚠ API server responded but may have issues');
        }
    } catch (error) {
        console.warn('⚠ Cannot connect to API server. Make sure it is running on', API_URL);
        console.warn('Error:', error.message);
    }
});