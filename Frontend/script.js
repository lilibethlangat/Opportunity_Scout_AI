const API_URL = 'http://127.0.0.1:8000';

async function renderDashboardTable() {
    const tableBody = document.getElementById('startup-table-body');
    try {
        const response = await fetch(`${API_URL}/top-opportunities`);
        const data = await response.json();
        if (data.error) return;

        document.getElementById('stat-total').innerText = data.length;
        let totalScore = 0;
        tableBody.innerHTML = '';

        data.forEach(item => {
            totalScore += item.ai_predicted_score;
            tableBody.innerHTML += `<tr>
                <td><strong>${item.name}</strong></td>
                <td>${item.industry}</td>
                <td>$${(item.funding_total_usd || 0).toLocaleString()}</td>
                <td style="color:#3b82f6; font-weight:800;">${item.ai_predicted_score.toFixed(1)}%</td>
                <td style="color:#059669; font-weight:600;">✨ ${item.explanation}</td>
            </tr>`;
        });
        document.getElementById('stat-avg-viability').innerText = `${(totalScore / data.length || 0).toFixed(1)}%`;
    } catch (e) { console.error("Dashboard failed to load."); }
}

async function handleEvaluation(event) {
    event.preventDefault();
    const resultDiv = document.getElementById('prediction-result');
    resultDiv.innerHTML = "<p>Processing...</p>";

    const payload = {
        name: document.getElementById('form-name').value,
        founded_year: parseInt(document.getElementById('form-year').value),
        funding_total_usd: parseFloat(document.getElementById('form-funding').value),
        funding_rounds: parseInt(document.getElementById('form-rounds').value),
        country: document.getElementById('form-country').value,
        industry: document.getElementById('form-industry').value
    };

    try {
        const response = await fetch(`${API_URL}/evaluate`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const result = await response.json();

        // FIXED: Using explicit keys from the backend response
        resultDiv.innerHTML = `
            <div style="background:#f0fdf4; padding:25px; border-radius:12px; border: 1px solid #bbf7d0; margin-top:20px;">
                <h3 style="color:#166534">${result.name} Analysis</h3>
                <p style="font-size:2.5rem; font-weight:800; color:#1e293b; margin:10px 0;">${result.score}%</p>
                <p style="color:#15803d; font-weight:600;">✨ ${result.explanation}</p>
                <button id="save-btn" class="btn-primary" style="background:#059669; width:100%; margin-top:15px;">📥 Save to Dashboard</button>
            </div>
        `;

        document.getElementById('save-btn').onclick = async () => {
            const sRes = await fetch(`${API_URL}/save-startup`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const sData = await sRes.json();
            if(sData.status === "saved") {
                alert("Successfully saved to database!");
                renderDashboardTable();
            }
        };
    } catch (e) { resultDiv.innerHTML = "<p style='color:red;'>Connection Error</p>"; }
}

function showSection(section) {
    document.getElementById('sec-dashboard').style.display = section === 'dashboard' ? 'block' : 'none';
    document.getElementById('sec-evaluation').style.display = section === 'evaluation' ? 'block' : 'none';
}

function enterApp() {
    document.getElementById('landing-page').style.display = 'none';
    document.getElementById('app-interface').style.display = 'flex';
    renderDashboardTable();
}

document.getElementById('evaluationForm').addEventListener('submit', handleEvaluation);