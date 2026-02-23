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
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await response.json();

        // Handle API error responses
        if (result.error) {
            resultDiv.innerHTML = `<div class="glass-card" style="padding:20px; border-color: rgba(239, 68, 68, 0.2); margin-top:20px;">
                <p style="color:#ef4444; font-weight:700;">⚠️ API Error</p>
                <p style="color:var(--text-muted); font-size:0.9rem; margin-top:8px;">${result.error}</p>
            </div>`;
            return;
        }

        resultDiv.innerHTML = `
            <div id="evaluation-report" class="result-card glass-card" style="padding:3rem; margin-top:2rem; background:#0d121f; border: 1px solid #1e293b; color: #fff;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2rem; border-bottom: 1px solid #1e293b; padding-bottom: 1.5rem;">
                    <div>
                        <h4 style="color:#3b82f6; text-transform:uppercase; letter-spacing:2px; font-size:0.75rem; margin-bottom:5px;">Venture Intelligence Report</h4>
                        <h3 style="font-size:2.5rem; font-weight:800; margin:0; color:#fff;">${result.name}</h3>
                    </div>
                    <div style="text-align:right;">
                        <p style="color:#94a3b8; font-size:0.8rem; margin:0;">Generated on</p>
                        <p style="color:#fff; font-weight:600; margin:0;">${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}</p>
                    </div>
                </div>

                <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:20px; margin-bottom:2.5rem; background:rgba(255,255,255,0.02); padding:20px; border-radius:16px;">
                    <div><p style="color:#94a3b8; font-size:0.7rem; text-transform:uppercase; margin-bottom:4px;">Industry</p><p style="font-weight:600; color:#fff; margin:0;">${payload.industry}</p></div>
                    <div><p style="color:#94a3b8; font-size:0.7rem; text-transform:uppercase; margin-bottom:4px;">Region</p><p style="font-weight:600; color:#fff; margin:0;">${payload.country}</p></div>
                    <div><p style="color:#94a3b8; font-size:0.7rem; text-transform:uppercase; margin-bottom:4px;">Founded</p><p style="font-weight:600; color:#fff; margin:0;">${payload.founded_year}</p></div>
                    <div><p style="color:#94a3b8; font-size:0.7rem; text-transform:uppercase; margin-bottom:4px;">Total Funding</p><p style="font-weight:600; color:#fff; margin:0;">$${payload.funding_total_usd.toLocaleString()}</p></div>
                    <div><p style="color:#94a3b8; font-size:0.7rem; text-transform:uppercase; margin-bottom:4px;">Rounds</p><p style="font-weight:600; color:#fff; margin:0;">${payload.funding_rounds}</p></div>
                </div>

                <div style="text-align:center; margin-bottom:2.5rem; padding:30px; border:1px solid rgba(59, 130, 246, 0.2); border-radius:20px; background: radial-gradient(circle at center, rgba(59, 130, 246, 0.05), transparent);">
                    <p style="color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:1px; font-size:0.8rem;">ML Predicted Viability</p>
                    <p style="font-size:5rem; font-weight:800; color:#fff; margin:10px 0; line-height:1;">${result.score}<span style="font-size:2rem; color:#3b82f6;">%</span></p>
                    <p style="color:#10b981; font-weight:600; font-size: 1.25rem; margin-top:15px;">✨ ${result.explanation}</p>
                </div>

                <div data-html2canvas-ignore="true" style="display:flex; gap:12px;">
                    <button id="save-btn" class="btn-primary" style="flex:1;">📥 Save to Dashboard</button>
                    <button id="download-btn" class="btn-primary" style="flex:1; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1);">📄 Download PDF</button>
                </div>
            </div>
        `;

        document.getElementById('download-btn').onclick = () => downloadPDF(result.name);

        document.getElementById('save-btn').onclick = async () => {
            const sRes = await fetch(`${API_URL}/save-startup`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const sData = await sRes.json();
            if (sData.status === "saved") {
                alert("Successfully saved to database!");
                renderDashboardTable();
            }
        };
    } catch (e) { resultDiv.innerHTML = "<p style='color:red;'>Connection Error</p>"; }
}

function downloadPDF(startupName) {
    const element = document.getElementById('evaluation-report');

    const opt = {
        margin: [15, 0],
        filename: `${startupName}_Venture_Report.pdf`,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: {
            scale: 2,
            backgroundColor: '#0d121f',
            useCORS: true,
            scrollY: 0,
            width: 800 // Explicitly set width of the element to capture
        },
        jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
    };

    html2pdf().set(opt).from(element).save();
}

function showSection(section) {
    document.getElementById('sec-dashboard').style.display = section === 'dashboard' ? 'block' : 'none';
    document.getElementById('sec-evaluation').style.display = section === 'evaluation' ? 'block' : 'none';

    // Update active nav link
    const navLinks = document.querySelectorAll('.nav-links li');
    navLinks.forEach(link => {
        const text = link.innerText.toLowerCase();
        if (text.includes(section)) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

function enterApp() {
    document.getElementById('landing-page').style.display = 'none';
    document.getElementById('app-interface').style.display = 'flex';
    renderDashboardTable();
}

function goToLanding() {
    document.getElementById('landing-page').style.display = 'flex';
    document.getElementById('app-interface').style.display = 'none';
}

document.getElementById('evaluationForm').addEventListener('submit', handleEvaluation);