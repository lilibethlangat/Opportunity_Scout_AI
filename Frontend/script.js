// ═══════════════════════════════════════════════════════════════════
//  Opportunity Scout AI — script.js v3
//  Profile toggle (entrepreneur/investor), GDP context panel,
//  funding_rounds sent to API, no year-penalty messaging
// ═══════════════════════════════════════════════════════════════════

const API = '';
const $   = id => document.getElementById(id);

function fmt(n) {
  if (n >= 1_000_000) return '$' + (n/1_000_000).toFixed(1) + 'M';
  if (n >= 1_000)     return '$' + (n/1_000).toFixed(0) + 'K';
  return '$' + n.toLocaleString();
}
function tierColor(tier) {
  if ((tier||'').includes('High'))  return 'green';
  if ((tier||'').includes('Lower')) return 'red';
  return 'amber';
}
function qualityDot(label) {
  if (label === 'High')   return '● 5 live signals';
  if (label === 'Medium') return '◑ mix live + historical';
  return '◌ historical benchmarks';
}
function smoothTo(id) {
  event.preventDefault();
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior:'smooth' });
}

// ── Navigation ────────────────────────────────────────────────────
function enterApp() {
  $('landing-page').style.display  = 'none';
  $('app-interface').style.display = 'block';
  window.scrollTo(0, 0);
}
function goToLanding() {
  $('app-interface').style.display = 'none';
  $('landing-page').style.display  = 'block';
  window.scrollTo(0, 0);
}

// ── Tabs ──────────────────────────────────────────────────────────
function switchTab(tab) {
  ['score','whatif','compare'].forEach(t => {
    $(`tab-${t}-panel`).style.display = t === tab ? 'block' : 'none';
    $(`tab-${t}`).classList.toggle('active', t === tab);
  });
  if (tab === 'whatif') prefillWhatIf();
}

// ── Profile Toggle ────────────────────────────────────────────────
const HINTS = {
  investor:     '<strong>Investor view:</strong> Weights funding validation (22%), market trend (20%), news sentiment (18%). Best for deciding whether to back a startup.',
  entrepreneur: '<strong>Entrepreneur view:</strong> Weights trend slope (35%), news sentiment (25%), GDP growth (15%). Funding is not required — asks "Is this market fertile to build in?"',
};

function setProfile(prefix, type) {
  const hidden = $(`${prefix}-user-type`);
  if (hidden) hidden.value = type;
  ['investor','entrepreneur'].forEach(t => {
    const btn = $(`${prefix}-btn-${t}`);
    if (btn) btn.classList.toggle('active', t === type);
  });
  const hint = $(`${prefix}-profile-hint`);
  if (hint) hint.innerHTML = HINTS[type] || '';
}

// ── Health check ──────────────────────────────────────────────────
async function checkHealth() {
  try {
    const d = await fetch(`${API}/health`).then(r => r.json());
    $('api-dot').style.background = d.model_loaded ? '#22c55e' : '#f59e0b';
    $('api-label').textContent    = d.model_loaded ? 'Model ready' : 'Model loading…';
  } catch {
    $('api-dot').style.background = '#ef4444';
    $('api-label').textContent    = 'API offline';
  }
}

// ── Dropdowns ─────────────────────────────────────────────────────
async function loadDropdowns() {
  try {
    const [industries, countries] = await Promise.all([
      fetch(`${API}/api/v1/industries`).then(r => r.json()),
      fetch(`${API}/api/v1/countries`).then(r => r.json()),
    ]);
    window._industries = industries;
    window._countries  = countries;
    ['s-industry','wi-industry'].forEach(id => {
      const el = $(id); if (!el) return;
      industries.forEach(i => el.insertAdjacentHTML('beforeend', `<option value="${i}">${i}</option>`));
    });
    ['s-country','wi-country'].forEach(id => {
      const el = $(id); if (!el) return;
      countries.forEach(c => el.insertAdjacentHTML('beforeend', `<option value="${c}">${c}</option>`));
    });
    // Populate any compare card selects that were created before data arrived
    document.querySelectorAll('.cmp-card').forEach(card => {
      const id  = card.id;
      const ind = document.getElementById(id + '-industry');
      const ctr = document.getElementById(id + '-country');
      if (ind && ind.options.length <= 1)
        industries.forEach(i => ind.insertAdjacentHTML('beforeend', `<option value="${i}">${i}</option>`));
      if (ctr && ctr.options.length <= 1)
        countries.forEach(c => ctr.insertAdjacentHTML('beforeend', `<option value="${c}">${c}</option>`));
    });
  } catch(e) { console.warn('Dropdowns failed', e); }
}

// ── Init ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  checkHealth();
  await loadDropdowns();
  initCompare();

  // Score form submit
  $('scoreForm').addEventListener('submit', async e => {
    e.preventDefault();
    const name     = $('s-name').value.trim();
    const industry = $('s-industry').value;
    const country  = $('s-country').value;
    const year     = parseInt($('s-year').value);
    const funding  = parseFloat($('s-funding').value) || 0;
    const rounds   = parseInt($('s-rounds').value)   || 1;
    const userType = $('s-user-type').value;
    const errEl    = $('form-error');

    let err = '';
    if (!name)     err = 'Please enter a venture name.';
    else if (!industry) err = 'Please select an industry.';
    else if (!country)  err = 'Please select a country.';
    else if (isNaN(year) || year < 2000 || year > 2030) err = 'Enter a valid year (2000–2030).';
    else if (funding < 0) err = 'Funding cannot be negative.';

    if (err) { errEl.textContent = err; errEl.style.display = 'block'; return; }
    errEl.style.display = 'none';

    const btn = $('scoreForm').querySelector('.a-btn-calc');
    btn.textContent = '⏳ Analysing…'; btn.disabled = true;

    try {
      const data = await fetch(`${API}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, industry, country,
          founded_year: year, funding_total_usd: funding,
          funding_rounds: rounds, user_type: userType }),
      }).then(r => r.json());

      window._lastScore     = { name, industry, country, year, funding, rounds, userType };
      window._lastScoreData = data;
      renderResult('score-result', data);
    } catch(e) {
      errEl.textContent = 'API error: ' + e.message;
      errEl.style.display = 'block';
    } finally {
      btn.textContent = '✦ Calculate Score'; btn.disabled = false;
    }
  });

  // What-if sliders
  $('wi-fund-slider').addEventListener('input', function() {
    $('wi-fund-display').textContent = fmt(+this.value);
    $('wi-funding').value = this.value;
  });
  $('wi-year-slider').addEventListener('input', function() {
    $('wi-year-display').textContent = this.value;
    $('wi-year').value = this.value;
  });
});

// ── What-If prefill ───────────────────────────────────────────────
function prefillWhatIf() {
  const d = window._lastScore; if (!d) return;
  const note = $('wi-prefill-note');
  if (note) note.style.display = 'block';
  if ($('wi-industry')) $('wi-industry').value = d.industry;
  if ($('wi-country'))  $('wi-country').value  = d.country;
  $('wi-year').value    = d.year;
  $('wi-funding').value = d.funding;
  $('wi-fund-slider').value        = 0;
  $('wi-year-slider').value        = d.year;
  $('wi-fund-display').textContent = fmt(0);
  $('wi-year-display').textContent = d.year;
  setProfile('wi', d.userType || 'investor');
}

async function runWhatIf() {
  if (!window._lastScoreData) { alert('Please score a venture in the Score tab first.'); return; }
  const industry = $('wi-industry').value;
  const country  = $('wi-country').value;
  const year     = parseInt($('wi-year').value);
  const userType = $('wi-user-type').value;
  const modFunding = parseFloat($('wi-fund-slider').value) || 0;
  const modYear    = parseInt($('wi-year-slider').value)   || year;

  const btn = document.querySelector('#tab-whatif-panel .a-btn-calc');
  btn.textContent = '⏳ Running…'; btn.disabled = true;

  try {
    const baseResult = { ...window._lastScoreData, name: 'Base' };
    const modResult  = await fetch(`${API}/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name:'Modified', industry, country,
        founded_year: modYear, funding_total_usd: modFunding,
        funding_rounds: 1, user_type: userType }),
    }).then(r => r.json());
    const delta = Math.round((modResult.score - baseResult.score) * 100) / 100;
    renderWhatIf('whatif-result', { original: baseResult, modified: modResult, score_delta: delta });
  } catch(e) {
    $('whatif-result').innerHTML = `<div class="a-card" style="color:#ef4444">Error: ${e.message}</div>`;
  } finally { btn.textContent = 'Run Scenario'; btn.disabled = false; }
}

// ── Compare ───────────────────────────────────────────────────────
let _cmpCount = 0;
function initCompare() { addCompareCard(); addCompareCard(); }

function addCompareCard() {
  _cmpCount++;
  const id  = `cmp-${_cmpCount}`;
  const div = document.createElement('div');
  div.className = 'a-card cmp-card'; div.id = id;
  const indOpts = (window._industries||[]).map(i=>`<option value="${i}">${i}</option>`).join('');
  const ctrOpts = (window._countries||[]).map(c=>`<option value="${c}">${c}</option>`).join('');
  div.innerHTML = `
    <div class="cmp-header">
      <span class="cmp-num">#${_cmpCount}</span>
      <button class="cmp-remove" onclick="removeCompareCard('${id}')">✕</button>
    </div>
    <div class="a-fg"><label>VENTURE NAME</label>
      <input type="text" id="${id}-name" value="Venture ${_cmpCount}"/></div>
    <div class="a-form-row">
      <div class="a-fg"><label>INDUSTRY</label>
        <select id="${id}-industry"><option value="">Select...</option>${indOpts}</select></div>
      <div class="a-fg"><label>COUNTRY</label>
        <select id="${id}-country"><option value="">Select...</option>${ctrOpts}</select></div>
    </div>
    <div class="a-form-row">
      <div class="a-fg"><label>YEAR</label>
        <input type="number" id="${id}-year" value="2019" min="2000" max="2030"/></div>
      <div class="a-fg"><label>FUNDING (USD)</label>
        <input type="number" id="${id}-funding" value="500000" min="0"/></div>
    </div>`;
  $('compare-cards').appendChild(div);
}

function removeCompareCard(id) { const el=$(id); if(el) el.remove(); }

async function runCompare() {
  const userType = $('cmp-user-type').value;
  const opps = [];
  document.querySelectorAll('.cmp-card').forEach(card => {
    const id  = card.id;
    const ind = $(`${id}-industry`)?.value;
    const ctr = $(`${id}-country`)?.value;
    if (!ind || !ctr) return;
    opps.push({
      name:              $(`${id}-name`)?.value    || ind,
      industry:          ind, country: ctr,
      founded_year:      parseInt($(`${id}-year`)?.value) || 2019,
      funding_total_usd: parseFloat($(`${id}-funding`)?.value) || 0,
      funding_rounds:    1, user_type: userType,
    });
  });

  if (opps.length < 2) { alert('Fill in at least 2 opportunities.'); return; }

  const btn = document.querySelector('#tab-compare-panel .a-btn-calc');
  btn.textContent = '⏳ Comparing…'; btn.disabled = true;

  try {
    const data = await fetch(`${API}/compare`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(opps),
    }).then(r => r.json());
    renderCompare('compare-result', data);
  } catch(e) {
    $('compare-result').innerHTML = `<div class="a-card" style="color:#ef4444">Error: ${e.message}</div>`;
  } finally { btn.textContent = 'Compare All'; btn.disabled = false; }
}

// ── Render ────────────────────────────────────────────────────────
function profileBadge(userType) {
  const label = userType === 'entrepreneur' ? '🚀 Entrepreneur' : '💼 Investor';
  return `<span class="profile-badge profile-badge-${userType}">${label}</span>`;
}

function ringHtml(score, color) {
  const c = 2 * Math.PI * 40;
  const d = (score/100) * c;
  return `<svg class="r-ring" viewBox="0 0 100 100">
    <circle cx="50" cy="50" r="40" fill="none" stroke="#e9e5fe" stroke-width="8"/>
    <circle cx="50" cy="50" r="40" fill="none" stroke="${color}" stroke-width="8"
      stroke-dasharray="${d} ${c}" stroke-linecap="round" transform="rotate(-90 50 50)"/>
  </svg>`;
}

function featureIcon(name) {
  if (name.includes('trend'))    return '📈';
  if (name.includes('funding') || name.includes('tier') || name.includes('log')) return '💰';
  if (name.includes('gdp') || name.includes('econ') || name.includes('inflation') || name.includes('real')) return '🌐';
  if (name.includes('reddit'))   return '💬';
  if (name.includes('news') || name.includes('sentiment')) return '📰';
  if (name.includes('startup') || name.includes('ecosystem') || name.includes('density')) return '🏙️';
  if (name.includes('market'))   return '📊';
  if (name.includes('country'))  return '🌍';
  if (name.includes('cat_'))     return '🏷️';
  return '⚡';
}

function renderResult(cid, data) {
  const el = $(cid); if (!el) return;
  el.style.display = 'block';
  if (data.error) { el.innerHTML=`<div class="a-card"><p style="color:#ef4444">${data.error}</p></div>`; return; }

  const tc       = tierColor(data.tier);
  const ringClr  = tc==='green'?'#22c55e':tc==='red'?'#ef4444':'#e09d2a';
  const bench    = data.benchmark || {};
  const maxImpact = Math.max(...(data.top_factors||[]).map(f => f.abs_impact), 0.1);

  const shapHtml = (data.top_factors||[]).slice(0,8).map((f,i) => {
    const pct = Math.min(96, (f.abs_impact / maxImpact) * 96);
    const dir = f.direction==='positive' ? 'pos' : 'neg';
    const icon = featureIcon(f.feature);
    const rank = i === 0 ? '<span class="shap-top-badge">TOP DRIVER</span>' : '';
    return `<div class="r-factor ${dir}-factor">
      <div class="r-factor-header">
        <span class="r-factor-icon">${icon}</span>
        <span class="r-factor-name">${f.feature.replace(/_/g,' ')}</span>
        ${rank}
        <span class="r-factor-val ${dir}">${f.direction==='positive'?'▲':'▼'} ${f.shap_value>0?'+':''}${f.shap_value.toFixed(3)}</span>
      </div>
      <div class="r-bar-track">
        <div class="r-bar-fill ${dir}" style="width:${pct}%">
          <span class="r-bar-pct">${f.shap_value>0?'+':''}${f.shap_value.toFixed(2)}</span>
        </div>
      </div>
      <div class="r-factor-exp">${f.explanation}</div>
    </div>`;
  }).join('');

  const recsHtml = (data.recommendations||[]).map(r => `
    <div class="r-rec">
      <div class="r-rec-problem"><span class="r-rec-warn">⚠</span> ${r.problem}</div>
      <div class="r-rec-action"><span class="r-rec-arrow">→</span> ${r.recommendation}</div>
      ${r.potential_gain?`<div class="r-rec-gain"><span>📈</span> ${r.potential_gain}</div>`:''}
    </div>`).join('');

  const gdpHtml = data.gdp_context ? `
    <div class="r-gdp-context">
      <div class="r-gdp-header">
        <span>🌍</span>
        <span class="r-gdp-label">GDP ECONOMIC CONTEXT</span>
        <span class="quality-badge quality-${(data.quality_label||'medium').toLowerCase()}">${qualityDot(data.quality_label)}</span>
      </div>
      <p class="r-gdp-text">${data.gdp_context}</p>
    </div>` : '';

  el.innerHTML = `
    <div class="a-card result-card">
      <div class="r-header">
        <div>
          <div class="r-venture-name">${data.name}</div>
          <div class="r-meta">${profileBadge(data.user_type)} <span class="r-tier-badge ${tc}">● ${data.tier}</span></div>
        </div>
        <div class="r-ring-wrap">${ringHtml(data.score, ringClr)}<div class="r-ring-val">${data.score}<span>/100</span></div></div>
      </div>
      <div class="r-stats">
        <div class="r-stat"><div class="r-stat-lbl">CONFIDENCE</div><div class="r-stat-val">${Math.round((data.confidence||0)*100)}%</div></div>
        <div class="r-stat"><div class="r-stat-lbl">RANGE</div><div class="r-stat-val">${data.score_range?data.score_range[0]+'–'+data.score_range[1]:'—'}</div></div>
        <div class="r-stat"><div class="r-stat-lbl">BASE VALUE</div><div class="r-stat-val">${data.base_value}</div></div>
        <div class="r-stat"><div class="r-stat-lbl">PERCENTILE</div><div class="r-stat-val">${bench.your_percentile||0}th</div></div>
      </div>
      ${gdpHtml}
      <div class="r-section">
        <div class="r-section-title">SHAP Feature Attribution <span class="r-shap-sub">— why this score?</span></div>
        <div class="r-factors">${shapHtml}</div>
      </div>
      ${recsHtml?`<div class="r-section recs-section"><div class="r-section-title">How to Improve Your Score</div>${recsHtml}</div>`:''}
      <div class="r-section bench-section"><div class="r-section-title">Benchmark</div>
        <div class="r-bench">
          <div class="r-bench-row">
            <div class="r-bench-stat"><div class="r-bench-val">${bench.your_percentile||0}th</div><div class="r-bench-lbl">PERCENTILE</div></div>
            <div class="r-bench-stat"><div class="r-bench-val">${bench.category_avg_score||'—'}</div><div class="r-bench-lbl">INDUSTRY AVG</div></div>
            <div class="r-bench-stat"><div class="r-bench-val">${(bench.successful_ventures||[]).length}</div><div class="r-bench-lbl">HIGH-SCORE PEERS</div></div>
          </div>
        </div>
      </div>
    </div>`;
  el.scrollIntoView({ behavior:'smooth', block:'nearest' });
}

function renderWhatIf(cid, data) {
  const el = $(cid); if (!el) return;
  const orig = data.original||{}, mod = data.modified||{};
  const delta = data.score_delta||0;
  const dCls  = delta>0?'delta-pos':delta<0?'delta-neg':'delta-zero';
  el.innerHTML = `
    <div class="wi-compare">
      <div class="a-card wi-side">
        <div class="wi-label">BASE</div>${profileBadge(orig.user_type||'investor')}
        <div class="wi-score">${orig.score}<span>/100</span></div>
        <div class="wi-tier ${tierColor(orig.tier||'')}">● ${orig.tier||'—'}</div>
        ${orig.gdp_context?`<div class="wi-gdp"><span class="gdp-tag">🌍</span> ${orig.gdp_context}</div>`:''}
        <div class="wi-factors">${(orig.top_factors||[]).slice(0,5).map(f=>`
          <div class="wi-factor ${f.direction}">${f.direction==='positive'?'▲':'▼'} ${f.feature.replace(/_/g,' ')} <span>${f.shap_value>0?'+':''}${f.shap_value.toFixed(3)}</span></div>`).join('')}</div>
      </div>
      <div class="wi-delta-panel"><div class="wi-delta-val ${dCls}">${delta>0?'+':''}${delta} pts</div><div class="wi-delta-lbl">score change</div></div>
      <div class="a-card wi-side">
        <div class="wi-label">MODIFIED</div>${profileBadge(mod.user_type||'investor')}
        <div class="wi-score">${mod.score}<span>/100</span></div>
        <div class="wi-tier ${tierColor(mod.tier||'')}">● ${mod.tier||'—'}</div>
        ${mod.gdp_context?`<div class="wi-gdp"><span class="gdp-tag">🌍</span> ${mod.gdp_context}</div>`:''}
        <div class="wi-factors">${(mod.top_factors||[]).slice(0,5).map(f=>`
          <div class="wi-factor ${f.direction}">${f.direction==='positive'?'▲':'▼'} ${f.feature.replace(/_/g,' ')} <span>${f.shap_value>0?'+':''}${f.shap_value.toFixed(3)}</span></div>`).join('')}</div>
      </div>
    </div>`;
}

function renderCompare(cid, data) {
  const el = $(cid); if (!el) return;
  const results = data.results||[], winner = data.winner||'';
  const cards = results.map(r => {
    const tc = tierColor(r.tier);
    const isW = r.name === winner;
    return `<div class="a-card cmp-result-card ${isW?'cmp-winner':''}">
      ${isW?'<div class="cmp-winner-badge">🏆 Best Score</div>':''}
      <div class="cmp-result-header">
        <div class="cmp-result-name">${r.name}</div>
        ${profileBadge(r.user_type||'investor')}
        <span class="r-tier-badge ${tc}">● ${r.tier}</span>
      </div>
      <div class="cmp-result-score">${r.score}<span>/100</span></div>
      ${r.gdp_context?`<div class="cmp-gdp"><span class="gdp-tag">🌍</span> ${r.gdp_context}</div>`:''}
      <div class="cmp-factors">${(r.top_factors||[]).slice(0,4).map(f=>`
        <div class="wi-factor ${f.direction}">${f.direction==='positive'?'▲':'▼'} ${f.feature.replace(/_/g,' ')} <span>${f.shap_value>0?'+':''}${f.shap_value.toFixed(3)}</span></div>`).join('')}</div>
    </div>`;
  }).join('');
  el.innerHTML = `<div class="cmp-results-grid">${cards}</div>`;
}
