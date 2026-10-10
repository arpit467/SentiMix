// Global Chart Instances
let singlePolarChart = null;
let singleBarChart = null;
let batchDonutChart = null;
let batchCategoryChart = null;
let batchLangChart = null;
let batchTermsChart = null;
let benchF1Chart = null;
let benchLatChart = null;

let selectedBatchFile = null;
let globalBatchAspectRecords = [];

// Preset Reviews
const PRESETS = {
  flipkart: "Display bahut vibrant hai 120Hz AMOLED aur battery life 2 days chalti hai lekin camera zoom night me bekar hai",
  amazon: "Voice recognition accurate hai Hindi and English me aur bass clarity super solid hai, fast delivery",
  meesho: "Fabric bahut achcha hai aur design bhi bahut pyara hai but pant ki fitting thodi loose hai",
  laptop: "Laptop ka boot time fast hai aur keyboard typing smooth hai par battery backup sirf 3 hours chalta hai",
  negative: "Ekdum ghatiya product hai mat khareedo total waste of money",
  beauty: "Fragrance bahut mild aur soothing hai, face glows nicely after applying but price thoda expensive laga"
};

// Tab Switching
function switchNavTab(tabId) {
  document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));

  const target = document.getElementById(`view-${tabId}`);
  const btn = document.getElementById(`tab-btn-${tabId}`);
  if (target) target.classList.add('active');
  if (btn) btn.classList.add('active');

  if (tabId === 'benchmarks') {
    loadBenchmarkHubData();
  } else if (tabId === 'datasets') {
    fetchDatasetRecords();
  }
}

// Preset Handler
function applyPreset(key) {
  if (PRESETS[key]) {
    document.getElementById('manual-review-text').value = PRESETS[key];
    document.getElementById('manual-custom-aspects').value = "";
    runManualAnalysis();
  }
}

// ----------------------------------------------------
// 1. MANUAL REVIEW ANALYZER
// ----------------------------------------------------
async function runManualAnalysis() {
  const text = document.getElementById('manual-review-text').value.trim();
  const modelName = document.getElementById('manual-model-select').value;
  const customAspects = document.getElementById('manual-custom-aspects').value.trim();
  const btn = document.getElementById('btn-manual-analyze');

  if (!text) {
    alert("Please enter a customer review to analyze.");
    return;
  }

  btn.innerHTML = '<span>⏳ Analyzing Sentiment...</span>';
  btn.disabled = true;

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        model_name: modelName,
        custom_aspects: customAspects
      })
    });

    if (!res.ok) {
      throw new Error(`Server returned error: ${res.statusText}`);
    }

    const data = await res.json();
    renderManualAnalysisResults(data);
  } catch (err) {
    console.error("Analysis failed:", err);
    alert("Error running sentiment analysis: " + err.message);
  } finally {
    btn.innerHTML = '<span>🚀 Analyze Review Sentiments</span>';
    btn.disabled = false;
  }
}

function renderManualAnalysisResults(data) {
  const section = document.getElementById('manual-results-section');
  section.style.display = 'block';

  // 1. Update KPI Values
  document.getElementById('kpi-aspects-count').textContent = data.aspects_found;
  
  const overallEl = document.getElementById('kpi-overall-sentiment');
  overallEl.textContent = data.overall_sentiment;
  if (data.overall_sentiment === 'Positive') {
    overallEl.style.color = 'var(--success)';
  } else if (data.overall_sentiment === 'Negative') {
    overallEl.style.color = 'var(--danger)';
  } else {
    overallEl.style.color = 'var(--neutral)';
  }

  document.getElementById('kpi-language-type').textContent = data.is_hinglish ? 'Hinglish (Code-Mixed)' : 'English (Standard)';
  document.getElementById('kpi-latency-val').textContent = `${data.latency_ms} ms`;

  // 2. Clause & Normalization Preview
  const clauseBox = document.getElementById('manual-clause-preview');
  let clauseHtml = `<div><b>Cleaned & Demojized Text:</b> <span style="color: #38bdf8;">"${data.cleaned_text}"</span></div>`;
  
  if (data.aspect_breakdown && data.aspect_breakdown.length > 0) {
    clauseHtml += `<div style="margin-top: 0.5rem;"><b>Segmented Clause Contexts:</b></div><div style="margin-top: 0.25rem;">`;
    data.aspect_breakdown.forEach(item => {
      clauseHtml += `<span class="clause-chip">🔍 <b>${item.aspect}</b> &rarr; "${item.clause_context}"</span> `;
    });
    clauseHtml += `</div>`;
  }
  clauseBox.innerHTML = clauseHtml;

  // 3. Render Aspect Breakdown Cards
  const grid = document.getElementById('manual-aspects-grid');
  grid.innerHTML = '';

  if (!data.aspect_breakdown || data.aspect_breakdown.length === 0) {
    grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 2rem;">No specific aspect keywords detected. Product-level sentiment evaluated.</div>';
  } else {
    data.aspect_breakdown.forEach(item => {
      const card = document.createElement('div');
      card.className = 'aspect-card';

      const pol = item.sentiment;
      const badgeClass = `badge-pill ${pol}`;
      const badgeIcon = pol === 'positive' ? '🟢 POSITIVE' : (pol === 'negative' ? '🔴 NEGATIVE' : '🟡 NEUTRAL');
      const confPct = (item.confidence * 100).toFixed(1);

      const probs = item.probabilities || { positive: 0.33, neutral: 0.34, negative: 0.33 };
      const posPct = ((probs.positive || 0) * 100).toFixed(1);
      const neuPct = ((probs.neutral || 0) * 100).toFixed(1);
      const negPct = ((probs.negative || 0) * 100).toFixed(1);
      const catTitle = (item.category || 'general').replace(/_/g, ' ').toUpperCase();

      card.innerHTML = `
        <div class="aspect-card-header">
          <div class="aspect-card-title">${item.aspect}</div>
          <div class="${badgeClass}">${badgeIcon}</div>
        </div>
        <div class="aspect-card-cat">Category: <b>${catTitle}</b> • Confidence: <b style="color: #38bdf8;">${confPct}%</b></div>
        <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.5rem; background: rgba(0,0,0,0.25); padding: 0.4rem 0.6rem; border-radius: 6px;">
          Context: <i>"${item.clause_context}"</i>
        </div>
        <div class="prob-list">
          <div class="prob-row">
            <span style="color: #34d399;">Positive</span>
            <div class="prob-track"><div class="prob-fill positive" style="width: ${posPct}%;"></div></div>
            <span style="text-align: right;">${posPct}%</span>
          </div>
          <div class="prob-row">
            <span style="color: #fbbf24;">Neutral</span>
            <div class="prob-track"><div class="prob-fill neutral" style="width: ${neuPct}%;"></div></div>
            <span style="text-align: right;">${neuPct}%</span>
          </div>
          <div class="prob-row">
            <span style="color: #fb7185;">Negative</span>
            <div class="prob-track"><div class="prob-fill negative" style="width: ${negPct}%;"></div></div>
            <span style="text-align: right;">${negPct}%</span>
          </div>
        </div>
      `;
      grid.appendChild(card);
    });
  }

  // 4. Render Dynamic Charts for THIS Particular Review
  renderSingleReviewCharts(data.aspect_breakdown);
}

function renderSingleReviewCharts(aspects) {
  if (!aspects || aspects.length === 0) return;

  const labels = aspects.map(a => a.aspect.toUpperCase());
  const confidences = aspects.map(a => (a.confidence * 100).toFixed(1));
  const posScores = aspects.map(a => ((a.probabilities?.positive || 0) * 100).toFixed(1));
  const neuScores = aspects.map(a => ((a.probabilities?.neutral || 0) * 100).toFixed(1));
  const negScores = aspects.map(a => ((a.probabilities?.negative || 0) * 100).toFixed(1));

  // Polar / Confidence Chart
  const ctxPolar = document.getElementById('singleReviewPolarChart');
  if (ctxPolar) {
    if (singlePolarChart) singlePolarChart.destroy();
    
    const colors = aspects.map(a => {
      if (a.sentiment === 'positive') return 'rgba(16, 185, 129, 0.75)';
      if (a.sentiment === 'negative') return 'rgba(244, 63, 94, 0.75)';
      return 'rgba(245, 158, 11, 0.75)';
    });

    singlePolarChart = new Chart(ctxPolar, {
      type: 'polarArea',
      data: {
        labels: labels,
        datasets: [{
          data: confidences,
          backgroundColor: colors,
          borderColor: '#0f172a',
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'right', labels: { color: '#cbd5e1', font: { family: 'Outfit', size: 11 } } }
        },
        scales: {
          r: {
            grid: { color: 'rgba(255, 255, 255, 0.08)' },
            ticks: { display: false },
            suggestedMin: 0,
            suggestedMax: 100
          }
        }
      }
    });
  }

  // Stacked Probabilities Bar Chart
  const ctxBar = document.getElementById('singleReviewBarChart');
  if (ctxBar) {
    if (singleBarChart) singleBarChart.destroy();

    singleBarChart = new Chart(ctxBar, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          { label: 'Positive %', data: posScores, backgroundColor: '#10b981' },
          { label: 'Neutral %', data: neuScores, backgroundColor: '#f59e0b' },
          { label: 'Negative %', data: negScores, backgroundColor: '#f43f5e' }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { color: '#cbd5e1', font: { family: 'Outfit', size: 11 } } }
        },
        scales: {
          x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
          y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } }
        }
      }
    });
  }
}

// ----------------------------------------------------
// 2. DATASET BATCH ABSA STUDIO
// ----------------------------------------------------
const dropZone = document.getElementById('dataset-drop-zone');
if (dropZone) {
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    }, false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length) {
      handleDatasetFile(files[0]);
    }
  });
}

function onDatasetFileSelected(e) {
  if (e.target.files && e.target.files.length > 0) {
    handleDatasetFile(e.target.files[0]);
  }
}

function handleDatasetFile(file) {
  selectedBatchFile = file;
  document.getElementById('drop-zone-title').innerHTML = `✅ Selected: <b>${file.name}</b> (${(file.size/1024).toFixed(1)} KB)`;
  document.getElementById('batch-config-bar').style.display = 'flex';
}

function loadSampleTestDataset() {
  fetch('/api/datasets?split=combined_test&limit=30')
    .then(r => r.json())
    .then(d => {
      if (d.records && d.records.length > 0) {
        // Create a mock CSV Blob
        const rows = ["review,category"];
        d.records.forEach(r => {
          const cleanSent = r.sentence.replace(/"/g, '""');
          rows.push(`"${cleanSent}","${r.domain}"`);
        });
        const csvContent = rows.join("\n");
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const file = new File([blob], "sample_ecommerce_reviews.csv", { type: "text/csv" });
        handleDatasetFile(file);
      }
    });
}

async function executeBatchAnalysis() {
  if (!selectedBatchFile) {
    alert("Please select or drop a dataset file first.");
    return;
  }

  const modelName = document.getElementById('batch-model-select').value;
  const textCol = document.getElementById('batch-text-column').value;
  const loading = document.getElementById('batch-loading-state');
  const dashboard = document.getElementById('batch-analytics-dashboard');

  loading.style.display = 'block';
  dashboard.style.display = 'none';

  const formData = new FormData();
  formData.append('file', selectedBatchFile);
  formData.append('model_name', modelName);
  formData.append('text_col', textCol);

  try {
    const res = await fetch('/api/batch', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }

    const data = await res.json();
    renderBatchDashboard(data);
  } catch (err) {
    console.error("Batch processing failed:", err);
    alert("Batch Processing Failed: " + err.message);
  } finally {
    loading.style.display = 'none';
  }
}

function renderBatchDashboard(data) {
  const dashboard = document.getElementById('batch-analytics-dashboard');
  dashboard.style.display = 'block';

  // 1. KPI Cards
  document.getElementById('batch-kpi-total-reviews').textContent = data.total_rows.toLocaleString();
  document.getElementById('batch-kpi-total-aspects').textContent = data.total_aspects.toLocaleString();
  document.getElementById('batch-kpi-hinglish-pct').textContent = `${data.hinglish_percentage}%`;
  
  const netEl = document.getElementById('batch-kpi-net-sentiment');
  netEl.textContent = (data.net_sentiment_score >= 0 ? `+${data.net_sentiment_score}` : `${data.net_sentiment_score}`) + '%';
  netEl.style.color = data.net_sentiment_score >= 0 ? 'var(--success)' : 'var(--danger)';

  document.getElementById('batch-kpi-avg-conf').textContent = `${data.avg_confidence}%`;
  document.getElementById('batch-chart-total-count').textContent = `${data.total_rows} Reviews`;

  // Download Button
  const dlBtn = document.getElementById('btn-download-enriched-csv');
  dlBtn.href = data.download_url;

  // 2. Render 4 Dedicated Charts For THIS Uploaded Dataset
  renderBatchCharts(data);

  // 3. Render Table
  globalBatchAspectRecords = data.aspect_records || [];
  renderBatchTable(globalBatchAspectRecords);
}

function renderBatchCharts(data) {
  // Chart 1: Sentiment Donut
  const ctxDonut = document.getElementById('batchSentimentDonutChart');
  if (ctxDonut) {
    if (batchDonutChart) batchDonutChart.destroy();
    
    const counts = data.sentiment_counts || {};
    batchDonutChart = new Chart(ctxDonut, {
      type: 'doughnut',
      data: {
        labels: ['Positive', 'Neutral', 'Negative'],
        datasets: [{
          data: [counts.Positive || 0, counts.Neutral || 0, counts.Negative || 0],
          backgroundColor: ['#10b981', '#f59e0b', '#f43f5e'],
          borderColor: '#0f172a',
          borderWidth: 3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: '#cbd5e1', font: { family: 'Outfit', size: 12 } } }
        },
        cutout: '68%'
      }
    });
  }

  // Chart 2: Top Aspect Categories Stacked Bar
  const ctxCat = document.getElementById('batchAspectCategoryBarChart');
  if (ctxCat) {
    if (batchCategoryChart) batchCategoryChart.destroy();

    const topCats = data.top_categories || [];
    const labels = topCats.map(c => c.category);
    const pos = topCats.map(c => c.positive);
    const neu = topCats.map(c => c.neutral);
    const neg = topCats.map(c => c.negative);

    batchCategoryChart = new Chart(ctxCat, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          { label: 'Positive', data: pos, backgroundColor: '#10b981' },
          { label: 'Neutral', data: neu, backgroundColor: '#f59e0b' },
          { label: 'Negative', data: neg, backgroundColor: '#f43f5e' }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {
          legend: { position: 'top', labels: { color: '#cbd5e1', font: { family: 'Outfit', size: 11 } } }
        },
        scales: {
          x: { stacked: true, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
          y: { stacked: true, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } }
        }
      }
    });
  }

  // Chart 3: Language Pie
  const ctxLang = document.getElementById('batchLanguagePieChart');
  if (ctxLang) {
    if (batchLangChart) batchLangChart.destroy();

    batchLangChart = new Chart(ctxLang, {
      type: 'pie',
      data: {
        labels: ['Hinglish (Code-Mixed)', 'Standard English / Hindi'],
        datasets: [{
          data: [data.hinglish_percentage, data.pure_lang_percentage],
          backgroundColor: ['#6366f1', '#06b6d4'],
          borderColor: '#0f172a',
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: '#cbd5e1', font: { family: 'Outfit', size: 11 } } }
        }
      }
    });
  }

  // Chart 4: Top Aspect Terms
  const ctxTerms = document.getElementById('batchAspectTermsBarChart');
  if (ctxTerms) {
    if (batchTermsChart) batchTermsChart.destroy();

    const terms = data.top_aspect_terms || [];
    const labels = terms.map(t => t.aspect.toUpperCase());
    const counts = terms.map(t => t.total);

    batchTermsChart = new Chart(ctxTerms, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Frequency',
          data: counts,
          backgroundColor: 'rgba(99, 102, 241, 0.8)',
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
          y: { beginAtZero: true, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } }
        }
      }
    });
  }
}

function renderBatchTable(records) {
  const tbody = document.getElementById('batch-data-table-body');
  tbody.innerHTML = '';

  if (!records || records.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No records found.</td></tr>';
    return;
  }

  records.forEach((r, i) => {
    const tr = document.createElement('tr');
    const badgeClass = `badge-pill ${r.sentiment}`;
    const badgeText = r.sentiment.toUpperCase();
    const confPct = ((r.confidence || 0.85) * 100).toFixed(1);
    const langStr = r.is_hinglish ? 'Hinglish' : 'English';

    tr.innerHTML = `
      <td>${i + 1}</td>
      <td style="max-width: 320px; font-size: 0.85rem;">${r.review}</td>
      <td><b style="color: #fff;">${r.aspect}</b></td>
      <td><span class="badge-mini">${r.category}</span></td>
      <td><span class="${badgeClass}">${badgeText}</span></td>
      <td><span style="color: #38bdf8;">${confPct}%</span></td>
      <td><span style="font-size: 0.8rem; color: var(--text-muted);">${langStr}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function filterBatchTable() {
  const search = document.getElementById('batch-table-search').value.toLowerCase().trim();
  const polarity = document.getElementById('batch-table-polarity-filter').value;

  const filtered = globalBatchAspectRecords.filter(r => {
    const matchesSearch = !search || r.review.toLowerCase().includes(search) || r.aspect.toLowerCase().includes(search) || r.category.toLowerCase().includes(search);
    const matchesPol = (polarity === 'all') || (r.sentiment.toLowerCase() === polarity);
    return matchesSearch && matchesPol;
  });

  renderBatchTable(filtered);
}

// ----------------------------------------------------
// 3. 5-MODEL BENCHMARK HUB
// ----------------------------------------------------
async function loadBenchmarkHubData() {
  try {
    const res = await fetch('/api/benchmarks');
    const data = await res.json();
    const tableData = data.table || [];

    if (tableData.length > 0) {
      const labels = tableData.map(r => r['Model'].replace(' + Logistic Regression', ' + LR').replace('Baseline', '').trim());
      const macroF1 = tableData.map(r => r['Macro F1 (%)']);
      const hinglishF1 = tableData.map(r => r['Hinglish F1 (%)']);
      const latencies = tableData.map(r => r['Latency (ms/item)']);

      // Macro F1 Chart
      const ctxF1 = document.getElementById('benchmarkF1Chart');
      if (ctxF1) {
        if (benchF1Chart) benchF1Chart.destroy();
        benchF1Chart = new Chart(ctxF1, {
          type: 'bar',
          data: {
            labels: labels,
            datasets: [
              { label: 'Macro F1 (%)', data: macroF1, backgroundColor: '#6366f1', borderRadius: 6 },
              { label: 'Hinglish Subset F1 (%)', data: hinglishF1, backgroundColor: '#06b6d4', borderRadius: 6 }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { position: 'top', labels: { color: '#cbd5e1', font: { family: 'Outfit', size: 12 } } }
            },
            scales: {
              x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
              y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } }
            }
          }
        });
      }

      // Latency Chart
      const ctxLat = document.getElementById('benchmarkLatencyChart');
      if (ctxLat) {
        if (benchLatChart) benchLatChart.destroy();
        const colors = ['#10b981', '#6366f1', '#06b6d4', '#f59e0b', '#f43f5e'];
        benchLatChart = new Chart(ctxLat, {
          type: 'bar',
          data: {
            labels: labels,
            datasets: [{
              label: 'Latency (ms/item)',
              data: latencies,
              backgroundColor: colors.slice(0, labels.length),
              borderRadius: 6
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
              y: { beginAtZero: true, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } }
            }
          }
        });
      }
    }
  } catch (err) {
    console.error("Benchmark fetch error:", err);
  }
}

// ----------------------------------------------------
// 4. PRELOADED DATASET EXPLORER
// ----------------------------------------------------
async function fetchDatasetRecords() {
  const split = document.getElementById('dataset-split-selector').value;
  const polarity = document.getElementById('dataset-polarity-selector').value;
  const tbody = document.getElementById('preloaded-datasets-table-body');

  tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 2rem;">Loading dataset records...</td></tr>';

  try {
    const res = await fetch(`/api/datasets?split=${split}&polarity=${polarity}&limit=40`);
    const data = await res.json();
    tbody.innerHTML = '';

    if (!data.records || data.records.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No records found.</td></tr>';
      return;
    }

    data.records.forEach(r => {
      const tr = document.createElement('tr');
      const badgeClass = `badge-pill ${r.polarity}`;
      const langStr = r.is_hinglish ? 'Hinglish' : 'English';

      tr.innerHTML = `
        <td style="max-width: 480px; font-size: 0.88rem;">${r.sentence}</td>
        <td><b style="color: #fff;">${r.aspect}</b></td>
        <td><span class="${badgeClass}">${r.polarity.toUpperCase()}</span></td>
        <td><span style="font-size: 0.8rem; color: var(--text-muted);">${langStr}</span></td>
        <td><span class="badge-mini">${r.domain}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Dataset records fetch error:", err);
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--danger);">Error: ${err.message}</td></tr>`;
  }
}

// Initial Auto-Execution on Page Load
document.addEventListener('DOMContentLoaded', () => {
  runManualAnalysis();
});
