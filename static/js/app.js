/* ── Plotly chart config ─────────────────────────────── */
const PLOTLY_CONFIG = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ['select2d','lasso2d','toImage'],
  displaylogo: false,
};

async function loadChart(endpoint, divId) {
  try {
    const res  = await fetch(endpoint);
    const spec = await res.json();
    if (spec.error) { console.error(divId, spec.error); return; }
    Plotly.newPlot(divId, spec.data, spec.layout, PLOTLY_CONFIG);
  } catch(e) { console.error('Chart error', divId, e); }
}

// Load all 6 interactive charts
loadChart('/api/chart/correlation',          'chart-correlation');
loadChart('/api/chart/boxplot',              'chart-boxplot');
loadChart('/api/chart/price-by-neighbourhood','chart-neighbourhood');
loadChart('/api/chart/coefficients',         'chart-coef');
loadChart('/api/chart/scatter-lineal',       'chart-scatter-lin');
loadChart('/api/chart/scatter-poly2',        'chart-scatter-p2');

/* ── Prediction Form ─────────────────────────────────── */
document.getElementById('predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('submit-btn');
  btn.disabled = true;
  btn.textContent = 'Calculating...';

  const data = {
    room_type:        document.getElementById('room_type').value,
    neighbourhood:    document.getElementById('neighbourhood').value,
    accommodates:     document.getElementById('accommodates').value,
    bedrooms:         document.getElementById('bedrooms').value,
    beds:             document.getElementById('beds').value,
    minimum_nights:   document.getElementById('minimum_nights').value,
    availability_365: document.getElementById('availability_365').value,
  };

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const json = await res.json();
    if (json.error) {
      alert('Error: ' + json.error);
    } else {
      const box = document.getElementById('result-box');
      document.getElementById('price-result').textContent =
        '$' + json.prediction.toLocaleString('es-MX', { minimumFractionDigits: 2 });
      box.classList.remove('hidden');
    }
  } catch {
    alert('Error de conexión con el servidor.');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Estimate Price';
  }
});

/* ── Leaflet Map ─────────────────────────────────────── */
const map = L.map('map', { center: [19.42, -99.13], zoom: 11 });

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; <a href="https://carto.com">CARTO</a>',
  maxZoom: 18,
}).addTo(map);

function priceColor(price) {
  if (!price || price <= 0) return '#6b7fa0';
  if (price < 600)  return '#4ade80';
  if (price < 1500) return '#facc15';
  return '#f87171';
}

function buildPopup(d) {
  const price    = d.price ? `$${Math.round(d.price).toLocaleString('es-MX')}` : 'N/A';
  const superTag = (d.host_is_superhost === 1 || d.host_is_superhost === true)
    ? '<span class="popup-tag super">Superhost</span>' : '';
  const roomTag  = d.room_type ? `<span class="popup-tag">${d.room_type}</span>` : '';
  const rating   = d.review_scores_rating ? `<span>&#9733; ${Number(d.review_scores_rating).toFixed(1)}</span>` : '';
  const guests   = d.accommodates ? `<span>${d.accommodates} huéspedes</span>` : '';
  const beds     = d.bedrooms ? `<span>${d.bedrooms} recámaras</span>` : '';
  return `<div class="popup-inner">
    <div class="popup-neighbourhood">${d.neighbourhood || ''}</div>
    <div class="popup-price">${price} <span>MXN / noche</span></div>
    <div class="popup-tags">${roomTag}${superTag}</div>
    <div class="popup-meta">${rating}${guests}${beds}</div>
  </div>`;
}

fetch('/api/map-data')
  .then(r => r.json())
  .then(listings => {
    listings.forEach(d => {
      L.circleMarker([d.lat, d.lng], {
        radius: 5.5,
        fillColor: priceColor(d.price),
        color: 'rgba(0,0,0,0.3)',
        weight: 0.8,
        fillOpacity: 0.8,
      })
      .addTo(map)
      .bindPopup(buildPopup(d), { maxWidth: 260 });
    });
  })
  .catch(err => console.error('Map data error:', err));
