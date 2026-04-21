/* ════════════════════════════════════════════════════
   APP.JS — Airbnb CDMX Dashboard
   ════════════════════════════════════════════════════ */

/* ── Plotly charts ────────────────────────────────── */
const PLOTLY_CFG = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'toImage'],
  displaylogo: false,
};

async function loadChart(endpoint, divId) {
  try {
    const res  = await fetch(endpoint);
    const spec = await res.json();
    if (spec.error) { console.error(divId, spec.error); return; }
    Plotly.newPlot(divId, spec.data, spec.layout, PLOTLY_CFG);
  } catch (e) { console.error('Chart error', divId, e); }
}

loadChart('/api/chart/correlation',           'chart-correlation');
loadChart('/api/chart/boxplot',               'chart-boxplot');
loadChart('/api/chart/price-by-neighbourhood','chart-neighbourhood');
loadChart('/api/chart/coefficients',          'chart-coef');
loadChart('/api/chart/scatter-lineal',        'chart-scatter-lin');
loadChart('/api/chart/scatter-poly2',         'chart-scatter-p2');

/* ── Prediction form ──────────────────────────────── */
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
      document.getElementById('price-result').textContent =
        '$' + json.prediction.toLocaleString('es-MX', { minimumFractionDigits: 2 });
      document.getElementById('result-box').classList.remove('hidden');
    }
  } catch {
    alert('Error de conexión con el servidor.');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Estimate Price';
  }
});

/* ════════════════════════════════════════════════════
   LEAFLET MAP
   ════════════════════════════════════════════════════ */
const CDMX = [19.4326, -99.1332];

const map = L.map('map', {
  center: CDMX,
  zoom: 11,
  minZoom: 9,
  maxZoom: 18,
  preferCanvas: true,
});

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; <a href="https://carto.com">CARTO</a> &copy; OpenStreetMap',
  maxZoom: 18,
}).addTo(map);

setTimeout(() => { map.invalidateSize(); }, 300);

/* ── Color helpers ────────────────────────────────── */
function listingColor(price) {
  if (!price || price <= 0) return '#6b7fa0';
  if (price < 600)  return '#4ade80';
  if (price < 1500) return '#facc15';
  return '#f87171';
}

function choroColor(t) {
  // t: 0 (barato, azul) -> 1 (caro, rojo)
  const stops = [
    [56,  189, 248],  // azul (0)
    [250, 204, 21],   // amarillo (0.5)
    [248, 113, 113],  // rojo (1)
  ];
  let a, b, u;
  if (t < 0.5) { a = stops[0]; b = stops[1]; u = t * 2; }
  else         { a = stops[1]; b = stops[2]; u = (t - 0.5) * 2; }
  const r = Math.round(a[0] + (b[0] - a[0]) * u);
  const g = Math.round(a[1] + (b[1] - a[1]) * u);
  const bl= Math.round(a[2] + (b[2] - a[2]) * u);
  return `rgba(${r},${g},${bl},0.42)`;
}

/* ── Listing popup ────────────────────────────────── */
function buildPopup(d) {
  const price    = d.price ? `$${Math.round(d.price).toLocaleString('es-MX')}` : 'N/A';
  const color    = listingColor(d.price);
  const superTag = (d.host_is_superhost === 1 || d.host_is_superhost === true)
    ? '<span class="popup-tag super">Superhost</span>' : '';
  const roomTag  = d.room_type ? `<span class="popup-tag">${d.room_type}</span>` : '';
  const rating   = d.review_scores_rating ? `<span>&#9733; ${Number(d.review_scores_rating).toFixed(1)}</span>` : '';
  const guests   = d.accommodates ? `<span>${d.accommodates} huéspedes</span>` : '';
  const beds     = d.bedrooms     ? `<span>${d.bedrooms} recámaras</span>` : '';
  return `<div class="popup-inner">
    <div class="popup-neighbourhood">${d.neighbourhood || ''}</div>
    <div class="popup-price" style="color:${color}">${price} <span>MXN / noche</span></div>
    <div class="popup-tags">${roomTag}${superTag}</div>
    <div class="popup-meta">${rating}${guests}${beds}</div>
  </div>`;
}

/* ── Load map data and GeoJSON in parallel ──────── */
Promise.all([
  fetch('/api/map-data').then(r => r.json()),
  fetch('/static/cdmx_alcaldias.geojson').then(r => r.json()),
]).then(([listings, geojson]) => {

  // ── Calcular estadísticas por alcaldía desde listings
  const statsMap = {};
  listings.forEach(d => {
    const n = d.neighbourhood;
    if (!n) return;
    if (!statsMap[n]) statsMap[n] = { prices: [], count: 0 };
    statsMap[n].count++;
    if (d.price > 0) statsMap[n].prices.push(d.price);
  });

  const byName = {};
  Object.keys(statsMap).forEach(k => {
    const p = statsMap[k].prices.sort((a, b) => a - b);
    const mid = Math.floor(p.length / 2);
    byName[k.toLowerCase().trim()] = {
      name:   k,
      count:  statsMap[k].count,
      median: p.length ? p[mid] : 0,
      avg:    p.length ? Math.round(p.reduce((s, v) => s + v, 0) / p.length) : 0,
    };
  });

  const medians = Object.values(byName).map(s => s.median).filter(Boolean);
  const minP = Math.min(...medians);
  const maxP = Math.max(...medians);

  // ── Choropleth de alcaldías
  let choroLayer;
  choroLayer = L.geoJSON(geojson, {
    style: feature => {
      const name = (feature.properties.NOMGEO || '').toLowerCase().trim();
      const st   = byName[name];
      const t    = st && maxP > minP ? (st.median - minP) / (maxP - minP) : null;
      return {
        fillColor:   t !== null ? choroColor(t) : 'rgba(255,255,255,0.05)',
        color:       'rgba(56,189,248,0.7)',
        weight:      1.8,
        fillOpacity: 1,
        dashArray:   null,
      };
    },
    onEachFeature: (feature, layer) => {
      const rawName = feature.properties.NOMGEO || 'Desconocida';
      const st = byName[rawName.toLowerCase().trim()];

      const ttHtml = st
        ? `<div class="tt-name">${rawName}</div>
           <div class="tt-row"><span>Listings</span><span class="tt-val">${st.count.toLocaleString('es-MX')}</span></div>
           <div class="tt-row"><span>Precio mediano</span><span class="tt-val">$${Math.round(st.median).toLocaleString('es-MX')}</span></div>
           <div class="tt-row"><span>Promedio</span><span class="tt-val">$${st.avg.toLocaleString('es-MX')}</span></div>`
        : `<div class="tt-name">${rawName}</div><div class="tt-row"><span style="color:#6b7fa0">Sin datos en muestra</span></div>`;

      layer.bindTooltip(ttHtml, {
        className: 'deleg-tooltip',
        sticky: true,
        direction: 'top',
        offset: [0, -6],
      });

      layer.on({
        mouseover: e => e.target.setStyle({ weight: 3, color: '#38bdf8', fillOpacity: 0.6 }),
        mouseout:  e => choroLayer.resetStyle(e.target),
      });
    },
  }).addTo(map);

  // ── Listings como puntos encima del choropleth
  listings.forEach(d => {
    L.circleMarker([d.lat, d.lng], {
      radius: 4,
      fillColor: listingColor(d.price),
      color: 'transparent',
      weight: 0,
      fillOpacity: 0.78,
    })
    .addTo(map)
    .bindPopup(buildPopup(d), { maxWidth: 260 });
  });

  // Ajustar vista al extent del GeoJSON
  map.fitBounds(choroLayer.getBounds(), { padding: [10, 10] });

}).catch(err => console.error('Map init error:', err));
