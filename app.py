from flask import Flask, render_template, request, jsonify, make_response
import pandas as pd
import numpy as np
import joblib
import json
import os
import math
# Importar módulos de sklearn necesarios para cargar el modelo
import sklearn.linear_model
import sklearn.compose
import sklearn.pipeline
import sklearn.impute
import sklearn.preprocessing

app = Flask(__name__)

def safe_json(obj):
    """Convierte NaN/Inf a null para JSON válido."""
    def _clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    def _walk(o):
        if isinstance(o, dict):  return {k: _walk(v) for k, v in o.items()}
        if isinstance(o, list):  return [_walk(v) for v in o]
        return _clean(o)
    resp = make_response(json.dumps(_walk(obj)))
    resp.headers['Content-Type'] = 'application/json'
    return resp

# ── Carga inicial ─────────────────────────────────────────────────────────────
def load_data():
    ctx = {}
    try:
        ctx['modelo']      = joblib.load('modelo_lineal.pkl')
        ctx['resultados']  = pd.read_csv('resultados_modelos.csv').to_dict(orient='records')
        ctx['coeficientes']= pd.read_csv('coeficientes_modelo_lineal.csv').to_dict(orient='records')
        ctx['X_template']  = pd.read_csv('template_prediccion.csv')
        ctx['model_ready'] = True
    except Exception as e:
        print(f"[WARN] Modelo no listo: {e}")
        ctx['modelo'] = ctx['X_template'] = None
        ctx['resultados'] = ctx['coeficientes'] = []
        ctx['model_ready'] = False

    try:
        df = pd.read_csv('listings.csv', low_memory=False)
        df['price_clean'] = pd.to_numeric(
            df['price'].astype(str).str.replace(r'[\$,]', '', regex=True),
            errors='coerce'
        )
        ctx['df_full'] = df

        df_map = df.dropna(subset=['latitude','longitude','price_clean']).copy()
        df_map = df_map[df_map['price_clean'] > 0]
        ctx['df_map'] = df_map.sample(min(3000, len(df_map)), random_state=42)

        ctx['neighbourhoods'] = sorted(df['neighbourhood_cleansed'].dropna().unique().tolist())
        ctx['room_types']     = sorted(df['room_type'].dropna().unique().tolist())
        ctx['property_types'] = sorted(df['property_type'].dropna().unique().tolist())

        valid = df['price_clean'].dropna()
        valid = valid[valid > 0]
        ctx['kpis'] = {
            'n_listings':       len(df),
            'median_price':     round(valid.median(), 0),
            'avg_price':        round(valid.mean(), 0),
            'n_neighbourhoods': df['neighbourhood_cleansed'].nunique(),
        }

        dleg = df.groupby('neighbourhood_cleansed')['price_clean'].agg(
            listings='count', avg_price='mean', median_price='median'
        ).reset_index().dropna()
        dleg['avg_price']    = dleg['avg_price'].round(0).astype(int)
        dleg['median_price'] = dleg['median_price'].round(0).astype(int)
        ctx['delegaciones_stats'] = dleg.sort_values('median_price', ascending=False).to_dict(orient='records')

    except Exception as e:
        print(f"[ERROR] listings.csv: {e}")
        ctx.setdefault('df_full', pd.DataFrame())
        ctx.setdefault('df_map', pd.DataFrame())
        ctx.setdefault('neighbourhoods', [])
        ctx.setdefault('room_types', [])
        ctx.setdefault('property_types', [])
        ctx.setdefault('kpis', {})
        ctx.setdefault('delegaciones_stats', [])

    return ctx

CTX = load_data()

# ── Rutas principales ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
        resultados       = CTX['resultados'],
        coeficientes     = CTX['coeficientes'],
        neighbourhoods   = CTX['neighbourhoods'],
        room_types       = CTX['room_types'],
        property_types   = CTX['property_types'],
        delegaciones_stats = CTX['delegaciones_stats'],
        kpis             = CTX['kpis'],
        model_ready      = CTX['model_ready'],
    )

# ── API: datos del mapa ───────────────────────────────────────────────────────
@app.route('/api/map-data')
def map_data():
    df_map = CTX.get('df_map', pd.DataFrame())
    if df_map.empty:
        return safe_json([])
    cols = ['latitude','longitude','price_clean','room_type',
            'neighbourhood_cleansed','name','accommodates',
            'bedrooms','beds','review_scores_rating',
            'host_is_superhost']
    available = [c for c in cols if c in df_map.columns]
    rec = df_map[available].rename(columns={
        'latitude':'lat','longitude':'lng',
        'price_clean':'price','neighbourhood_cleansed':'neighbourhood'
    })
    return safe_json(rec.to_dict(orient='records'))

# ── API: gráficas interactivas (Plotly JSON) ──────────────────────────────────
@app.route('/api/chart/correlation')
def chart_correlation():
    try:
        df = pd.read_csv('correlacion.csv').dropna()
        df = df.sort_values('corr', ascending=False)
        colors = ['#38bdf8' if v >= 0 else '#f87171' for v in df['corr']]
        return safe_json({
            'data': [{
                'type': 'bar',
                'x': df['feature'].tolist(),
                'y': df['corr'].round(4).tolist(),
                'marker': {'color': colors},
                'name': 'Correlacion',
            }],
            'layout': {
                'title': '',
                'xaxis': {'tickangle': -40, 'color': '#7d8fa8'},
                'yaxis': {'title': 'Correlacion con Price', 'color': '#7d8fa8'},
                'plot_bgcolor':  'rgba(0,0,0,0)',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': '#eef4ff', 'family': 'Inter'},
                'margin': {'t': 20, 'b': 120, 'l': 60, 'r': 20},
            }
        })
    except Exception as e:
        return safe_json({'error': str(e)})

@app.route('/api/chart/boxplot')
def chart_boxplot():
    try:
        df = pd.read_csv('price_sample.csv').dropna()
        return jsonify({
            'data': [{
                'type': 'box',
                'y': df['price'].tolist(),
                'name': 'Precio',
                'marker': {'color': '#38bdf8'},
                'boxpoints': 'outliers',
                'line': {'color': '#818cf8'},
                'fillcolor': 'rgba(56,189,248,0.15)',
            }],
            'layout': {
                'title': '',
                'yaxis': {'title': 'Precio (MXN/noche)', 'color': '#7d8fa8'},
                'xaxis': {'color': '#7d8fa8'},
                'plot_bgcolor':  'rgba(0,0,0,0)',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': '#eef4ff', 'family': 'Inter'},
                'margin': {'t': 20, 'b': 30, 'l': 60, 'r': 20},
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart/scatter-lineal')
def chart_scatter_lineal():
    try:
        df = pd.read_csv('scatter_lineal.csv').dropna()
        max_val = float(df['real'].quantile(0.99))
        return jsonify({
            'data': [
                {
                    'type': 'scatter', 'mode': 'markers',
                    'x': df['real'].tolist(), 'y': df['pred'].tolist(),
                    'name': 'Listings',
                    'marker': {'color': '#818cf8', 'opacity': 0.5, 'size': 4},
                },
                {
                    'type': 'scatter', 'mode': 'lines',
                    'x': [0, max_val], 'y': [0, max_val],
                    'name': 'Predicción perfecta',
                    'line': {'color': '#f87171', 'dash': 'dash', 'width': 1.5},
                }
            ],
            'layout': {
                'title': '',
                'xaxis': {'title': 'Valor real', 'color': '#7d8fa8', 'range': [0, max_val]},
                'yaxis': {'title': 'Valor predicho', 'color': '#7d8fa8', 'range': [0, max_val]},
                'plot_bgcolor':  'rgba(0,0,0,0)',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': '#eef4ff', 'family': 'Inter'},
                'legend': {'font': {'size': 10}},
                'margin': {'t': 20, 'b': 50, 'l': 60, 'r': 20},
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart/scatter-poly2')
def chart_scatter_poly2():
    try:
        df = pd.read_csv('scatter_poly2.csv').dropna()
        max_val = float(df['real'].quantile(0.99))
        return jsonify({
            'data': [
                {
                    'type': 'scatter', 'mode': 'markers',
                    'x': df['real'].tolist(), 'y': df['pred'].tolist(),
                    'name': 'Listings',
                    'marker': {'color': '#f472b6', 'opacity': 0.5, 'size': 4},
                },
                {
                    'type': 'scatter', 'mode': 'lines',
                    'x': [0, max_val], 'y': [0, max_val],
                    'name': 'Predicción perfecta',
                    'line': {'color': '#f87171', 'dash': 'dash', 'width': 1.5},
                }
            ],
            'layout': {
                'title': '',
                'xaxis': {'title': 'Valor real', 'color': '#7d8fa8', 'range': [0, max_val]},
                'yaxis': {'title': 'Valor predicho', 'color': '#7d8fa8', 'range': [0, max_val]},
                'plot_bgcolor':  'rgba(0,0,0,0)',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': '#eef4ff', 'family': 'Inter'},
                'legend': {'font': {'size': 10}},
                'margin': {'t': 20, 'b': 50, 'l': 60, 'r': 20},
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart/coefficients')
def chart_coefficients():
    try:
        df = pd.read_csv('coeficientes_modelo_lineal.csv')
        top = pd.concat([df.head(10), df.tail(10)]).drop_duplicates()
        colors = ['#38bdf8' if v >= 0 else '#f87171' for v in top['Coeficiente']]
        return jsonify({
            'data': [{
                'type': 'bar', 'orientation': 'h',
                'x': top['Coeficiente'].tolist(),
                'y': top['Variable'].tolist(),
                'marker': {'color': colors},
                'name': 'Coeficiente',
            }],
            'layout': {
                'title': '',
                'xaxis': {'title': 'Coeficiente', 'color': '#7d8fa8'},
                'yaxis': {'color': '#7d8fa8', 'automargin': True},
                'plot_bgcolor':  'rgba(0,0,0,0)',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': '#eef4ff', 'family': 'Inter', 'size': 11},
                'margin': {'t': 20, 'b': 40, 'l': 220, 'r': 20},
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart/price-by-neighbourhood')
def chart_price_neighbourhood():
    try:
        df = pd.read_csv('price_by_neighbourhood.csv').dropna().sort_values('median_price', ascending=True)
        return jsonify({
            'data': [{
                'type': 'bar', 'orientation': 'h',
                'x': df['median_price'].tolist(),
                'y': df['neighbourhood'].tolist(),
                'marker': {
                    'color': df['median_price'].tolist(),
                    'colorscale': [[0,'#4ade80'],[0.5,'#facc15'],[1,'#f87171']],
                    'showscale': False,
                },
                'name': 'Precio mediano',
            }],
            'layout': {
                'title': '',
                'xaxis': {'title': 'Precio mediano (MXN/noche)', 'color': '#7d8fa8'},
                'yaxis': {'color': '#7d8fa8', 'automargin': True},
                'plot_bgcolor':  'rgba(0,0,0,0)',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': '#eef4ff', 'family': 'Inter'},
                'margin': {'t': 20, 'b': 40, 'l': 210, 'r': 20},
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── API: predicción ───────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if not CTX.get('model_ready'):
        return jsonify({'error': 'Modelo no disponible. Ejecuta modelo.py primero.'}), 500
    try:
        data = request.json
        df_input = CTX['X_template'].copy()
        str_map = {
            'neighbourhood_cleansed': 'neighbourhood',
            'room_type': 'room_type',
            'property_type': 'property_type',
            'host_response_time': 'host_response_time',
        }
        for col, key in str_map.items():
            if col in df_input.columns and key in data:
                df_input[col] = data[key]
        for col in ['accommodates','bedrooms','beds','bathrooms','minimum_nights','availability_365']:
            if col in df_input.columns and col in data:
                try:
                    df_input[col] = float(data[col])
                except (ValueError, TypeError):
                    pass
        pred = CTX['modelo'].predict(df_input)[0]
        return jsonify({'prediction': round(max(pred, 0), 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
