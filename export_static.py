from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import joblib
import pandas as pd

from app import app


ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
STATIC_DIR = ROOT / "static"
MODEL_JSON = STATIC_DIR / "model.json"

API_ROUTES = {
    "api/map-data": "/api/map-data",
    "api/chart/correlation": "/api/chart/correlation",
    "api/chart/boxplot": "/api/chart/boxplot",
    "api/chart/scatter-lineal": "/api/chart/scatter-lineal",
    "api/chart/scatter-poly2": "/api/chart/scatter-poly2",
    "api/chart/coefficients": "/api/chart/coefficients",
    "api/chart/price-by-neighbourhood": "/api/chart/price-by-neighbourhood",
}


def json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def clean_number(value):
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def export_browser_model() -> None:
    model = joblib.load(ROOT / "modelo_lineal.pkl")
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["reg"]

    numeric_pipeline = preprocessor.named_transformers_["num"]
    categorical_pipeline = preprocessor.named_transformers_["cat"]

    numeric_imputer = numeric_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]
    categorical_imputer = categorical_pipeline.named_steps["imputer"]
    onehot = categorical_pipeline.named_steps["onehot"]

    numeric_columns = list(preprocessor.transformers_[0][2])
    categorical_columns = list(preprocessor.transformers_[1][2])
    template = pd.read_csv(ROOT / "template_prediccion.csv").iloc[0].to_dict()

    browser_model = {
        "intercept": clean_number(regressor.intercept_),
        "coefficients": [clean_number(value) for value in regressor.coef_],
        "input_map": {
            "neighbourhood_cleansed": "neighbourhood",
            "room_type": "room_type",
            "property_type": "property_type",
            "host_response_time": "host_response_time",
            "accommodates": "accommodates",
            "bedrooms": "bedrooms",
            "beds": "beds",
            "bathrooms": "bathrooms",
            "minimum_nights": "minimum_nights",
            "availability_365": "availability_365",
        },
        "template": {key: json_value(value) for key, value in template.items()},
        "numeric": {
            "columns": numeric_columns,
            "imputer_statistics": [clean_number(value) for value in numeric_imputer.statistics_],
            "scaler_mean": [clean_number(value) for value in scaler.mean_],
            "scaler_scale": [clean_number(value) for value in scaler.scale_],
        },
        "categorical": {
            "columns": categorical_columns,
            "imputer_statistics": [json_value(value) for value in categorical_imputer.statistics_],
            "categories": [
                [json_value(category) for category in categories]
                for categories in onehot.categories_
            ],
        },
    }

    MODEL_JSON.write_text(
        json.dumps(browser_model, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )


def write_response(client, source_route: str, output_path: Path) -> None:
    response = client.get(source_route)
    if response.status_code >= 400:
        raise RuntimeError(f"{source_route} returned HTTP {response.status_code}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.get_data())


def export_static_site() -> None:
    export_browser_model()

    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    shutil.copytree(STATIC_DIR, DOCS_DIR / "static")

    client = app.test_client()
    write_response(client, "/", DOCS_DIR / "index.html")
    for output, route in API_ROUTES.items():
        write_response(client, route, DOCS_DIR / output)

    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")


if __name__ == "__main__":
    export_static_site()
    print(f"Static site exported to {DOCS_DIR}")
