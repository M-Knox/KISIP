"""App-wide constants."""

ACCENT = "#4FC3A1"
ACCENT_MUKURU = "#F5A623"
TEXT = "#E8EAF0"
TEXT_MUTED = "#8B95A8"
SURFACE = "#1A1F2E"
BG = "#0F1117"

MODEL_OPTIONS = ["Ridge Regression", "Random Forest", "XGBoost"]

MODEL_COLUMNS = {
    "Ridge Regression": {"kisip": "ridge_pred", "mukuru": "ridge_scmi"},
    "Random Forest": {"kisip": "rf_pred", "mukuru": "rf_scmi"},
    "XGBoost": {"kisip": "xgb_pred", "mukuru": "xgb_scmi"},
}

FEATURES = [
    "NDVI", "NDBI", "MNDWI", "Contrast", "Entropy",
    "Homogeneity", "Correlation", "road_density", "paved_proportion",
]

TIER_COLORS = {
    "High Readiness": ACCENT,
    "Moderate Readiness": ACCENT_MUKURU,
    "Low Readiness": "#E74C3C",
}

SCMI_PERIODS = "2009–2011 and 2021–2023"
