"""Cached data loaders."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import streamlit as st


@st.cache_data
def load_kisip_zones():
    gdf = gpd.read_file("data/kisip_zones_spatial.geojson").to_crs("EPSG:4326")
    gdf["type"] = "KISIP"
    return gdf


@st.cache_data
def load_mukuru_zones():
    gdf = gpd.read_file("data/mukuru_zones_spatial.geojson").to_crs("EPSG:4326")
    gdf["type"] = "Mukuru"
    return gdf


@st.cache_data
def load_kisip_predictions():
    return pd.read_csv("data/kisip_model_predictions.csv")


@st.cache_data
def load_mukuru_predictions():
    return pd.read_csv("data/kisip_mukuru_predictions.csv")


@st.cache_data
def load_kisip_page2_data():
    gdf = load_kisip_zones()
    preds = load_kisip_predictions()
    shap_set = pd.read_csv("data/kisip_shap_by_settlement.csv")
    features = pd.read_csv("data/kisip_baseline_features_9final.csv")
    scmi = pd.read_csv("data/kisip_zone_scmi_both.csv")
    return gdf, preds, shap_set, features, scmi


@st.cache_data
def load_mukuru_page3_data():
    profiles = pd.read_csv("data/kisip_mukuru_readiness_profiles.csv")
    shap_df = pd.read_csv("data/kisip_mukuru_shap_attribution.csv")
    mukuru_z = load_mukuru_zones()
    kisip_z = load_kisip_zones()
    mukuru_preds = load_mukuru_predictions()
    return profiles, shap_df, mukuru_z, kisip_z, mukuru_preds


@st.cache_data
def load_model_comparison_data():
    model_comp = pd.read_csv("data/kisip_model_comparison.csv")
    cva_pca = pd.read_csv("data/kisip_cva_vs_pca_comparison.csv")
    shap_imp = pd.read_csv("data/kisip_shap_importance.csv")
    preds = load_kisip_predictions()
    return model_comp, cva_pca, shap_imp, preds


@st.cache_data
def load_home_metrics():
    kisip_gdf = load_kisip_zones()
    mukuru_gdf = load_mukuru_zones()
    model_comp = pd.read_csv("data/kisip_model_comparison.csv")
    return kisip_gdf, mukuru_gdf, model_comp
