import pandas as pd
import geopandas as gpd

print("=== mukuru_zones_spatial.geojson ===")
gdf = gpd.read_file("data/mukuru_zones_spatial.geojson")
print(gdf.columns.tolist())

print("\n=== kisip_zones_spatial.geojson ===")
gdf2 = gpd.read_file("data/kisip_zones_spatial.geojson")
print(gdf2.columns.tolist())

print("\n=== kisip_mukuru_predictions.csv ===")
print(pd.read_csv("data/kisip_mukuru_predictions.csv").columns.tolist())

print("\n=== kisip_model_predictions.csv ===")
print(pd.read_csv("data/kisip_model_predictions.csv").columns.tolist())

print("\n=== kisip_shap_by_settlement.csv ===")
print(pd.read_csv("data/kisip_shap_by_settlement.csv").columns.tolist())

print("\n=== kisip_mukuru_shap_attribution.csv ===")
print(pd.read_csv("data/kisip_mukuru_shap_attribution.csv").columns.tolist())

print("\n=== kisip_shap_importance.csv ===")
print(pd.read_csv("data/kisip_shap_importance.csv").columns.tolist())

print("\n=== Zone ID match (mukuru) ===")
mukuru_z = gpd.read_file("data/mukuru_zones_spatial.geojson")
preds = pd.read_csv("data/kisip_mukuru_predictions.csv")
for s in preds["settlement"].unique():
    gdf_ids  = set(mukuru_z[mukuru_z["settlement"] == s]["zone_id"])
    pred_ids = set(preds[preds["settlement"] == s]["zone_id"])
    print(f"{s}: GDF={len(gdf_ids)}, Preds={len(pred_ids)}, Match={len(gdf_ids & pred_ids)}")