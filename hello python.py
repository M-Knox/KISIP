import pandas as pd
import geopandas as gpd

gdf = gpd.read_file("data/mukuru_zones_spatial.geojson")
preds = pd.read_csv("data/kisip_mukuru_predictions.csv")

print("GDF zone_ids:", gdf["zone_id"].head(5).tolist())
print("Preds zone_ids:", preds["zone_id"].head(5).tolist())
print("Matching zone_ids:", gdf["zone_id"].isin(preds["zone_id"]).sum(), "out of", len(gdf))