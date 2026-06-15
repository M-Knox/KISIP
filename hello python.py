import geopandas as gpd;
import pandas as pd;

head = pd.read_csv("data/kisip_zone_scmi_both.csv");
print(head.columns.tolist())
print(head.head(5))