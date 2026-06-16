import pandas as pd
import geopandas as gpd


summary = pd.read_csv("data/kisip_zone_scmi.csv ")
record_count = len(summary)
print(summary)
