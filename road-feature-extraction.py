import os
os.chdir(r'C:\Users\USER\Documents\KISIP')

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------
# SECTION 1: LOAD ZONE GRID
# ----------------------------------------------------------------
print('Loading zone grid...')
zones = gpd.read_file(r'data\kisip_analysis_zones_50m.geojson')
zones = zones.to_crs(epsg=4326)
print(f'Zones loaded: {len(zones)}')

# ----------------------------------------------------------------
# SECTION 2: LOAD NAIROBI ROADS FROM LOCAL SHAPEFILE/GEOJSON
# Update filename to match what BBBike sent you
# It will be something like: roads.shp or roads.geojson
# or if from Geofabrik shapefiles: kenya-latest-free.shp/gis_osm_roads_free_1.shp
# ----------------------------------------------------------------
print('\nLoading local road network...')

# Try shapefile first
try:
    roads_all = gpd.read_file(r'planet\shape\roads.shp')
    print('Loaded roads.shp')
except:
    # Try geojson
    try:
        roads_all = gpd.read_file(r'planet\shape\roads.geojson')
        print('Loaded roads.geojson')
    except Exception as e:
        print(f'Could not load roads file: {e}')
        exit()

roads_all = roads_all.to_crs(epsg=4326)
print(f'Total road segments: {len(roads_all)}')
print(f'Columns: {roads_all.columns.tolist()}')
print(f'\nRoad type distribution:')
print(roads_all['type'].value_counts())


# Check surface tags
if 'surface' in roads_all.columns:
    print(f'\nSurface values:\n{roads_all["surface"].value_counts().head(10)}')

# ----------------------------------------------------------------
# SECTION 3: CLIP TO NAIROBI BBOX TO REDUCE SIZE
# ----------------------------------------------------------------
from shapely.geometry import box
nairobi_bbox = box(36.65, -1.45, 37.05, -1.10)
roads_all = roads_all[roads_all.intersects(nairobi_bbox)].copy()
print(f'Roads within Nairobi bbox: {len(roads_all)}')

# ----------------------------------------------------------------
# PAVED ROAD TYPES — derived from OSM highway type
# No surface column available in BBBike export
# Paved status inferred from road classification
# which correlates strongly with surface quality in Nairobi
# ----------------------------------------------------------------
PAVED_TYPES = [
    'motorway', 'motorway_link',
    'trunk', 'trunk_link',
    'primary', 'primary_link',
    'secondary', 'secondary_link',
    'tertiary', 'tertiary_link'
]

def compute_zone_road_features(zone_row, roads_gdf):
    zone_geom     = zone_row.geometry
    zone_area_m2  = zone_geom.area * (111000 ** 2)
    zone_area_km2 = zone_area_m2 / 1e6

    if roads_gdf is None or len(roads_gdf) == 0:
        return {
            'road_density': 0.0,
            'paved_proportion': 0.0,
            'total_road_length_m': 0.0,
            'road_segment_count': 0
        }

    try:
        candidates = roads_gdf[roads_gdf.intersects(zone_geom)].copy()

        if len(candidates) == 0:
            return {
                'road_density': 0.0,
                'paved_proportion': 0.0,
                'total_road_length_m': 0.0,
                'road_segment_count': 0
            }

        candidates['geometry'] = candidates.geometry.intersection(zone_geom)
        candidates = candidates[~candidates.geometry.is_empty].copy()

        if len(candidates) == 0:
            return {
                'road_density': 0.0,
                'paved_proportion': 0.0,
                'total_road_length_m': 0.0,
                'road_segment_count': 0
            }

        candidates['clipped_length_m'] = candidates.geometry.length * 111000
        total_length_m = candidates['clipped_length_m'].sum()
        road_density   = total_length_m / zone_area_km2 if zone_area_km2 > 0 else 0.0

        # Paved proportion from road type column
        if 'type' in candidates.columns:
            paved_length     = candidates.loc[
                candidates['type'].isin(PAVED_TYPES), 'clipped_length_m'
            ].sum()
            paved_proportion = (
                paved_length / total_length_m
                if total_length_m > 0 else 0.0
            )
        else:
            paved_proportion = 0.0

        return {
            'road_density': round(road_density, 4),
            'paved_proportion': round(paved_proportion, 4),
            'total_road_length_m': round(total_length_m, 2),
            'road_segment_count': len(candidates)
        }

    except Exception as e:
        return {
            'road_density': 0.0,
            'paved_proportion': 0.0,
            'total_road_length_m': 0.0,
            'road_segment_count': 0
        }
# ----------------------------------------------------------------
# SECTION 5: GET ROADS PER SETTLEMENT
# ----------------------------------------------------------------
def get_settlement_roads(settlement_name, settlement_geometry):
    print(f'  Filtering roads for {settlement_name}...')
    roads_in_settlement = roads_all[
        roads_all.intersects(settlement_geometry)
    ].copy()

    if len(roads_in_settlement) == 0:
        print(f'  {settlement_name}: No roads found')
        return None

    print(f'  {settlement_name}: {len(roads_in_settlement)} segments found')
    return roads_in_settlement

# ----------------------------------------------------------------
# SECTION 6: COMPUTE ROAD FEATURES PER ZONE
# ----------------------------------------------------------------
def compute_zone_road_features(zone_row, roads_gdf):
    zone_geom     = zone_row.geometry
    zone_area_m2  = zone_geom.area * (111000 ** 2)
    zone_area_km2 = zone_area_m2 / 1e6

    if roads_gdf is None or len(roads_gdf) == 0:
        return {
            'road_density': 0.0,
            'paved_proportion': 0.0,
            'total_road_length_m': 0.0,
            'road_segment_count': 0
        }

    try:
        candidates = roads_gdf[roads_gdf.intersects(zone_geom)].copy()

        if len(candidates) == 0:
            return {
                'road_density': 0.0,
                'paved_proportion': 0.0,
                'total_road_length_m': 0.0,
                'road_segment_count': 0
            }

        candidates['geometry'] = candidates.geometry.intersection(zone_geom)
        candidates = candidates[~candidates.geometry.is_empty].copy()

        if len(candidates) == 0:
            return {
                'road_density': 0.0,
                'paved_proportion': 0.0,
                'total_road_length_m': 0.0,
                'road_segment_count': 0
            }

        candidates['clipped_length_m'] = candidates.geometry.length * 111000
        total_length_m = candidates['clipped_length_m'].sum()
        road_density   = total_length_m / zone_area_km2 if zone_area_km2 > 0 else 0.0

        if 'surface' in candidates.columns:
            paved_length     = candidates.loc[
                candidates['surface'].isin(PAVED_TYPES), 'clipped_length_m'
            ].sum()
            paved_proportion = paved_length / total_length_m if total_length_m > 0 else 0.0
        else:
            paved_proportion = 0.0

        return {
            'road_density': round(road_density, 4),
            'paved_proportion': round(paved_proportion, 4),
            'total_road_length_m': round(total_length_m, 2),
            'road_segment_count': len(candidates)
        }

    except Exception as e:
        return {
            'road_density': 0.0,
            'paved_proportion': 0.0,
            'total_road_length_m': 0.0,
            'road_segment_count': 0
        }

# ----------------------------------------------------------------
# SECTION 7: MAIN PROCESSING LOOP
# ----------------------------------------------------------------
all_results      = []
settlements_list = zones['settlement'].unique()

print(f'\nProcessing {len(settlements_list)} settlements...')
print(f'Total zones: {len(zones)}\n')

for settlement_name in settlements_list:
    print(f'--- {settlement_name} ---')

    settlement_zones    = zones[zones['settlement'] == settlement_name].copy()
    settlement_boundary = unary_union(settlement_zones.geometry)
    settlement_roads    = get_settlement_roads(settlement_name, settlement_boundary)

    zone_count = len(settlement_zones)
    print(f'  Processing {zone_count} zones...')

    for i, (idx, zone_row) in enumerate(settlement_zones.iterrows()):
        if i % 100 == 0 and i > 0:
            print(f'  Progress: {i}/{zone_count}')

        features               = compute_zone_road_features(zone_row, settlement_roads)
        features['zone_id']    = zone_row['zone_id']
        features['settlement'] = settlement_name
        all_results.append(features)

    print(f'  Completed: {zone_count} zones\n')

# ----------------------------------------------------------------
# SECTION 8: ASSEMBLE AND EXPORT
# ----------------------------------------------------------------
road_features_df = pd.DataFrame(all_results)[[
    'zone_id', 'settlement', 'road_density',
    'paved_proportion', 'total_road_length_m', 'road_segment_count'
]]

print('=' * 50)
print('EXTRACTION COMPLETE')
print('=' * 50)
print(f'Total zones: {len(road_features_df)}')
print(f'Zones with roads:    {(road_features_df["road_density"] > 0).sum()}')
print(f'Zones without roads: {(road_features_df["road_density"] == 0).sum()}')

print('\nSummary by settlement:')
summary = road_features_df.groupby('settlement').agg(
    zones_total       = ('zone_id', 'count'),
    zones_with_roads  = ('road_density', lambda x: (x > 0).sum()),
    mean_road_density = ('road_density', 'mean'),
    mean_paved_prop   = ('paved_proportion', 'mean')
).reset_index()
print(summary.to_string(index=False))

road_features_df.to_csv('kisip_road_features.csv', index=False)
print('\nExported: kisip_road_features.csv')