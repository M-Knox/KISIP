"""Folium map helpers."""

from __future__ import annotations

import folium
import geopandas as gpd
import pandas as pd

from utils.constants import ACCENT, ACCENT_MUKURU


def _lerp_color(t: float, low: tuple, high: tuple) -> str:
    t = max(0.0, min(1.0, t))
    r = int(low[0] + (high[0] - low[0]) * t)
    g = int(low[1] + (high[1] - low[1]) * t)
    b = int(low[2] + (high[2] - low[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def scmi_color(val: float, vmin: float, vmax: float) -> str:
    rng = vmax - vmin if vmax != vmin else 1.0
    t = (val - vmin) / rng
    return _lerp_color(t, (79, 195, 161), (245, 166, 35))


def direction_color(deg: float) -> str:
    """Map CVA direction (degrees) to hue on colour wheel."""
    import colorsys

    h = (deg % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.85)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def base_map(center: tuple[float, float], zoom: int = 14) -> folium.Map:
    return folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )


def add_zone_layer(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    value_col: str,
    layer_name: str,
    color_fn,
    tooltip_fn,
) -> None:
    layer = folium.FeatureGroup(name=layer_name)
    for _, row in gdf.iterrows():
        val = row[value_col] if pd.notna(row.get(value_col)) else 0
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, c=color_fn(val): {
                "fillColor": c,
                "color": "#0F1117",
                "weight": 0.5,
                "fillOpacity": 0.78,
            },
            tooltip=folium.Tooltip(tooltip_fn(row, val), sticky=False),
        ).add_to(layer)
    layer.add_to(m)


def add_study_area_layers(m: folium.Map, kisip_plot, mukuru_plot) -> None:
    kisip_layer = folium.FeatureGroup(name="KISIP treated")
    for _, row in kisip_plot.iterrows():
        scmi_val = row["ensemble_scmi"] if pd.notna(row["ensemble_scmi"]) else 0
        opacity = 0.4 + 0.5 * min(scmi_val / 0.4, 1.0)
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, op=opacity: {
                "fillColor": ACCENT,
                "color": "#2D8A74",
                "weight": 0.6,
                "fillOpacity": op,
            },
            tooltip=folium.Tooltip(
                f"<b>{row['settlement']}</b><br>Zone {row['zone_id']}<br>"
                f"SCMI {scmi_val:.4f}",
                sticky=False,
            ),
        ).add_to(kisip_layer)
    kisip_layer.add_to(m)

    mukuru_layer = folium.FeatureGroup(name="Mukuru predicted")
    for _, row in mukuru_plot.iterrows():
        scmi_val = row["ensemble_scmi"] if pd.notna(row["ensemble_scmi"]) else 0
        opacity = 0.35 + 0.55 * min(scmi_val / 0.15, 1.0)
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, op=opacity: {
                "fillColor": ACCENT_MUKURU,
                "color": "#C47D0E",
                "weight": 0.6,
                "fillOpacity": op,
            },
            tooltip=folium.Tooltip(
                f"<b>{row['settlement']}</b><br>Zone {row['zone_id']}<br>"
                f"Predicted {scmi_val:.4f}",
                sticky=False,
            ),
        ).add_to(mukuru_layer)
    mukuru_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
