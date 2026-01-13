import calendar
from pathlib import Path
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import plotly.express as px


class DataPipeline:
    """
    Phase 0 Pipeline: Ingests, Audits, and Cleans Bike Data.
    Enforces ULTRA-STRICT data quality rules.
    """

    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Artifact Paths
        self.agg_data_path = self.processed_dir / "final_aggregated_rides.csv"
        self.flow_data_path = self.processed_dir / "manhattan_morning_rush_flow.csv"
        self.mapping_output = self.processed_dir / "station_borough_mapping.csv"
        self.borough_stats_output = self.processed_dir / "final_aggregated_borough_data.csv"
        self.geojson_path = self.raw_dir / "new-york-city-boroughs.geojson"

        # Standardization Schema
        self.column_map = {
            "started_at": ["started_at", "start_time", "trip_start"],
            "member_casual": ["member_casual", "user_type"],
            "rideable_type": ["rideable_type", "bike_type", "ride_type"],
            "start_station_id": ["start_station_id", "start station id"],
            "end_station_id": ["end_station_id", "end station id"],
            "start_station_name": ["start_station_name", "start station name"],
            "lat": ["start_lat", "start station latitude"],
            "lon": ["start_lng", "start station longitude"],
        }

    def _smart_load(self, file_path: Path, required_cols: list = None) -> pd.DataFrame | None:
        """Attempts to load a CSV. If required_cols is None, loads ALL columns."""
        try:
            header = pd.read_csv(file_path, nrows=0)
            use_cols = {}
            rename_map = {}

            target_cols = required_cols if required_cols else self.column_map.keys()

            for std_col in target_cols:
                variants = self.column_map.get(std_col, [std_col])
                for variant in variants:
                    if variant in header.columns:
                        use_cols[variant] = std_col
                        rename_map[variant] = std_col
                        break
            
            if required_cols and not use_cols: return None

            if required_cols:
                df = pd.read_csv(file_path, usecols=use_cols.keys(), low_memory=False)
            else:
                df = pd.read_csv(file_path, low_memory=False)
            
            df.rename(columns=rename_map, inplace=True)
            return df
        except Exception:
            return None

    # --- 1. HEALTH SCANNER (Exact Drop Calculation) ---
    def scan_data_health(self) -> pd.DataFrame:
        """Scans raw files to count EXACTLY how many rows will be dropped."""
        all_files = sorted(self.raw_dir.rglob("*.csv"))
        summary = []
        
        progress_bar = st.progress(0)
        for i, file in enumerate(all_files):
            # Load all columns to define "Dirty"
            df = self._smart_load(file, required_cols=None)
            
            if df is not None:
                total_rows = len(df)
                # Count rows that have ANY missing value in ANY column
                dirty_rows = df.isna().any(axis=1).sum()
                
                stats = {
                    "File Name": file.name,
                    "Total Rows": total_rows,
                    "Dirty Rows": dirty_rows, # Rows to drop
                    "Clean Rows": total_rows - dirty_rows,
                    "% Loss": round((dirty_rows / total_rows) * 100, 2) if total_rows > 0 else 0
                }
                # Also count missing by column for the graph
                for col in df.columns:
                    stats[f"Missing {col}"] = df[col].isna().sum()
                
                summary.append(stats)
            progress_bar.progress((i + 1) / len(all_files))
            
        return pd.DataFrame(summary)

    # --- 2. PIPELINES (Strict Dropping) ---
    def run_aggregation_pipeline(self) -> dict:
        """Aggregates data. DROPS ANY ROW with ANY missing value."""
        all_files = sorted(self.raw_dir.rglob("*.csv"))
        if not all_files: return {"status": "NO_RAW_DATA"}

        chunks = []
        stats = {"raw_rows": 0, "clean_rows": 0}
        progress_bar = st.progress(0)

        for i, file_path in enumerate(all_files):
            df = self._smart_load(file_path, ["started_at", "member_casual", "rideable_type"])
            if df is not None and not df.empty:
                rows_before = len(df)
                stats["raw_rows"] += rows_before
                
                # ULTRA-STRICT DROP
                df.dropna(inplace=True)
                df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
                df.dropna(subset=["started_at"], inplace=True) # Ensure date is valid too
                
                rows_after = len(df)
                stats["clean_rows"] += rows_after

                if not df.empty:
                    df["date"] = df["started_at"].dt.date
                    df["hour"] = df["started_at"].dt.hour
                    grouped = df.groupby(["date", "hour", "member_casual", "rideable_type"]).size().reset_index(name="count")
                    chunks.append(grouped)
            progress_bar.progress((i + 1) / len(all_files))

        if chunks:
            final_df = pd.concat(chunks, ignore_index=True)
            final_df = final_df.groupby(["date", "hour", "member_casual", "rideable_type"])["count"].sum().reset_index()
            final_df.to_csv(self.agg_data_path, index=False)
            return {"status": "SUCCESS", "metrics": stats}
        return {"status": "EMPTY"}

    def run_flow_pipeline(self) -> dict:
        """Calculates flow. DROPS ANY ROW with ANY missing value."""
        all_files = sorted(self.raw_dir.rglob("*.csv"))
        metrics_chunks = []
        stats = {"raw_rows": 0, "clean_rows": 0}
        progress_bar = st.progress(0)
        
        cols = ["started_at", "start_station_id", "end_station_id", "start_station_name", "lat", "lon"]

        for i, file_path in enumerate(all_files):
            df = self._smart_load(file_path, cols)
            if df is not None and not df.empty:
                rows_before = len(df)
                stats["raw_rows"] += rows_before

                # ULTRA-STRICT DROP
                df.dropna(inplace=True)
                df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
                df.dropna(subset=["started_at"], inplace=True)
                
                rows_after = len(df)
                stats["clean_rows"] += rows_after
                
                df["start_station_id"] = df["start_station_id"].astype(str)
                df["end_station_id"] = df["end_station_id"].astype(str)

                if df.empty: continue

                df["hour"] = df["started_at"].dt.hour
                df["day"] = df["started_at"].dt.dayofweek
                rush_mask = (df["day"] < 5) & (df["hour"].between(6, 9))
                rush_df = df[rush_mask].copy()

                if not rush_df.empty:
                    rush_df["month_key"] = rush_df["started_at"].dt.to_period("M").astype(str)
                    starts = rush_df.groupby(["month_key", "start_station_id", "start_station_name"]).size().reset_index(name="starts")
                    starts.rename(columns={"start_station_id": "station_id", "start_station_name": "station_name"}, inplace=True)
                    returns = rush_df.groupby(["month_key", "end_station_id"]).size().reset_index(name="returns")
                    returns.rename(columns={"end_station_id": "station_id"}, inplace=True)
                    
                    daily = pd.merge(starts, returns, on=["month_key", "station_id"], how="outer").fillna(0)
                    daily["avg_daily_flow"] = (daily["returns"] - daily["starts"]) / 20.0
                    
                    if "lat" in rush_df.columns:
                        coords = rush_df.groupby("start_station_id")[["lat", "lon"]].first().reset_index()
                        coords.rename(columns={"start_station_id": "station_id"}, inplace=True)
                        coords["station_id"] = coords["station_id"].astype(str)
                        daily = pd.merge(daily, coords, on="station_id", how="left")
                    
                    metrics_chunks.append(daily)
            progress_bar.progress((i + 1) / len(all_files))

        if metrics_chunks:
            final_flow = pd.concat(metrics_chunks)
            final_flow.to_csv(self.flow_data_path, index=False)
            return {"status": "SUCCESS", "metrics": stats}
        return {"status": "EMPTY"}
    
    def run_geographic_pipeline(self) -> dict:
        """Maps stations. DROPS any station with missing info."""
        if not self.geojson_path.exists(): return {"status": "MISSING_GEOJSON"}
        all_files = sorted(self.raw_dir.rglob("*.csv"))
        station_chunks = []
        trend_chunks = []
        stats = {"total_stations_found": 0, "mapped_stations": 0}
        progress_bar = st.progress(0)

        for i, file in enumerate(all_files):
            try:
                df_geo = self._smart_load(file, ["start_station_id", "lat", "lon"])
                if df_geo is not None:
                    df_geo.columns = ["station_id", "lat", "lng"]
                    unique_stations = df_geo.dropna().drop_duplicates("station_id")
                    unique_stations["station_id"] = unique_stations["station_id"].astype(str)
                    station_chunks.append(unique_stations)

                df_trends = self._smart_load(file, ["started_at", "start_station_id"])
                if df_trends is not None:
                    df_trends["started_at"] = pd.to_datetime(df_trends["started_at"], errors="coerce")
                    df_trends.dropna(inplace=True)
                    df_trends["start_station_id"] = df_trends["start_station_id"].astype(str)
                    monthly = df_trends.groupby([pd.Grouper(key="started_at", freq="M"), "start_station_id"]).size().reset_index(name="count")
                    trend_chunks.append(monthly)
            except Exception:
                continue
            progress_bar.progress((i + 1) / len(all_files))

        if not station_chunks: return {"status": "EMPTY"}

        all_stations = pd.concat(station_chunks).drop_duplicates("station_id")
        stats["total_stations_found"] = len(all_stations)
        
        nyc_boroughs = gpd.read_file(self.geojson_path)
        gdf_stations = gpd.GeoDataFrame(all_stations, geometry=gpd.points_from_xy(all_stations.lng, all_stations.lat), crs="EPSG:4326")

        if nyc_boroughs.crs != gdf_stations.crs:
            nyc_boroughs = nyc_boroughs.to_crs(gdf_stations.crs)

        joined = gdf_stations.sjoin(nyc_boroughs, how="left", predicate="within")
        mapping = joined[["station_id", "name", "lat", "lng"]].rename(columns={"name": "borough"})
        mapping["borough"] = mapping["borough"].fillna("Unknown")
        mapping.to_csv(self.mapping_output, index=False)
        
        stats["mapped_stations"] = len(mapping[mapping["borough"] != "Unknown"])

        station_map = mapping.set_index("station_id")["borough"].to_dict()
        all_trends = pd.concat(trend_chunks)
        all_trends["borough"] = all_trends["start_station_id"].map(station_map).fillna("Unknown")
        all_trends = all_trends[all_trends["borough"] != "Unknown"]

        final_trends = all_trends.groupby(["started_at", "borough"])["count"].sum().reset_index()
        final_trends.to_csv(self.borough_stats_output, index=False)
        return {"status": "SUCCESS", "metrics": stats}

    # --- 3. DASHBOARDS (UNCHANGED) ---
    def get_diagnostics_dashboard(self) -> go.Figure | None:
        input_file = self.agg_data_path
        if not input_file.exists(): return None
        df = pd.read_csv(input_file)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].dt.year >= 2023]
        if df.empty: return None

        daily_total = df.groupby(["date", "member_casual"])["count"].sum().reset_index()
        daily_total.sort_values("date", inplace=True)
        daily_total["rolling_avg"] = daily_total.groupby("member_casual")["count"].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        hourly_profile = df.groupby(["hour", "member_casual"])["count"].mean().reset_index()

        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15, subplot_titles=("Daily Ridership (7-Day Rolling Avg)", "Average Hourly Demand"))
        colors = {"member": "#005aab", "casual": "#7f7f7f"}

        for user_type in ["member", "casual"]:
            subset = daily_total[daily_total["member_casual"] == user_type]
            fig.add_trace(go.Scatter(x=subset["date"], y=subset["rolling_avg"], name=user_type.title(), mode="lines", line=dict(color=colors[user_type], width=2), legendgroup=user_type), row=1, col=1)

        for user_type in ["member", "casual"]:
            subset = hourly_profile[hourly_profile["member_casual"] == user_type]
            fig.add_trace(go.Scatter(x=subset["hour"], y=subset["count"], name=user_type.title(), mode="lines+markers", line=dict(color=colors[user_type], width=2, dash="solid" if user_type == "member" else "dot"), legendgroup=user_type, showlegend=False), row=2, col=1)

        for x0, x1 in [(7, 9), (16, 18)]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor="gray", opacity=0.1, layer="below", line_width=0, row=2, col=1)

        fig.update_layout(template="plotly_white", height=500, hovermode="x unified", legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"), margin=dict(t=60, b=40, l=40, r=40))
        return fig

    def get_comprehensive_dashboard(self) -> go.Figure | None:
        if not self.agg_data_path.exists(): return None
        df = pd.read_csv(self.agg_data_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].dt.year > 2022].copy()
        if df.empty: return None

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["month_name"] = df["date"].dt.strftime("%b")
        df["month_dt"] = df["date"].dt.to_period("M").dt.start_time

        monthly_user = df.groupby(["month_dt", "member_casual"])["count"].sum().reset_index()
        monthly_total = df.groupby(["year", "month", "month_name"])["count"].sum().reset_index()
        monthly_bike = df.groupby(["month_dt", "rideable_type"])["count"].sum().reset_index() if "rideable_type" in df.columns else pd.DataFrame(columns=["month_dt", "rideable_type", "count"])

        fig = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{"secondary_y": False}, {"secondary_y": False}]], subplot_titles=("Monthly User Composition", "Year-over-Year Monthly Totals", "Monthly Fleet Mix"), vertical_spacing=0.15, horizontal_spacing=0.1)

        colors_user = {"member": "#005aab", "casual": "#7f7f7f"}
        for user_type in ["member", "casual"]:
            subset = monthly_user[monthly_user["member_casual"] == user_type]
            fig.add_trace(go.Bar(x=subset["month_dt"], y=subset["count"], name=user_type.title(), marker_color=colors_user.get(user_type)), row=1, col=1)

        colors_year = {2023: "#a6cee3", 2024: "#1f78b4", 2025: "#b2df8a"}
        for year in sorted(monthly_total["year"].unique()):
            subset = monthly_total[monthly_total["year"] == year]
            fig.add_trace(go.Scatter(x=subset["month_name"], y=subset["count"], name=str(year), mode="lines+markers", line=dict(width=3, color=colors_year.get(year, "gray")), marker=dict(size=6)), row=2, col=1)

        colors_bike = {"classic_bike": "#95a5a6", "electric_bike": "#f1c40f"}
        if not monthly_bike.empty:
            for bike_type in colors_bike:
                subset = monthly_bike[monthly_bike["rideable_type"] == bike_type]
                if not subset.empty:
                    fig.add_trace(go.Scatter(x=subset["month_dt"], y=subset["count"], name=bike_type.replace("_", " ").title(), stackgroup="one", line=dict(width=0), fillcolor=colors_bike.get(bike_type)), row=2, col=2)

        fig.update_layout(template="plotly_white", height=800, barmode="stack", hovermode="x unified", legend=dict(orientation="v", y=1.0, x=1.02, xanchor="left"), margin=dict(t=60, r=120))
        fig.update_xaxes(tickformat="%b %Y", dtick="M3", row=1, col=1)
        return fig

    def get_geographic_dashboard(self) -> go.Figure | None:
        if not (self.mapping_output.exists() and self.borough_stats_output.exists()): return None
        stations = pd.read_csv(self.mapping_output)
        trends = pd.read_csv(self.borough_stats_output)
        trends["started_at"] = pd.to_datetime(trends["started_at"])
        colors = {"Manhattan": "#1f77b4", "Brooklyn": "#ff7f0e", "Queens": "#2ca02c", "Bronx": "#d62728"}

        fig = make_subplots(rows=4, cols=2, specs=[[{"type": "mapbox", "rowspan": 4}, {"type": "xy"}], [None, {"type": "xy"}], [None, {"type": "xy"}], [None, {"type": "xy"}]], column_widths=[0.6, 0.4], horizontal_spacing=0.05, vertical_spacing=0.08, subplot_titles=("<b>Infrastructure Density</b>", "Manhattan Volume", "Brooklyn Volume", "Queens Volume", "Bronx Volume"))

        for borough, color in colors.items():
            subset = stations[stations["borough"] == borough]
            fig.add_trace(go.Scattermapbox(lat=subset["lat"], lon=subset["lng"], mode="markers", marker=go.scattermapbox.Marker(size=5, color=color, opacity=0.8), name=borough, legendgroup="map", text=subset["station_id"], hoverinfo="text+name"), row=1, col=1)

        boroughs_ordered = ["Manhattan", "Brooklyn", "Queens", "Bronx"]
        for i, borough in enumerate(boroughs_ordered):
            subset = trends[trends["borough"] == borough].sort_values("started_at")
            fig.add_trace(go.Scatter(x=subset["started_at"], y=subset["count"], mode="lines+markers", line=dict(color=colors.get(borough, "gray"), width=2), name=borough, showlegend=False), row=i + 1, col=2)

        fig.update_layout(template="plotly_white", height=800, mapbox=dict(style="carto-positron", center=dict(lat=40.73, lon=-73.95), zoom=10), margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.02, x=0.3))
        return fig