import warnings
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st

warnings.filterwarnings("ignore")


class DiagnosticsAnalyzer:
    """
    Phase 1: Network Diagnostics Engine.
    Analyzes station health, net flow (tidal) dynamics, and hourly demand profiles.
    STRICTLY filters for Manhattan to ensure strategy focus.
    """

    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        # Artifact Paths
        self.mapping_file = self.processed_dir / "station_borough_mapping.csv"
        self.morning_file = self.processed_dir / "manhattan_morning_rush_flow.csv"
        self.evening_file = self.processed_dir / "manhattan_evening_rush_flow.csv"
        self.heartbeat_file = self.processed_dir / "manhattan_hourly_heartbeat.csv"

        self.YEARS_TO_PROCESS = ["2023", "2024", "2025"]
        self.MANHATTAN_FILTER_ENABLED = True
        self.df = None

    def _load_manhattan_filter(self) -> set | None:
        """Loads the set of Manhattan station IDs to filter the analysis."""
        if not self.mapping_file.exists():
            return None
        map_df = pd.read_csv(self.mapping_file)
        # Normalize IDs to string to ensure matching works
        return set(
            map_df[map_df["borough"] == "Manhattan"]["station_id"]
            .astype(str)
            .str.strip()
        )

    def _process_year_data(
        self,
        year: str,
        manhattan_ids: set,
        station_names: dict,
        station_coords: dict,
        days_map: dict,
        progress_callback=None,
    ) -> tuple[list, list]:
        """Scans raw data and extracts rush hour flow."""
        year_path = self.raw_dir / year
        if not year_path.exists():
            return [], []

        files = sorted(year_path.rglob("*.csv"))
        if not files:
            return [], []

        col_map = {
            "starttime": "started_at",
            "start station id": "start_station_id",
            "end station id": "end_station_id",
            "start station name": "start_station_name",
            "start station latitude": "start_lat",
            "start station longitude": "start_lng",
            "started_at": "started_at",
            "start_station_id": "start_station_id",
            "end_station_id": "end_station_id",
            "start_station_name": "start_station_name",
            "start_lat": "start_lat",
            "start_lng": "start_lng",
        }

        am_chunks, pm_chunks = [], []

        for i, file in enumerate(files):
            if progress_callback:
                progress_callback(f"Scanning {file.name}...")
            try:
                header = pd.read_csv(file, nrows=0)
                use_cols = [c for c in col_map if c in header.columns]
                df = pd.read_csv(file, usecols=use_cols, dtype=str).rename(
                    columns=col_map
                )

                df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
                df = df.dropna(subset=["started_at"])
                df = df[df["started_at"].dt.dayofweek < 5]

                if df.empty:
                    continue

                if "start_station_id" in df:
                    self._extract_metadata(df, station_names, station_coords)

                df["month_key"] = df["started_at"].dt.to_period("M").astype(str)
                df["temp_date"] = df["started_at"].dt.date
                batch_dates = df.groupby("month_key")["temp_date"].unique()

                for month, dates_array in batch_dates.items():
                    if month not in days_map:
                        days_map[month] = set()
                    days_map[month].update(dates_array)

                df.drop(columns=["temp_date"], inplace=True)

                h = df["started_at"].dt.hour
                m = df["started_at"].dt.minute

                mask_am = ((h == 6) & (m >= 30)) | h.isin([7, 8]) | ((h == 9) & (m <= 30))
                mask_pm = ((h == 16) & (m >= 30)) | h.isin([17, 18]) | ((h == 19) & (m <= 30))

                if mask_am.any():
                    am_chunks.append(
                        self._calculate_net_flow(df[mask_am], manhattan_ids)
                    )
                if mask_pm.any():
                    pm_chunks.append(
                        self._calculate_net_flow(df[mask_pm], manhattan_ids)
                    )

            except Exception:
                continue

        return am_chunks, pm_chunks

    def _extract_metadata(
        self, df: pd.DataFrame, station_names: dict, station_coords: dict
    ):
        """Helper to update station metadata."""
        has_name = "start_station_name" in df
        has_lat = "start_lat" in df
        has_lng = "start_lng" in df

        meta_cols = ["start_station_id"]
        if has_name: meta_cols.append("start_station_name")
        if has_lat: meta_cols.append("start_lat")
        if has_lng: meta_cols.append("start_lng")

        unique_stations = df[meta_cols].drop_duplicates("start_station_id")

        for row in unique_stations.itertuples(index=False):
            sid = getattr(row, "start_station_id", None)
            if not sid: continue

            if has_name:
                sname = getattr(row, "start_station_name", None)
                if sname: station_names[sid] = sname

            if sid not in station_coords and has_lat and has_lng:
                try:
                    lat = float(getattr(row, "start_lat"))
                    lng = float(getattr(row, "start_lng"))
                    if not pd.isna(lat) and not pd.isna(lng):
                        station_coords[sid] = (lat, lng)
                except (ValueError, TypeError):
                    pass

    def _calculate_net_flow(
        self, subset: pd.DataFrame, filter_ids: set
    ) -> pd.DataFrame | None:
        """Calculates flow. NOTE: We do NOT filter here to allow system-wide calculation if needed later."""
        if subset.empty: return None

        starts = subset.groupby(["month_key", "start_station_id"]).size().reset_index(name="starts")
        starts.rename(columns={"start_station_id": "station_id"}, inplace=True)

        returns = subset.groupby(["month_key", "end_station_id"]).size().reset_index(name="returns")
        returns.rename(columns={"end_station_id": "station_id"}, inplace=True)

        flow = pd.merge(starts, returns, on=["month_key", "station_id"], how="outer").fillna(0)
        
        # We apply the filter later during aggregation or loading
        return flow

    def _finalize_dataset(
        self,
        chunks: list,
        period_name: str,
        station_names: dict,
        station_coords: dict,
        days_map: dict,
    ):
        if not chunks: return

        df = pd.concat(chunks).groupby(["month_key", "station_id"]).sum().reset_index()

        def get_day_count(month):
            return len(days_map.get(month, [])) or 20

        df["days_observed"] = df["month_key"].apply(get_day_count)
        df["avg_daily_flow"] = round((df["returns"] - df["starts"]) / df["days_observed"], 2)
        df["station_name"] = df["station_id"].map(station_names)

        if station_coords:
            coords_df = pd.DataFrame.from_dict(station_coords, orient="index", columns=["lat", "lng"])
            coords_df.index.name = "station_id"
            df = df.merge(coords_df, on="station_id", how="left")

        output = self.processed_dir / f"manhattan_{period_name}_rush_flow.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)

    def _generate_heartbeat_metrics(self, manhattan_ids: set):
        """
        Generates hourly profiles. 
        FIX: Strictly filters for Manhattan IDs if provided.
        """
        if not self.morning_file.exists(): return

        m_df = pd.read_csv(self.morning_file)
        
        # --- FIX: Apply Manhattan Filter to Candidate Stations ---
        if manhattan_ids:
            # Ensure type match (string vs string)
            m_df = m_df[m_df["station_id"].astype(str).isin(manhattan_ids)]
            
        top_stations = set(m_df["station_id"].unique())

        year_path = self.raw_dir / "2023"
        files = sorted(year_path.rglob("*.csv"))[:5] if year_path.exists() else sorted(self.raw_dir.rglob("*.csv"))[:5]

        hourly_chunks = []
        for file in files:
            try:
                df = pd.read_csv(file, usecols=["started_at", "start_station_id", "end_station_id"], dtype=str)
                df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
                df = df.dropna(subset=["started_at"])
                df = df[df["started_at"].dt.dayofweek < 5]

                unique_days = max(1, df["started_at"].dt.date.nunique())
                df["hour"] = df["started_at"].dt.hour

                starts = df.groupby(["hour", "start_station_id"]).size().reset_index(name="starts")
                starts.rename(columns={"start_station_id": "station_id"}, inplace=True)

                returns = df.groupby(["hour", "end_station_id"]).size().reset_index(name="returns")
                returns.rename(columns={"end_station_id": "station_id"}, inplace=True)

                flow = pd.merge(starts, returns, on=["hour", "station_id"], how="outer").fillna(0)
                flow["starts"] /= unique_days
                flow["returns"] /= unique_days

                # Filter for only our top Manhattan stations
                filtered = flow[flow["station_id"].isin(top_stations)]
                if not filtered.empty:
                    hourly_chunks.append(filtered)

            except Exception:
                continue

        if hourly_chunks:
            full_df = pd.concat(hourly_chunks)
            final_stats = full_df.groupby(["hour", "station_id"]).mean(numeric_only=True).reset_index()
            final_stats["net"] = final_stats["returns"] - final_stats["starts"]

            am_net = final_stats[final_stats["hour"].between(6, 9)].groupby("station_id")["net"].sum()

            if not am_net.empty:
                sources = am_net.nsmallest(50).index
                sinks = am_net.nlargest(50).index

                source_trend = final_stats[final_stats["station_id"].isin(sources)].groupby("hour")["net"].mean().reset_index(name="net_flow")
                source_trend["type"] = "Residential (Source)"

                sink_trend = final_stats[final_stats["station_id"].isin(sinks)].groupby("hour")["net"].mean().reset_index(name="net_flow")
                sink_trend["type"] = "Commercial (Sink)"

                pd.concat([source_trend, sink_trend]).to_csv(self.heartbeat_file, index=False)

    def run_diagnostics_pipeline(self) -> str:
        """Main orchestrator."""
        manhattan_ids = self._load_manhattan_filter() if self.MANHATTAN_FILTER_ENABLED else None

        station_names = {}
        station_coords = {}
        days_map = {}
        all_am, all_pm = [], []

        with st.status("Running Diagnostics Pipeline...", expanded=True) as status:
            for year in self.YEARS_TO_PROCESS:
                st.write(f"📂 Processing {year}...")
                am, pm = self._process_year_data(year, manhattan_ids, station_names, station_coords, days_map)
                all_am.extend(am)
                all_pm.extend(pm)

            if not all_am and not all_pm:
                status.update(label="❌ Failed: No Data Matched", state="error")
                return "NO_DATA_MATCHED"

            st.write("💾 Finalizing Morning/Evening Datasets...")
            self._finalize_dataset(all_am, "morning", station_names, station_coords, days_map)
            self._finalize_dataset(all_pm, "evening", station_names, station_coords, days_map)

            st.write("💓 Generating Heartbeat Metrics...")
            self._generate_heartbeat_metrics(manhattan_ids)

            status.update(label="✅ Diagnostics Complete", state="complete")

        return "SUCCESS"

    def load_metrics(self) -> pd.DataFrame | None:
        """
        Loads metrics and STRICTLY filters for Manhattan using the mapping file.
        This ensures Metrics, Map, and Deterioration charts all see the same clean data.
        """
        if not self.morning_file.exists(): return None
        try:
            self.df = pd.read_csv(self.morning_file, dtype={"station_id": str})
            if self.df.empty: return None

            date_col = "month_key" if "month_key" in self.df.columns else "month"
            self.df["year"] = self.df[date_col].astype(str).str.split("-").str[0].astype(int)
            self.df["abs_flow"] = self.df["avg_daily_flow"].abs()

            # Fix Coordinates Names
            if "lon" in self.df.columns and "lng" not in self.df.columns:
                self.df.rename(columns={"lon": "lng"}, inplace=True)
            if "start_lat" in self.df.columns:
                self.df.rename(columns={"start_lat": "lat", "start_lng": "lng"}, inplace=True)

            # --- CRITICAL: GLOBAL MANHATTAN FILTER ---
            if self.MANHATTAN_FILTER_ENABLED and self.mapping_file.exists():
                map_df = pd.read_csv(self.mapping_file, dtype={"station_id": str})
                # Keep only stations that exist in the Manhattan mapping
                manhattan_stations = set(map_df[map_df["borough"] == "Manhattan"]["station_id"])
                self.df = self.df[self.df["station_id"].isin(manhattan_stations)]

            return self.df
        except Exception:
            return None

    def get_tidal_wave_map(self, target_year=2023) -> go.Figure | None:
        """Generates Mapbox scatter plot."""
        if self.df is None: self.load_metrics()
        if self.df is None: return None

        subset = self.df[self.df["year"] == target_year].copy()
        if subset.empty: return None

        # Coordinates Self-Healing (Last Resort)
        if "lat" not in subset.columns or "lng" not in subset.columns:
            try:
                raw_files = sorted(self.raw_dir.rglob("*.csv"))
                if raw_files:
                    coords = pd.read_csv(raw_files[0], usecols=["start_station_id", "start_lat", "start_lng"], dtype={"start_station_id": str}, nrows=50000)
                    coords = coords.dropna().drop_duplicates("start_station_id").rename(columns={"start_station_id": "station_id", "start_lat": "lat", "start_lng": "lng"})
                    subset = pd.merge(subset, coords, on="station_id", how="left")
            except Exception:
                pass

        if "lat" not in subset.columns or "lng" not in subset.columns:
            st.error("❌ Coordinates missing. Please re-run Phase 0 Flow Analysis.")
            return None

        df_map = subset.groupby(["station_id", "station_name"]).agg({"avg_daily_flow": "mean", "lat": "first", "lng": "first"}).reset_index()
        df_map = df_map.dropna(subset=["lat", "lng"])

        if df_map.empty: return None

        df_map["abs_flow"] = df_map["avg_daily_flow"].abs()
        df_map["text_label"] = df_map["station_name"] + "<br>Net: " + df_map["avg_daily_flow"].round(1).astype(str)

        fig = px.scatter_mapbox(
            df_map, lat="lat", lon="lng", color="avg_daily_flow", size="abs_flow",
            color_continuous_scale="RdBu", size_max=15, zoom=11,
            center={"lat": 40.75, "lon": -73.98}, # Centered on Manhattan
            hover_name="text_label", title=f"<b>Tidal Wave ({target_year}) - Manhattan Only</b>"
        )
        fig.update_layout(mapbox_style="carto-positron", height=700, margin={"r": 0, "t": 40, "l": 0, "b": 0})
        return fig

    def get_deterioration_analysis(self) -> tuple[go.Figure, pd.DataFrame] | tuple[None, None]:
        """Compares station performance (2023 vs 2025)."""
        if self.df is None: self.load_metrics()
        if self.df is None: return None, None

        pivot = self.df.groupby(["station_id", "station_name", "year"])["avg_daily_flow"].mean().unstack("year").reset_index()

        if 2023 not in pivot or 2025 not in pivot: return None, None

        pivot["delta"] = pivot[2025] - pivot[2023]
        data_2023 = pivot[2023].dropna().tolist()
        data_2025 = pivot[2025].dropna().tolist()

        fig = ff.create_distplot([data_2023, data_2025], ["2023", "2025"], show_hist=False, colors=["#1f77b4", "#d62728"])
        fig.update_layout(title="<b>Deterioration Analysis (2023 vs 2025)</b>", template="plotly_white", height=500)
        fig.add_vline(x=0, line_dash="dash", line_color="gray")

        cols = ["station_name", 2023, 2025, "delta"]
        top_outliers = pd.concat([
            pivot.nsmallest(5, "delta")[cols].assign(type="Starvation (Decreased Inflow)"),
            pivot.nlargest(5, "delta")[cols].assign(type="Blocking (Decreased Outflow)"),
        ])
        return fig, top_outliers

    def get_heartbeat_chart(self) -> go.Figure | None:
        """Visualizes hourly net flow."""
        if not self.heartbeat_file.exists(): return None
        try:
            df = pd.read_csv(self.heartbeat_file)
            if df.empty: return None

            fig = px.line(
                df, x="hour", y="net_flow", color="type",
                color_discrete_map={"Residential (Source)": "#d62728", "Commercial (Sink)": "#1f77b4"},
                markers=True
            )
            fig.add_vrect(x0=6, x1=9, fillcolor="gray", opacity=0.1, layer="below", annotation_text="Morning Rush")
            fig.add_vrect(x0=15.5, x1=18, fillcolor="gray", opacity=0.1, layer="below", annotation_text="Evening Rush")
            fig.update_layout(template="plotly_white", height=500, title="<b>Hourly Heartbeat</b>")
            return fig
        except Exception:
            return None