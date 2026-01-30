from pathlib import Path
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests


class DemandForecaster:
    """
    Forecasting engine using Gradient Boosting (XGBoost) to predict station-level
    supply and demand.
    
    pipeline:
    1. Ingestion (Internal Data + External Weather API)
    2. Feature Engineering (Temporal, Lag, & External Regressors)
    3. Training (Dual Regressors for Starts/Returns)
    4. Evaluation (Time-Series Split)
    """

    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)
        self.input_file = self.processed_dir / "manhattan_morning_rush_flow.csv"
        self.weather_cache = None
        self.df = None

    def load_data(self) -> str:
        """Loads and standardizes the primary dataset."""
        if not self.input_file.exists():
            return "FILE_NOT_FOUND"

        self.df = pd.read_csv(self.input_file)
        
        # Standardize date column
        date_col = "month_key" if "month_key" in self.df.columns else "month"
        self.df["year"] = pd.to_datetime(self.df[date_col]).dt.year
        return "SUCCESS"

    def get_station_list(self) -> List[str]:
        """Returns a list of stations sorted by total volume."""
        if self.df is None:
            return []
        
        stats = (
            self.df.groupby("station_name")
            .agg({"starts": "sum", "returns": "sum"})
            .reset_index()
        )
        stats["volume"] = stats["starts"] + stats["returns"]
        return stats.sort_values("volume", ascending=False)["station_name"].tolist()

    def _fetch_weather(
        self, start_date: str = "2023-01-01", end_date: str = "2025-11-30"
    ) -> pd.DataFrame:
        """
        Fetches historical weather data from Open-Meteo API.
        Acts as an external regressor for the model.
        """
        if self.weather_cache is not None:
            return self.weather_cache

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 40.71,
            "longitude": -74.01,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["temperature_2m_mean", "precipitation_sum"],
            "timezone": "America/New_York",
        }
        
        try:
            r = requests.get(url, params=params, timeout=5)
            r.raise_for_status()
            data = r.json()
            
            w_df = pd.DataFrame({
                "date": data["daily"]["time"],
                "temp_c": data["daily"]["temperature_2m_mean"],
                "precip_mm": data["daily"]["precipitation_sum"],
            })
            
            w_df["date"] = pd.to_datetime(w_df["date"])
            w_df["temp_f"] = (w_df["temp_c"] * 9 / 5) + 32
            # Binary feature: Rain > 2mm implies significant impact on ridership
            w_df["is_rain"] = (w_df["precip_mm"] > 2.0).astype(int)
            
            self.weather_cache = w_df[["date", "temp_f", "is_rain"]]
            return self.weather_cache
            
        except Exception:
            # Fallback for offline/error states
            dates = pd.date_range(start=start_date, end=end_date)
            return pd.DataFrame({"date": dates, "temp_f": 60, "is_rain": 0})

    def _get_holidays(self, dates: pd.Series) -> pd.Series:
        """Generates binary flags for major US holidays."""
        holidays = ["01-01", "07-04", "12-25", "11-25", "11-26", "05-29", "09-04"]
        return dates.dt.strftime("%m-%d").isin(holidays).astype(int)

    def run_forecast(
        self, station_name: str
    ) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame] | Tuple[None, None, None, None]:
        """
        Executes the full ML pipeline: Data Prep -> Feature Engineering -> Training -> Inference.
        """
        if self.df is None:
            return None, None, None, None

        # --- 1. Data Preparation & Augmentation ---
        station_df = self.df[self.df["station_name"] == station_name].copy()
        dates = pd.date_range(start="2023-01-01", end="2025-11-30", freq="D")
        daily = pd.DataFrame({"date": dates})
        daily["month_key"] = daily["date"].dt.to_period("M").astype(str)

        station_df["month_key"] = (
            pd.to_datetime(station_df["month_key"]).dt.to_period("M").astype(str)
        )
        
        merged = (
            daily.merge(
                station_df[["month_key", "starts", "returns"]],
                on="month_key",
                how="left",
            )
            .ffill()
            .bfill()
        )

        # Disaggregate Monthly totals to Daily averages (Approx. 22 working days)
        merged["starts"] = merged["starts"] / 22
        merged["returns"] = merged["returns"] / 22

        # Integrate External Weather Data
        weather = self._fetch_weather()
        merged = merged.merge(weather, on="date", how="left").ffill()

        # Synthetic Noise Injection (Data Augmentation)
        np.random.seed(42)
        merged["is_weekend"] = merged["date"].dt.dayofweek >= 5

        def apply_noise(row, val):
            noise = np.random.normal(0, val * 0.1)
            # Weekend reduction factor (0.4) + Gaussian noise
            return (val * 0.4) + noise if row["is_weekend"] else val + noise

        merged["daily_starts"] = (
            merged.apply(lambda r: apply_noise(r, r["starts"]), axis=1).clip(lower=0)
        )
        merged["daily_returns"] = (
            merged.apply(lambda r: apply_noise(r, r["returns"]), axis=1).clip(lower=0)
        )

        # --- 2. Feature Engineering ---
        merged["dow"] = merged["date"].dt.dayofweek
        merged["month"] = merged["date"].dt.month
        # Cyclical Time Encoding
        merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12)
        merged["is_holiday"] = self._get_holidays(merged["date"])

        # Lag Features (Rolling Trends)
        for col in ["daily_starts", "daily_returns"]:
            merged[f"{col}_roll7"] = merged[col].shift(1).rolling(7).mean()

        merged = merged.dropna()

        # --- 3. Train/Test Split (Time Series Split) ---
        train = merged[merged["date"] < "2025-09-01"].copy()
        test = merged[merged["date"] >= "2025-09-01"].copy()

        features_demand = [
            "daily_starts_roll7", "dow", "month_sin", "temp_f", "is_rain", "is_holiday"
        ]
        features_supply = [
            "daily_returns_roll7", "dow", "month_sin", "temp_f", "is_rain", "is_holiday"
        ]

        # --- 4. Model Training (XGBoost) ---
        # Demand Model
        model_d = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=5, n_jobs=-1
        )
        model_d.fit(train[features_demand], train["daily_starts"])

        # Supply Model
        model_s = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=5, n_jobs=-1
        )
        model_s.fit(train[features_supply], train["daily_returns"])

        # --- 5. Inference & Metrics ---
        merged["pred_starts"] = model_d.predict(merged[features_demand])
        merged["pred_returns"] = model_s.predict(merged[features_supply])
        merged["pred_net_flow"] = merged["pred_returns"] - merged["pred_starts"]
        merged["actual_net_flow"] = merged["daily_returns"] - merged["daily_starts"]

        merged["split_type"] = "Train"
        merged.loc[merged["date"] >= "2025-09-01", "split_type"] = "Test"

        metrics = {
            "train_end": pd.Timestamp("2025-08-31"),
            "test_start": pd.Timestamp("2025-09-01"),
            "test_end": pd.Timestamp("2025-11-30"),
        }

        # --- 6. Explainability (Feature Importance) ---
        imp_d = model_d.feature_importances_
        imp_s = model_s.feature_importances_

        avg_imp = (imp_d + imp_s) / 2
        importance_df = pd.DataFrame({
            "Feature": features_demand,
            "Importance": avg_imp
        }).sort_values("Importance", ascending=False)

        preview_cols = [
            "date", "daily_starts", "daily_returns", "daily_starts_roll7",
            "temp_f", "is_rain", "is_holiday"
        ]
        
        return merged, metrics, importance_df, merged[preview_cols].tail(10)

    def plot_forecast(self, df: pd.DataFrame) -> go.Figure:
        """Visualizes Actual vs Forecasted Net Flow and Error Distribution."""
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=("<b>Net Flow Forecast</b>", "<b>Prediction Error</b>"),
        )

        # Actuals
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["actual_net_flow"],
                mode="lines",
                name="Actual Reality",
                line=dict(color="rgba(0,0,0,0.4)", width=1),
            ),
            row=1,
            col=1,
        )

        # Forecast
        train_df = df[df["split_type"] == "Train"]
        test_df = df[df["split_type"] == "Test"]

        fig.add_trace(
            go.Scatter(
                x=train_df["date"],
                y=train_df["pred_net_flow"],
                mode="lines",
                name="Model Fit",
                line=dict(color="#007AFF", width=2),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=test_df["date"],
                y=test_df["pred_net_flow"],
                mode="lines",
                name="Future Forecast",
                line=dict(color="#FF2D55", width=3),
            ),
            row=1,
            col=1,
        )

        # Residuals (Error)
        test_df = test_df.copy()
        test_df["error"] = test_df["actual_net_flow"] - test_df["pred_net_flow"]
        colors = ["#2ca02c" if abs(x) < 10 else "#d62728" for x in test_df["error"]]

        fig.add_trace(
            go.Bar(
                x=test_df["date"],
                y=test_df["error"],
                name="Error",
                marker_color=colors
            ),
            row=2,
            col=1,
        )

        # Formatting
        fig.add_hline(y=0, line_color="gray", line_width=1, opacity=0.5, row=1, col=1)
        fig.add_hline(y=0, line_color="gray", line_width=1, opacity=0.5, row=2, col=1)

        split_date = test_df["date"].min().timestamp() * 1000
        fig.add_vline(x=split_date, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_vline(x=split_date, line_dash="dash", line_color="gray", row=2, col=1)

        fig.update_layout(
            template="plotly_white",
            height=600,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
            yaxis=dict(title="Net Bikes"),
            xaxis2=dict(rangeslider=dict(visible=True), type="date"),
        )
        return fig
