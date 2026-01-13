from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class LogisticsOptimizer:
    """
    Calculates overnight inventory rebalancing requirements based on net station flow.
    Generates manifests for operational logistics to reset system state before morning rush.
    """

    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)
        self.input_file = self.processed_dir / "manhattan_morning_rush_flow.csv"
        self.df = None
        self.manifest = None

    def load_data(self) -> str:
        """
        Loads flow data and filters for the target operational year (default: 2025).
        """
        if not self.input_file.exists():
            return "FILE_NOT_FOUND"

        df = pd.read_csv(self.input_file)

        # Normalize date column handling
        date_col = "month_key" if "month_key" in df.columns else "month"
        df["year"] = pd.to_datetime(df[date_col]).dt.year

        # Target 2025 scenario, fallback to latest available if missing
        target_year = 2025
        if target_year not in df["year"].unique():
            target_year = df["year"].max()

        self.df = (
            df[df["year"] == target_year]
            .groupby(["station_id", "station_name"])["avg_daily_flow"]
            .mean()
            .reset_index()
        )
        self.df.rename(columns={"avg_daily_flow": "net_flow"}, inplace=True)

        return "SUCCESS"

    def generate_night_manifest(
        self, safety_buffer: float = 1.0, min_batch_size: int = 5
    ) -> pd.DataFrame | None:
        """
        Calculates the required Pick-up and Drop-off actions to neutralize station imbalances.
        
        """
        if self.df is None:
            return None

        manifest = self.df.copy()

        # Invert flow: Negative flow (Source) requires positive inventory injection (Drop Off)
        manifest["required_move"] = manifest["net_flow"] * -1

        # Apply Safety Buffer (Multiplier) specifically to Drop Offs to prevent morning stock-outs
        manifest["adjusted_qty"] = manifest.apply(
            lambda x: x["required_move"] * safety_buffer
            if x["required_move"] > 0
            else x["required_move"],
            axis=1,
        )

        manifest["quantity"] = manifest["adjusted_qty"].round(0).astype(int)

        manifest["action"] = manifest["quantity"].apply(
            lambda x: "DROP_OFF"
            if x > 0
            else ("PICK_UP" if x < 0 else "NO_ACTION")
        )

        manifest["quantity"] = manifest["quantity"].abs()
        
        # Filter out micro-moves that are operationally inefficient
        manifest = manifest[manifest["quantity"] >= min_batch_size]

        self.manifest = manifest.sort_values("quantity", ascending=False)
        return self.manifest
    
    def get_imbalance_distribution(self) -> go.Figure | None:
        """
        Visualizes the Histogram of Net Flows to identify system outliers.
        Highlights 'Starvation' (empty docks) vs 'Blocking' (full docks) risks.
        
        """
        if self.df is None:
            return None

        fig = px.histogram(
            self.df,
            x="net_flow",
            nbins=60,
            title="<b>Network Imbalance Distribution (Morning Rush)</b><br><sup>Negative = Starvation Risk | Positive = Blocking Risk</sup>",
            color_discrete_sequence=["#5c5c5c"],
            labels={"net_flow": "Net Bike Flow (06:00 - 10:00)"},
        )

        # Highlight the optimal operational zone
        fig.add_vrect(
            x0=-5,
            x1=5,
            fillcolor="green",
            opacity=0.1,
            annotation_text="Self-Balancing Zone",
            annotation_position="top",
        )

        # Annotate operational extremes
        fig.add_annotation(
            x=-40,
            y=10,
            text="<b>Critical Sources<br>(Need Drops)</b>",
            showarrow=False,
            font=dict(color="red"),
        )
        fig.add_annotation(
            x=40,
            y=10,
            text="<b>Critical Sinks<br>(Need Pickups)</b>",
            showarrow=False,
            font=dict(color="blue"),
        )

        fig.update_layout(template="plotly_white", height=500, bargap=0.1)
        return fig

    def get_distribution_chart(self) -> go.Figure | None:
        """
        Generates a bar chart summarizing total logistical volume by action type.
        """
        if self.manifest is None:
            return None

        summary = self.manifest.groupby("action")["quantity"].sum().reset_index()

        fig = px.bar(
            summary,
            x="action",
            y="quantity",
            color="action",
            title="<b>Overnight Logistics Volume (Bikes Moved)</b>",
            color_discrete_map={"DROP_OFF": "#d62728", "PICK_UP": "#1f77b4"},
            text="quantity",
        )
        fig.update_layout(template="plotly_white", height=400)
        return fig