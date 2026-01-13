import pandas as pd
import plotly.express as px
from pathlib import Path

class SystemStressAnalyzer:
    """
    Phase 3: Strategic Investment (Infrastructure & Staffing)
    Objective: Categorize stations into CapEx (Build Docks) vs OpEx (Hire Valets).
    """
    def __init__(self, flow_path, rides_path=None):
        self.flow_path = Path(flow_path)
        self.rides_path = Path(rides_path) if rides_path else None 
        self.metrics = None
        # Initialize variables to store the calculated thresholds
        self.vol_threshold = 0 
        self.flow_threshold = 0

    def load_metrics(self, target_year=2025):
        if not self.flow_path.exists():
            return None

        # 1. Load Data
        raw_df = pd.read_csv(self.flow_path)
        
        # Smart Date Detection
        date_col = 'month_key' if 'month_key' in raw_df.columns else 'month'
        raw_df['year'] = pd.to_datetime(raw_df[date_col]).dt.year
        df_year = raw_df[raw_df['year'] == target_year].copy()

        if df_year.empty:
            return None

        # 2. Aggregation
        self.metrics = df_year.groupby(['station_id', 'station_name']).agg(
            avg_net_flow=('avg_daily_flow', 'mean'),
            total_starts=('starts', 'sum'),
            total_returns=('returns', 'sum')
        ).reset_index()

        self.metrics['total_volume'] = self.metrics['total_starts'] + self.metrics['total_returns']

        # 3. Dynamic Thresholds (CALCULATING THE STD DEV HERE)
        # Volume: Top 5%
        self.vol_threshold = self.metrics['total_volume'].quantile(0.95)
        
        # Flow: 1 Standard Deviation from the mean
        self.flow_threshold = self.metrics['avg_net_flow'].std()
        
        # 4. Categorize using the dynamic numbers
        def categorize(row):
            if row['total_volume'] > self.vol_threshold:
                return "VALET_SERVICE"
            elif row['avg_net_flow'] > self.flow_threshold:
                return "EXPAND_DOCKS"
            elif row['avg_net_flow'] < -self.flow_threshold:
                return "PRIORITY_RESTOCK"
            else:
                return "STANDARD_OPS"

        self.metrics['category'] = self.metrics.apply(categorize, axis=1)
        
        return self.metrics

    def get_quadrants_chart(self):
        if self.metrics is None: return None

        fig = px.scatter(
            self.metrics,
            x="avg_net_flow",
            y="total_volume",
            color="category",
            hover_name="station_name",
            size="total_volume",
            # We display the values in the title now too
            title=f"<b>Strategic Stress Map</b><br><sup>Valet Vol > {int(self.vol_threshold):,} | Expansion Flow > {self.flow_threshold:.1f}</sup>",
            color_discrete_map={
                "VALET_SERVICE": "#d62728",      # Red
                "EXPAND_DOCKS": "#ff7f0e",       # Orange
                "PRIORITY_RESTOCK": "#1f77b4",   # Blue
                "STANDARD_OPS": "#e5e5e5"        # Gray
            },
            labels={
                "avg_net_flow": "Net Flow (Left=Starvation, Right=Blocking)",
                "total_volume": "Total Activity Intensity"
            },
            opacity=0.8
        )

        # Dynamic Reference Lines
        fig.add_hline(y=self.vol_threshold, line_dash="dot", line_color="red", annotation_text="Valet Threshold", annotation_position="top left")
        fig.add_vline(x=self.flow_threshold, line_dash="dot", line_color="orange", annotation_text=f"Blocking (+{self.flow_threshold:.1f})", annotation_position="top right")
        fig.add_vline(x=-self.flow_threshold, line_dash="dot", line_color="blue", annotation_text=f"Starvation (-{self.flow_threshold:.1f})", annotation_position="top left")

        fig.update_layout(template="plotly_white", height=600)
        return fig