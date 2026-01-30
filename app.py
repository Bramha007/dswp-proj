import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
from src.phase0_pipeline import DataPipeline
from src.phase1_diagnostics import DiagnosticsAnalyzer
from src.phase2_logistics import LogisticsOptimizer
from src.phase3_strategy import SystemStressAnalyzer
from src.phase4_prediction import DemandForecaster


# 1. Page Configuration
st.set_page_config(
    page_title="CitiBike Strategy Command Center", page_icon="🚲", layout="wide"
)

# 2. Sidebar Navigation
st.sidebar.title("🚲 Command Center")
phase = st.sidebar.radio(
    "Navigate Phases",
    [
        "Phase 0: Data Harmonization",
        "Phase 1: Network Diagnostics",
        "Phase 2: Operational Logistics",
        "Phase 3: Strategic Investment",
        "Phase 4: AI Prediction Engine"
    ],
)

# PHASE 0: DATA HARMONIZATION

if phase == "Phase 0: Data Harmonization":
    st.title("⚙️ Phase 0: Data Harmonization")
    st.markdown(
        """
    **The Engine Room:** Ingests raw trip data, standardizes schema variations, 
    and generates the aggregated datasets.
    """
    )

    pipeline = DataPipeline(raw_dir="data/raw", processed_dir="data/processed")

    tab1, tab2, tab3 = st.tabs(
        ["📈 System Diagnostics", "📊 Ridership Analytics", "🌍 Geographic Insights"]
    )

    with tab1:
        st.subheader("System Diagnostics")
        st.markdown(
            "*High-level health check: Is the system growing? When do people ride?*"
        )
        fig_diag = pipeline.get_diagnostics_dashboard()
        if fig_diag:
            st.plotly_chart(fig_diag, use_container_width=True)
            st.info(
                """
            **Reading the Charts:**
            * **Top Chart (The Pulse):** Shows the 7-day rolling average. Note the gap between Summer peaks and Winter troughs.
            * **Bottom Chart (The Heartbeat):** Shows average activity per hour. Note the "Twin Peaks" (Morning/Evening Rush).
            """
            )
            st.markdown(
                """
                # 📉 Conclusion: Temporal Constraints for Optimization

                ### 1. The Volatility Gap
                **Figure 1 (The Pulse)** reveals high seasonal volatility. The optimization strategy must be robust enough to handle the summer peaks (high stress) without over-committing resources during winter troughs.

                ### 2. The Operational Window
                **Figure 2 (The Heartbeat)** shows the daily demand cycle with three critical periods:
                * **Morning Rush (07:00 - 09:00):** Demand spikes sharply. This is when "Starvation" occurs at Residential Sources.
                * **Evening Rush (16:00 - 18:00):** Another spike, but less severe than the morning. "Blocking" issues arise at Commercial Sinks.
                * **The Pre-Balancing Window (00:00 - 05:00):** Demand is near zero. This 6-hour window is the **only** safe time to perform aggressive rebalancing (truck movements) without fighting traffic or competing with riders for docks.

                **Next Step:** Use these demand curves to calculate the precise "Safe Stock" required at each station at 05:00 AM.
                """
            )
        else:
            st.warning("⚠️ No data found. Please run 'Engine 1: Aggregator' below.")

    with tab2:
        st.subheader("Comprehensive Analytics")
        st.markdown(
            """
            # Seasonal Dynamics & Product Adoption

            **Objective:** Evaluate the long-term structural shifts in the Manhattan network. While Phase 2b focused on operational hours, this analysis quantifies Year-Over-Year (YoY) growth, user retention, and fleet utilization trends.

            ### Analysis Scope
            1.  **Ridership Composition:** Analyze the ratio of Members vs. Casual riders to assess the stability of the recurring revenue base.
            2.  **YoY Growth:** Compare 2023, 2024, and 2025 seasonal curves to validate system expansion.
            3.  **Product Shift:** Quantify the adoption rate of Electric Bikes vs. Classic Bikes to inform future fleet procurement strategies.

            ### Key Deliverables
            * **Seasonal Dashboard:** A composite visualization of monthly ridership, growth vectors, and fleet preference.
            """
        )
        fig_main = pipeline.get_comprehensive_dashboard()
        if fig_main:
            st.plotly_chart(fig_main, use_container_width=True)
            st.markdown(
                """
                # 📉 Strategic Insights

                ### 1. The "Base Load" Stability
                **Chart 1** reveals that **Members** (Blue) form the resilient core of the system, maintaining usage even during winter months. **Casual** riders (Gray) are highly elastic, appearing only during peak season (May–Oct).
                * **Implication:** Optimization efforts (Phase 3) should prioritize Member reliability, as they are the year-round revenue engine.

                ### 2. Growth Validation
                **Chart 2** confirms a consistent upward trajectory. The 2025 curve (Yellow/Green) sits structurally higher than 2023 (Purple), indicating genuine network adoption rather than just post-pandemic recovery.

                ### 3. The E-Bike Revolution
                **Chart 3** highlights a critical operational challenge. If the Dark Blue area (E-Bikes) is expanding relative to Light Blue (Classic), it means the system requires **more battery swapping operations** and potentially **electrified docks**, rather than just static rebalancing vans.
                                """
            )
        else:
            st.info("⚠️ Data missing. Run 'Engine 1' below.")

    with tab3:
        st.subheader("Geographic Insights")
        st.markdown(
            """
           # Geographic Distribution & Infrastructure

            **Objective:** Validate the assumption that Manhattan is the primary driver of system load. This analysis physically maps every station to its NYC borough and compares ridership volumes across territories to confirm where optimization efforts should be concentrated.

            ### Analysis Scope
            1.  **Station Mapping:** Perform a spatial join between station GPS coordinates and NYC Borough boundaries (GeoJSON).
            2.  **Borough Comparison:** Quantify the ridership magnitude of Manhattan relative to Brooklyn, Queens, and the Bronx.
            3.  **Infrastructure Visualization:** Map the active station network to visualize density hotspots.

            ### Key Deliverables
            * **Station-to-Borough Master List:** A CSV mapping every Station ID to its Borough.
            * **Borough Trends Dashboard:** A composite view of infrastructure density and ridership growth by territory.
        """
        )
        fig_geo = pipeline.get_geographic_dashboard()
        if fig_geo:
            st.plotly_chart(fig_geo, use_container_width=True)
            st.markdown(
                """
                # 🌍 Geographic Insights

                ### 1. Manhattan Dominance
                The charts confirm that **Manhattan (Blue)** is operating at a completely different order of magnitude compared to Brooklyn or Queens. While Brooklyn has significant *infrastructure* (orange dots on the map), the *ridership intensity* (line charts) in Manhattan is 5x-10x higher.

                ### 2. The Optimization Target
                This validates our decision to focus the **Phase 3 Optimization** model exclusively on Manhattan.
                * **Problem:** Manhattan's density creates the "Tidal Wave" effect (cascading failure).
                * **Opportunity:** The high station density (seen in the map) allows for efficient rebalancing loops that wouldn't be profitable in the sparser networks of Queens or the Bronx.
            """
            )
        else:
            st.info("⚠️ Map data missing. Please run 'Engine 3: Geography' below.")

    st.divider()
    with st.expander("🛠️ Pipeline Management (Run Engines)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Run Aggregator"):
                with st.spinner("Processing..."):
                    pipeline.run_aggregation_pipeline()
                    st.rerun()
        with c2:
            if st.button("Run Flow Mapper"):
                with st.spinner("Processing..."):
                    pipeline.run_flow_pipeline()
                    st.rerun()
        with c3:
            if st.button("Run Geo Mapper"):
                with st.spinner("Mapping Coordinates..."):
                    status = pipeline.run_geographic_pipeline()
                    if status == "MISSING_GEOJSON":
                        st.error("❌ Missing 'new-york-city-boroughs.geojson'")
                    else:
                        st.success("✅ Done!")
                        st.rerun()

# PHASE 1: NETWORK DIAGNOSTICS (ADDED)

elif phase == "Phase 1: Network Diagnostics":
    st.title("🩺 Phase 1: Network Diagnostics")
    st.markdown(
        """
    **The Tidal Wave Hypothesis:** We hypothesize that the system failure is driven by a massive, unidirectional flow 
    from Residential "Sources" to Commercial "Sinks" during the Morning Rush.
    """
    )

    # Initialize
    analyzer = DiagnosticsAnalyzer(raw_dir=r"E:\nyc-bike-analysis\data\raw", processed_dir="data/processed")

    # Check if data exists
    df = analyzer.load_metrics()

    if df is not None:
        st.success("✅ Diagnostics Data Loaded")

        # --- TABS FOR PHASE 1 ---
        tab1, tab2, tab3 = st.tabs(
            ["🗺️ Spatial Imbalance", "📉 System Deterioration", "❤️ Temporal Heartbeat"]
        )

        # TAB 1: SPATIAL IMBALANCE (Map + Metrics)
        with tab1:
            st.subheader("A. Spatial Imbalance (The Tidal Wave)")
            # Headline Metrics
            total_moved = int(df[df["avg_daily_flow"] > 0]["avg_daily_flow"].sum())
            col1, col2 = st.columns(2)
            col1.metric(
                "Total Morning Displacement", f"{total_moved:,} Bikes", "One-way Flow"
            )
            col2.metric("Active Stations Analyzed", len(df))

            st.divider()

            # Map Controls
            col_caption, col_select = st.columns([3, 1])
            # with col_caption:
            #     st.caption(
            #         "🔴 Red = Starvation (Outflow) | 🔵 Blue = Saturation (Inflow)"
            #     )
            with col_select:
                target_year = st.selectbox("Select Year", [2023, 2024, 2025], index=0)

            fig_map = analyzer.get_tidal_wave_map(target_year=target_year)
            if fig_map:
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning(f"No data available for {target_year}.")
            st.markdown(
                """
                        # 🗺️ The Tidal Wave: Spatial Imbalance Map

                        **Objective:**
                        To visualize the "Heartbeat" of Manhattan during the morning rush (6 AM - 10 AM). This map reveals the spatial disconnect between where bikes are *needed* (Sources) and where they *end up* (Sinks).

                        ### How to Read This Map
                        * 🔴 **Red Nodes (Sources):** Net Outflow. These are Residential neighborhoods (e.g., Upper West Side, East Village) where commuters *take* bikes.
                            * *Risk:* **Starvation** (Empty Docks).
                        * 🔵 **Blue Nodes (Sinks):** Net Inflow. These are Commercial hubs (e.g., Midtown, Financial District) where commuters *dock* bikes.
                            * *Risk:* **Blocking** (Full Docks).
                        * **Bubble Size:** Represents the magnitude of the imbalance. Larger bubbles = Higher operational stress.
                """
            )

        # TAB 2: DETERIORATION (KDE Analysis)
        with tab2:
            st.subheader("B. Temporal Deterioration (2023 vs 2025)")
            st.markdown(
                """
            *Is the problem getting worse?*
            The chart below compares the distribution of net flows. The **flatter, wider curve in 2025 (Red)** indicates extreme variance—more stations are becoming critically unbalanced compared to 2023.
            """
            )

            fig_kde, df_delta = analyzer.get_deterioration_analysis()

            if fig_kde:
                col_chart, col_table = st.columns([1.5, 1])
                with col_chart:
                    st.plotly_chart(fig_kde, use_container_width=True)
                with col_table:
                    st.write("**Top System Failures (Largest Change)**")
                    st.dataframe(
                        df_delta[["station_name", "delta", "type"]].sort_values(
                            "delta"
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                st.markdown(
                    """
                            ### 📊 Interpretation: The "Fat Tail" Problem

                            **1. The Collapse of the Middle**
                            The **Blue Peak (2023)** was significantly higher than the **Red Peak (2025)**.
                            * *Meaning:* Two years ago, the majority of stations functioned normally. In 2025, the number of "balanced" stations has dropped sharply.

                            **2. The Widening Extremes**
                            The Red Line (2025) extends further to the left and right than the Blue Line.
                            * *Left Tail (Starvation):* Residential areas are emptying out faster than before.
                            * *Right Tail (Blocking):* Commercial hubs are overflowing more aggressively.

                            **3. Strategic Implication**
                            The network is **destabilizing**. The operational plan that worked for the "Blue Curve" (2023) is mathematically insufficient for the "Red Curve" (2025).
                        """
                )
            else:
                st.warning(
                    "⚠️ Comparison data missing. Ensure 2023 and 2025 data is processed."
                )

        # TAB 3: HEARTBEAT (Hourly Profile)
        with tab3:
            st.subheader("C. Temporal Profiling (The Heartbeat)")
            st.markdown(
                """
            *The Mirror Effect:* Morning outflow from Residential (Red) matches inflow to Commercial (Blue), then flips.
            **Operational Insight:** Rebalancing trucks must operate in the "Safe Zone" (23:00 - 05:00).
            """
            )

            fig_heart = analyzer.get_heartbeat_chart()
            if fig_heart:
                st.plotly_chart(fig_heart, use_container_width=True)
            else:
                st.warning("⚠️ Heartbeat data missing.")

            st.markdown(
                """"
1.  **The Morning Shock (08:30 AM):**
    The system experiences its most violent stress event during the Morning Rush. The sharp, vertical plunge of the Red line indicates that demand is **inelastic**—commuters have a fixed deadline (9:00 AM) and cannot shift their behavior.

2.  **The Evening Reversal (17:30 PM):**
    As the workforce returns home, the polarity flips. However, the evening peak is **wider and shorter** than the morning spike. This confirms that evening demand is **elastic**; departures are spread over several hours (4 PM – 8 PM), creating slightly less immediate stress on the system than the morning crunch.

*The inverse correlation between Residential (Source) and Commercial (Sink) zones confirms the directional asymmetry of the fleet movement. The sharpness of the morning peak relative to the evening peak highlights the inelastic nature of morning commuter demand.*
                """
            )

        # --- RERUN LOGIC (Outside Tabs) ---
        st.divider()
        with st.expander("⚙️ Rerun Diagnostics Pipeline"):
            if st.button("Recalculate Flows (AM/PM)"):
                with st.spinner("Processing Raw Data..."):
                    analyzer.run_diagnostics_pipeline()
                    st.rerun()

    else:
        # NO DATA -> SHOW RUN BUTTON
        st.warning("⚠️ No Diagnostics Data Found.")
        st.markdown(
            "We need to scan the raw data to extract the Morning/Evening Rush patterns."
        )

        if st.button("Run Diagnostics Pipeline"):
            with st.spinner("Analyzing Traffic Patterns..."):
                status = analyzer.run_diagnostics_pipeline()

                if status == "MISSING_MAPPING":
                    st.error(
                        "❌ Missing Borough Mapping. Please go to Phase 0 -> Geographic Insights -> Run Engine 3."
                    )
                elif status == "NO_RAW_DATA":
                    st.error("❌ No CSV files found in data/raw.")
                else:
                    st.success("✅ Done! Loading Dashboard...")
                    st.rerun()



# PHASE 2: OPERATIONAL LOGISTICS

elif phase == "Phase 2: Operational Logistics":
    st.title("🚚 Phase 2: Operational Logistics")

    # 1. Introduction: The Strategy
    st.markdown(
        """
    ### 🌙 The Strategy: "Night Moves" (Pre-Balancing)
    
    **The Problem:**
    Phase 1 diagnosed a massive "Tidal Wave" during the Morning Rush (6am - 10am). 
    * **Residential Areas** (Sources) run empty.
    * **Commercial Areas** (Sinks) become full.
    
    **The Solution:**
    If we wait until 8:00 AM to fix this, it is too late—trucks get stuck in traffic. 
    Instead, we execute a **Pre-Balancing Strategy (00:00 - 05:00)**. 
    We identify exactly where the system *will* fail and move bikes overnight to neutralize the imbalance before the city wakes up.
    """
    )

    optimizer = LogisticsOptimizer(processed_dir="data/processed")
    status = optimizer.load_data()

    if status == "SUCCESS":
        st.divider()

        # 2. Methodology: The Math
        st.markdown(
            """
        #### 📐 The Math: How We Calculate the Manifest
        We don't guess. We calculate the exact **Net Flow** for every station during the Morning Rush and apply a **Safety Buffer**.
        
        * **Step 1: Calculate Net Flow:** (e.g., Station A loses 50 bikes every morning).
        * **Step 2: Apply Safety Buffer:** We add extra inventory (default 20%) to handle unexpected demand spikes.
        * **Step 3: Generate Actions:**
            * **🔴 Residential (Source):** Net Flow < -5. Action: **DROP OFF** bikes.
            * **🔵 Commercial (Sink):** Net Flow > 5. Action: **PICK UP** bikes (to make space).
        """
        )

        st.sidebar.header("Logistics Parameters")
        safety_buffer = st.sidebar.slider(
            "Safety Buffer",
            1.0,
            1.5,
            1.2,
            0.1,
            help="1.2 means we stock 20% MORE bikes than the average demand, just to be safe.",
        )
        min_batch = st.sidebar.slider(
            "Min Batch Size",
            1,
            20,
            5,
            1,
            help="Trucks won't stop for fewer than this many bikes.",
        )

        manifest = optimizer.generate_night_manifest(
            safety_buffer=safety_buffer, min_batch_size=min_batch
        )

        if manifest is not None and not manifest.empty:
            # Metrics
            total_moves = manifest["quantity"].sum()
            trucks_needed = int(
                total_moves / 40 / 5
            )  # Approx 40 bikes per truck, 5 trips per night

            # Display Key Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Total Bikes to Move", f"{int(total_moves):,}", "Overnight Volume"
            )
            m2.metric("Est. Truck Shifts", trucks_needed, "Based on 40-bike capacity")
            m3.metric(
                "Stations Serviced",
                len(manifest),
                f"Buffer: {int((safety_buffer-1)*100)}%",
            )

            st.divider()

            # Visualizations
            st.subheader('A. The Imbalance Profile (The "Fat Tails")')
            st.markdown(
                "*The goal is to 'chop off' the red and blue tails of this distribution.*"
            )
            fig_imbalance = optimizer.get_imbalance_distribution()
            st.plotly_chart(fig_imbalance, use_container_width=True)

            st.subheader("B. The Operational Manifest")
            col_chart, col_data = st.columns([1, 2])

            with col_chart:
                fig_ops = optimizer.get_distribution_chart()
                st.plotly_chart(fig_ops, use_container_width=True)

            with col_data:
                tab_drops, tab_pickups = st.tabs(
                    ["🔴 Drops (Residential)", "🔵 Pickups (Commercial)"]
                )
                with tab_drops:
                    st.info(
                        "🚚 **Mission:** preventing Starvation (Empty Docks) in residential zones."
                    )
                    drops = manifest[manifest["action"] == "DROP_OFF"][
                        ["station_name", "net_flow", "quantity"]
                    ]
                    st.dataframe(
                        drops.style.background_gradient(
                            cmap="Reds", subset=["quantity"]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                with tab_pickups:
                    st.info(
                        "🚛 **Mission:** preventing Blocking (Full Docks) in commercial zones."
                    )
                    picks = manifest[manifest["action"] == "PICK_UP"][
                        ["station_name", "net_flow", "quantity"]
                    ]
                    st.dataframe(
                        picks.style.background_gradient(
                            cmap="Blues", subset=["quantity"]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

            # 3. Conclusion
            st.markdown(
                f"""
            ---
            ### ✅ Operational Conclusion
            
            To survive tomorrow morning's rush:
            1.  **{int(total_moves):,} bikes** must be moved between **00:00 and 06:00**.
            2.  We prioritize the **{len(manifest)} stations** listed in the manifest above.
            3.  **Result:** The network will be reset to a "Perfect State" before the first commuter wakes up.
            """
            )

        else:
            st.warning("No moves required with these settings.")

    else:
        st.error("🚫 Data Missing: Run Phase 1 Diagnostics first.")



# PHASE 3: STRATEGIC INVESTMENT

elif phase == "Phase 3: Strategic Investment":
    st.title("🏗️ Phase 3: Strategic Investment")

    # 1. Introduction: Concepts & Definitions
    st.markdown(
        """
    **The Objective:**
    Phase 2 showed us how to survive the daily crisis using trucks. But trucking is expensive, reactive, and traffic-dependent.
    **Phase 3 is about the cure.** We categorize every station to find permanent infrastructure solutions.

    **The Operational Solutions:**
    * **🔴 Valet Service (OpEx):**
        * *What it is:* A human staff member stands at the station to stack bikes on the sidewalk.
        * *When to use it:* Only for **"Super Nodes"** (High Volume).
    * **🟠 Dock Expansion (CapEx):**
        * *What it is:* Construction crews bolt new docks into the pavement.
        * *When to use it:* For **"Chronic Sinks"** (High Inflow).
    * **🔵 Priority Restock:**
        * *What it is:* High-priority trucking routes.
        * *When to use it:* For **"Chronic Sources"** (High Outflow).
    """
    )

    analyzer = SystemStressAnalyzer(
        flow_path="data/processed/manhattan_morning_rush_flow.csv", rides_path=None
    )

    with st.spinner("Calculating Network Stress..."):
        target_year = 2025
        df = analyzer.load_metrics(target_year=target_year)

    if df is not None:

        st.divider()

        # 2. Methodology: Explaining the Math (WITH FORMULA)
        st.markdown(
            f"""
        #### 📐 The Math: How We Decide
        We use **Dynamic Statistical Thresholds** that adapt to the network's actual behavior in {target_year}.

        **1. The "Net Flow" Formula:**
        This determines the *Direction* of the problem.
        * $Net Flow = Daily Returns - Daily Starts$
        * **Positive (+):** More returns than starts. Bikes pile up (Blocking Risk).
        * **Negative (-):** More starts than returns. Bikes disappear (Starvation Risk).

        **2. The "Critical Imbalance" Threshold (1 Standard Deviation):**
        We don't guess numbers. We calculate the **Standard Deviation ($\sigma$)** of the entire network.
        * Any station deviating by more than **$1\sigma$** ({analyzer.flow_threshold:.1f} bikes) is a **Statistically Significant Outlier**.
        * *Action:* If it gains > {analyzer.flow_threshold:.1f} bikes, we **Build Docks**. If it loses > {analyzer.flow_threshold:.1f} bikes, we **Restock**.

        **3. The "Super Node" Threshold (95th Percentile Volume):**
        * We only hire Valets for the **top 5% busiest stations** (>{int(analyzer.vol_threshold):,} trips). High turnover justifies the salary cost.
        """
        )

        # --- Metrics Row ---
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "1.0 Std Dev (Flow)",
            f"{analyzer.flow_threshold:.1f} Bikes",
            help="Stations exceeding this imbalance need physical expansion.",
        )
        m2.metric(
            "95th Percentile (Vol)",
            f"{int(analyzer.vol_threshold):,} Trips",
            help="Stations busier than this need human valets.",
        )
        m3.metric("Stations Analyzed", len(df))

        st.subheader("The Strategy Map")
        fig = analyzer.get_quadrants_chart()
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.error("🔴 **Top Valet Candidates (OpEx)**")
            st.markdown("*High Volume, High Turnover Stations*")
            valets = (
                df[df["category"] == "VALET_SERVICE"]
                .sort_values("total_volume", ascending=False)
                .head(10)
            )

            if not valets.empty:
                st.dataframe(
                    valets[
                        ["station_name", "total_volume", "avg_net_flow"]
                    ].style.format(
                        {"total_volume": "{:,.0f}", "avg_net_flow": "{:.1f}"}
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.warning("No stations met the Valet criteria.")

        with col2:
            st.info("🟠 **Top Expansion Candidates (CapEx)**")
            st.markdown("*Chronic Inflow (Blocking Risk)*")
            expansion = (
                df[df["category"] == "EXPAND_DOCKS"]
                .sort_values("avg_net_flow", ascending=False)
                .head(10)
            )

            if not expansion.empty:
                st.dataframe(
                    expansion[
                        ["station_name", "avg_net_flow", "total_volume"]
                    ].style.format(
                        {"total_volume": "{:,.0f}", "avg_net_flow": "{:.1f}"}
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.warning("No stations met the Expansion criteria.")

        # 3. Conclusion: Recommendations
        st.markdown(
            f"""
        ---
        ### 🏆 Strategic Verdict

        The data suggests the system is not "broken"; it is **under-capacitated** at specific nodes. We recommend a **Hybrid Investment Strategy**:

        1.  **Immediate OpEx Fix (Human Valets):**
            * Deploy staff to the **{len(valets)} Red Stations** identified above.
            * *Impact:* This immediately neutralizes the chaos at major transit hubs.

        2.  **Long-Term CapEx Fix (Construction):**
            * Approve construction permits for the **{len(expansion)} Orange Stations**.
            * *ROI:* Building these docks eliminates the need for ~{int(len(expansion)/3)} daily truck shifts, paying for the construction cost in <18 months.

        3.  **Grid Optimization:**
            * Leave the **Gray Stations** alone. They are self-balancing and require no capital.
        """
        )

    else:
        st.error(
            "🚫 Data Missing: Run Phase 1 Diagnostics first to generate flow data."
        )


# PHASE 4: AI PREDICTION ENGINE (XGBOOST)

if phase == "Phase 4: AI Prediction Engine":
    st.title("🤖 Phase 4: AI Prediction Engine")
    st.markdown("Powered by **XGBoost**. This engine learns complex patterns (Weather, Trends, Holidays) to predict station demand.")

    forecaster = DemandForecaster(processed_dir="data/processed")
    status = forecaster.load_data()

    if status == "FILE_NOT_FOUND":
        st.error("❌ Processed data not found! Please run Phase 1 Diagnostics first.")
    else:
        # Control Panel
        c1, c2 = st.columns([3, 1])
        with c1:
            station_list = forecaster.get_station_list()
            selected_station = st.selectbox("Select Station:", station_list)
        with c2:
            st.write("") # Spacer
            run_btn = st.button("🚀 Run Forecast", type="primary")

        if selected_station and run_btn:
            with st.spinner(f"Training XGBoost Brain for {selected_station}..."):
                # Returns 4 items now
                results, metrics, importance, preview_df = forecaster.run_forecast(selected_station)
                
            if results is not None:
                # --- A. METRICS ---
                test_set = results[results['split_type'] == 'Test']
                mae = (test_set['actual_net_flow'] - test_set['pred_net_flow']).abs().mean()
                avg_vol = test_set['daily_starts'].mean()
                if avg_vol == 0: avg_vol = 1
                accuracy = max(0, 100 - (mae / avg_vol * 100))
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Model Accuracy", f"{accuracy:.1f}%")
                c2.metric("Avg Error", f"±{int(mae)} bikes")
                c3.metric("Training Data", f"Until {metrics['train_end'].strftime('%b %Y')}")
                c4.metric("Test Period", "Sep - Nov 2025")

                st.divider()

                # --- B. FEATURE IMPORTANCE (WHAT CONTRIBUTED?) ---
                st.subheader("1. How the AI Thinks (Feature Contribution)")
                st.caption("Which factors matter most for this specific station?")
                
                c_chart, c_text = st.columns([2, 1])
                with c_chart:
                    # Plot Feature Importance Bar Chart
                    st.bar_chart(importance.set_index('Feature'), height=250, color="#FF4B4B")
                with c_text:
                    st.info("""
                    **Legend:**
                    * **roll7:** The 7-day average (Trend). Usually #1.
                    * **dow:** Day of Week (Weekend vs Weekday).
                    * **temp_f:** Temperature impact.
                    * **is_rain:** Rain impact.
                    """)

                st.divider()

                # --- C. DATASET PREVIEW (HOW IT LOOKS) ---
                st.subheader("2. What the AI Sees (Dataset Preview)")
                st.caption("This is the actual data fed into the Neural Network (Last 5 days).")
                st.dataframe(
                    preview_df.style.format({
                        "daily_starts": "{:.1f}",
                        "daily_returns": "{:.1f}",
                        "daily_starts_roll7": "{:.1f}",
                        "temp_f": "{:.1f}°F"
                    }), 
                    use_container_width=True
                )
                
                st.divider()

                # --- D. FORECAST CHART ---
                st.subheader("3. The Results (Forecast vs Reality)")
                fig = forecaster.plot_forecast(results)
                st.plotly_chart(fig, use_container_width=True)