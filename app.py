"""
Blacklane Quality Dashboard
---------------------------
Interactive dashboard for the three quality pillars and LSP Scorecard,
built on the Q1 2019 EMEA accepted tours + rejected tours dataset.

Run locally:
    streamlit run app.py

Expected input: an Excel file with two sheets:
  - "EMEA Q1 19 Accepted Tours"
  - "EMEA Q1 19 Rejected Tours"
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Blacklane Quality Dashboard",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Operational color palette — readable, not flashy
COLOR_INK = "#1F2937"
COLOR_MUTED = "#6B7280"
COLOR_BAD = "#991B1B"
COLOR_WARN = "#B45309"
COLOR_GOOD = "#065F46"
COLOR_PANEL = "#F3F4F6"
COLOR_BG = "#FFFFFF"

SEK_TO_EUR = 11  # Stockholm currency correction

# Tier thresholds for LSP Scorecard (based on Cell B rate)
TIER_D_THRESHOLD = 15.0  # >= 15% Cell B → Tier D (deactivation candidate)
TIER_C_THRESHOLD = 8.0   # 8-15% → Tier C (improvement plan)
TIER_B_THRESHOLD = 3.0   # 3-8% → Tier B (monitoring)
                         # < 3% → Tier A (standard)

TIER_COLORS = {
    "A": COLOR_GOOD,
    "B": "#0E7490",  # teal
    "C": COLOR_WARN,
    "D": COLOR_BAD,
}


def value_gradient(value: float, vmin: float, vmax: float, palette: str = "red") -> str:
    """
    Return a CSS background-color style for a numeric value, scaled vmin→vmax.
    Replaces pandas.Styler.background_gradient() to avoid matplotlib dependency.

    palette: 'red' (low=light, high=dark red), 'green' (low=light, high=dark green),
             'blue' (low=light, high=dark blue).
    """
    if pd.isna(value):
        return ""
    # Normalize to [0, 1]
    if vmax == vmin:
        intensity = 0.5
    else:
        intensity = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))

    # Two-stop interpolation: light tint → strong color
    if palette == "red":
        # #FEE2E2 (light red) → #991B1B (dark red)
        r = int(254 + (153 - 254) * intensity)
        g = int(226 + (27 - 226) * intensity)
        b = int(226 + (27 - 226) * intensity)
    elif palette == "green":
        # #D1FAE5 (light green) → #065F46 (dark green)
        r = int(209 + (6 - 209) * intensity)
        g = int(250 + (95 - 250) * intensity)
        b = int(229 + (70 - 229) * intensity)
    elif palette == "blue":
        # #DBEAFE (light blue) → #1E40AF (dark blue)
        r = int(219 + (30 - 219) * intensity)
        g = int(234 + (64 - 234) * intensity)
        b = int(254 + (175 - 254) * intensity)
    else:
        return ""

    # Pick text color for contrast: dark text on light bg, white on dark bg
    text_color = "#1F2937" if intensity < 0.55 else "#FFFFFF"
    return f"background-color: rgb({r},{g},{b}); color: {text_color};"

# ============================================================
# DATA LOADING + PREPARATION
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(uploaded_bytes: bytes | None, fallback_path: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load accepted + rejected tours from uploaded file or fallback path."""
    if uploaded_bytes is not None:
        accepted = pd.read_excel(io.BytesIO(uploaded_bytes), sheet_name="EMEA Q1 19 Accepted Tours")
        rejected = pd.read_excel(io.BytesIO(uploaded_bytes), sheet_name="EMEA Q1 19 Rejected Tours")
    elif fallback_path and Path(fallback_path).exists():
        accepted = pd.read_excel(fallback_path, sheet_name="EMEA Q1 19 Accepted Tours")
        rejected = pd.read_excel(fallback_path, sheet_name="EMEA Q1 19 Rejected Tours")
    else:
        raise FileNotFoundError("No data file provided.")

    # Stockholm currency correction (SEK → EUR)
    stockholm_mask = accepted["Ride Bd"] == "Stockholm"
    for col in ["Avg. Winning Price", "Avg. Gross Revenue"]:
        accepted.loc[stockholm_mask, col] = accepted.loc[stockholm_mask, col] / SEK_TO_EUR

    # Parse timestamps
    accepted["pickup_dt"] = pd.to_datetime(accepted["Pickup at local time"], errors="coerce")
    accepted["finished_dt"] = pd.to_datetime(accepted["Finished at local time"], errors="coerce")
    accepted["booking_dt"] = pd.to_datetime(accepted["Booking Date Local Time"], errors="coerce")
    accepted["ride_dt"] = pd.to_datetime(accepted["Ride Date Local Time"], errors="coerce")
    accepted["accepted_dt"] = pd.to_datetime(accepted["Accepted At Local Time"], errors="coerce")
    accepted["date"] = accepted["pickup_dt"].dt.date

    # Operational telemetry — derived signals
    # Pickup arrival delta: chauffeur arrival minus booked pickup time, in minutes.
    # Negative = early (good); positive = late.
    mask = accepted["pickup_dt"].notna() & accepted["ride_dt"].notna()
    accepted.loc[mask, "pickup_delta_min"] = (
        (accepted.loc[mask, "pickup_dt"] - accepted.loc[mask, "ride_dt"]).dt.total_seconds() / 60
    )

    # Booking-to-acceptance latency: time between booking and an LSP claiming it (seconds).
    mask = accepted["accepted_dt"].notna() & accepted["booking_dt"].notna()
    accepted.loc[mask, "ba_latency_sec"] = (
        (accepted.loc[mask, "accepted_dt"] - accepted.loc[mask, "booking_dt"]).dt.total_seconds()
    )

    # Lead time at booking: hours between booking and scheduled ride.
    mask = accepted["ride_dt"].notna() & accepted["booking_dt"].notna()
    accepted.loc[mask, "lead_time_hr"] = (
        (accepted.loc[mask, "ride_dt"] - accepted.loc[mask, "booking_dt"]).dt.total_seconds() / 3600
    )

    # Trip duration (hours) and speed (km/h)
    mask = accepted["finished_dt"].notna() & accepted["pickup_dt"].notna()
    accepted.loc[mask, "trip_hr"] = (
        (accepted.loc[mask, "finished_dt"] - accepted.loc[mask, "pickup_dt"]).dt.total_seconds() / 3600
    )
    accepted["speed_kmh"] = accepted["Route Distance KM"] / accepted["trip_hr"]

    # Derived classification flags
    accepted["has_pickup_ts"] = accepted["Pickup at local time"].notna()
    accepted["is_no_show"] = accepted["Tour State"] == "no_show"
    accepted["is_finished"] = accepted["Tour State"] == "finished"
    accepted["is_cell_b"] = accepted["has_pickup_ts"] & accepted["is_no_show"]      # chauffeur arrived, no-show
    accepted["is_cell_c"] = (~accepted["has_pickup_ts"]) & accepted["is_no_show"]    # true chauffeur no-show
    accepted["is_cell_d"] = (~accepted["has_pickup_ts"]) & accepted["is_finished"]   # inverse anomaly

    # Rating availability
    accepted["is_rated"] = accepted["Avg. Driver Rating"].notna() | accepted["Avg. Car Rating"].notna()

    # Zero-km classification (for the Cell B split)
    accepted["is_zero_km"] = accepted["Route Distance KM"] == 0

    # Contribution per ride
    accepted["contribution"] = accepted["Avg. Gross Revenue"] - accepted["Avg. Winning Price"]

    return accepted, rejected


def filter_data(accepted: pd.DataFrame, rejected: pd.DataFrame, filters: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply sidebar filters to both datasets."""
    a = accepted.copy()
    r = rejected.copy()

    if filters.get("cities"):
        a = a[a["Ride Bd"].isin(filters["cities"])]
        r = r[r["Ride Bd"].isin(filters["cities"])]
    if filters.get("transfer_types"):
        a = a[a["Transfer type"].isin(filters["transfer_types"])]
        if "Transfer type" in r.columns:
            r = r[r["Transfer type"].isin(filters["transfer_types"])]
    if filters.get("classes"):
        a = a[a["Car Class"].isin(filters["classes"])]
        if "Car Class" in r.columns:
            r = r[r["Car Class"].isin(filters["classes"])]
    if filters.get("vip_only"):
        a = a[a["Is VIP Airline? (Y/N)"] == "Y"]
        if "Is VIP Airline? (Y/N)" in r.columns:
            r = r[r["Is VIP Airline? (Y/N)"] == "Y"]
    elif filters.get("non_vip_only"):
        a = a[a["Is VIP Airline? (Y/N)"] == "N"]
        if "Is VIP Airline? (Y/N)" in r.columns:
            r = r[r["Is VIP Airline? (Y/N)"] == "N"]

    return a, r


def assign_tier(cell_b_rate: float) -> str:
    """Classify an LSP into A/B/C/D based on Cell B rate."""
    if pd.isna(cell_b_rate):
        return "N/A"
    if cell_b_rate >= TIER_D_THRESHOLD:
        return "D"
    if cell_b_rate >= TIER_C_THRESHOLD:
        return "C"
    if cell_b_rate >= TIER_B_THRESHOLD:
        return "B"
    return "A"


def build_lsp_scorecard(accepted: pd.DataFrame, min_rides: int = 100) -> pd.DataFrame:
    """Build the LSP Scorecard with all measurable metrics."""
    if len(accepted) == 0:
        return pd.DataFrame()

    grouped = accepted.groupby("LSP Name").agg(
        rides=("Avg. Gross Revenue", "size"),
        revenue=("Avg. Gross Revenue", "sum"),
        avg_revenue=("Avg. Gross Revenue", "mean"),
        winning_price_total=("Avg. Winning Price", "sum"),
        contribution_total=("contribution", "sum"),
        cell_b_count=("is_cell_b", "sum"),
        cell_c_count=("is_cell_c", "sum"),
        cell_d_count=("is_cell_d", "sum"),
        finished_count=("is_finished", "sum"),
        rated_count=("is_rated", "sum"),
        vip_count=("Is VIP Airline? (Y/N)", lambda s: (s == "Y").sum()),
        avg_driver_rating=("Avg. Driver Rating", "mean"),
        avg_car_rating=("Avg. Car Rating", "mean"),
    ).reset_index()

    if len(grouped) == 0:
        return pd.DataFrame()

    grouped["volume_share_pct"] = 100 * grouped["rides"] / grouped["rides"].sum()
    grouped["vip_share_pct"] = 100 * grouped["vip_count"] / grouped["rides"]
    grouped["cell_b_rate_pct"] = 100 * grouped["cell_b_count"] / grouped["rides"]
    grouped["cell_c_rate_pct"] = 100 * grouped["cell_c_count"] / grouped["rides"]
    grouped["finished_rate_pct"] = 100 * grouped["finished_count"] / grouped["rides"]
    grouped["rating_coverage_pct"] = 100 * grouped["rated_count"] / grouped["rides"]
    grouped["contribution_per_ride"] = grouped["contribution_total"] / grouped["rides"]
    grouped["tier"] = grouped["cell_b_rate_pct"].apply(assign_tier)

    # Filter to LSPs with enough volume to be meaningful
    grouped = grouped[grouped["rides"] >= min_rides].copy()
    grouped = grouped.sort_values("rides", ascending=False)
    return grouped


def airport_pickup_failure_rate(accepted: pd.DataFrame, lsp_name: str) -> tuple[int, int, float]:
    """Compute Cell B rate on airport pickups for a specific LSP."""
    df = accepted[(accepted["LSP Name"] == lsp_name) & (accepted["Transfer type"] == "airport pickup")]
    if len(df) == 0:
        return 0, 0, 0.0
    cb = df["is_cell_b"].sum()
    return int(cb), int(len(df)), 100 * cb / len(df)


# ============================================================
# UI HELPERS
# ============================================================

def kpi_card(label: str, value: str, sub: str = "", color: str = COLOR_INK):
    """Render a single KPI as a styled card."""
    st.markdown(
        f"""
        <div style="
            background: {COLOR_BG};
            border: 1px solid #E5E7EB;
            border-left: 4px solid {color};
            padding: 14px 18px;
            border-radius: 4px;
            height: 100%;
        ">
            <div style="color: {COLOR_MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;">{label}</div>
            <div style="color: {color}; font-size: 28px; font-weight: 700; margin-top: 4px; font-feature-settings: 'tnum';">{value}</div>
            <div style="color: {COLOR_MUTED}; font-size: 12px; margin-top: 4px;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div style="margin: 28px 0 16px 0; padding-bottom: 8px; border-bottom: 2px solid {COLOR_INK};">
            <div style="font-size: 22px; font-weight: 700; color: {COLOR_INK};">{title}</div>
            {f'<div style="font-size: 13px; color: {COLOR_MUTED}; margin-top: 4px; font-style: italic;">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def tier_pill(tier: str) -> str:
    color = TIER_COLORS.get(tier, COLOR_MUTED)
    label = {
        "A": "A · Standard",
        "B": "B · Monitor",
        "C": "C · Improve",
        "D": "D · Review",
    }.get(tier, tier)
    return f'<span style="background:{color};color:#FFF;padding:2px 8px;border-radius:3px;font-size:11px;font-weight:600;letter-spacing:0.5px;">{label}</span>'


# ============================================================
# MAIN APP
# ============================================================

# Sidebar — data load + filters
with st.sidebar:
    st.markdown(f"<div style='font-size: 20px; font-weight: 700; color: {COLOR_INK}; padding-bottom: 6px;'>BLACKLANE QUALITY</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size: 11px; color: {COLOR_MUTED}; letter-spacing: 1px; text-transform: uppercase; padding-bottom: 18px;'>Q1 2019 EMEA · Dashboard</div>", unsafe_allow_html=True)

    st.subheader("Data")
    uploaded = st.file_uploader("Upload source Excel file", type=["xlsx", "xls"])
    fallback_path = st.text_input(
        "Or path to local file",
        value="data.xlsx",
        help="If no file is uploaded, will try to load from this path."
    )

    try:
        uploaded_bytes = uploaded.read() if uploaded else None
        accepted_full, rejected_full = load_data(uploaded_bytes, fallback_path)
        data_loaded = True
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

    st.success(f"Loaded {len(accepted_full):,} accepted + {len(rejected_full):,} rejected rides.")

    st.markdown("---")
    st.subheader("Filters")

    cities = st.multiselect(
        "Cities",
        options=sorted(accepted_full["Ride Bd"].dropna().unique().tolist()),
        default=[],
        help="Empty = all cities"
    )

    transfer_types = st.multiselect(
        "Transfer types",
        options=sorted(accepted_full["Transfer type"].dropna().unique().tolist()),
        default=[],
    )

    classes = st.multiselect(
        "Car classes",
        options=sorted(accepted_full["Car Class"].dropna().unique().tolist()),
        default=[],
    )

    vip_filter = st.radio(
        "Segment",
        options=["All", "VIP only", "Non-VIP only"],
        horizontal=True,
    )

    filters = {
        "cities": cities,
        "transfer_types": transfer_types,
        "classes": classes,
        "vip_only": vip_filter == "VIP only",
        "non_vip_only": vip_filter == "Non-VIP only",
    }

accepted, rejected = filter_data(accepted_full, rejected_full, filters)

# Top banner — title + framing
st.markdown(
    f"""
    <div style="padding: 16px 0 8px 0;">
        <div style="font-size: 13px; color: {COLOR_MUTED}; letter-spacing: 2px; font-weight: 600; text-transform: uppercase;">Quality Dashboard</div>
        <div style="font-size: 30px; font-weight: 700; color: {COLOR_INK}; margin-top: 4px;">Three pillars · LSP Scorecard · Operational telemetry</div>
        <div style="font-size: 14px; color: {COLOR_MUTED}; font-style: italic; margin-top: 8px; max-width: 800px;">
            Premium customers pay to stop thinking about the ride. Every metric here measures whether we kept that promise — without asking the customer for their time.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Active filter readout
active_filters = []
if cities: active_filters.append(f"{len(cities)} cit{'ies' if len(cities)>1 else 'y'}")
if transfer_types: active_filters.append(f"{len(transfer_types)} transfer type{'s' if len(transfer_types)>1 else ''}")
if classes: active_filters.append(f"{len(classes)} class{'es' if len(classes)>1 else ''}")
if vip_filter != "All": active_filters.append(vip_filter)
filter_text = " · ".join(active_filters) if active_filters else "All EMEA · all segments"
st.markdown(f"<div style='font-size: 11px; color: {COLOR_MUTED}; letter-spacing: 1px; padding-top: 8px;'>SCOPE: {filter_text.upper()} · {len(accepted):,} accepted, {len(rejected):,} rejected rides</div>", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================

tab_overview, tab_pillars, tab_scorecard, tab_telemetry, tab_chauffeurs = st.tabs([
    "Overview",
    "Quality pillars",
    "LSP Scorecard",
    "Operational telemetry",
    "Chauffeur audit",
])

# ============================================================
# TAB: OVERVIEW
# ============================================================

with tab_overview:
    section_header("Headline metrics", "The four numbers that anchor every operational conversation")

    # Top row of KPIs
    n_finished = int(accepted["is_finished"].sum())
    n_cell_b = int(accepted["is_cell_b"].sum())
    n_cell_c = int(accepted["is_cell_c"].sum())
    n_total_accepted = len(accepted)
    n_rejected = len(rejected)

    cell_b_rate = 100 * n_cell_b / n_total_accepted if n_total_accepted else 0
    finished_rate = 100 * n_finished / n_total_accepted if n_total_accepted else 0
    revenue_at_risk = float(accepted.loc[accepted["is_cell_b"], "Avg. Gross Revenue"].sum())
    rating_cov = 100 * int(accepted.loc[accepted["is_finished"], "is_rated"].sum()) / n_finished if n_finished else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("CELL B RATE", f"{cell_b_rate:.2f}%", f"{n_cell_b:,} ops no-shows · the marketplace leak", color=COLOR_BAD)
    with c2:
        kpi_card("COMPLETION", f"{finished_rate:.1f}%", f"{n_finished:,} of {n_total_accepted:,} accepted", color=COLOR_GOOD)
    with c3:
        kpi_card("REVENUE AT RISK", f"€{revenue_at_risk:,.0f}", "Q1 · Cell B gross revenue", color=COLOR_BAD)
    with c4:
        kpi_card("RATING COVERAGE", f"{rating_cov:.1f}%", f"only {int(accepted.loc[accepted['is_finished'], 'is_rated'].sum()):,} rated · bias gate", color=COLOR_WARN)

    # Second row — segment context
    st.write("")
    c1, c2, c3, c4 = st.columns(4)

    vip_subset = accepted[accepted["Is VIP Airline? (Y/N)"] == "Y"]
    nv_subset = accepted[accepted["Is VIP Airline? (Y/N)"] == "N"]
    vip_cell_b_rate = 100 * int(vip_subset["is_cell_b"].sum()) / len(vip_subset) if len(vip_subset) else 0
    nv_cell_b_rate = 100 * int(nv_subset["is_cell_b"].sum()) / len(nv_subset) if len(nv_subset) else 0

    # Zero-km share of Cell B
    cell_b_subset = accepted[accepted["is_cell_b"]]
    zero_km_share = 100 * int(cell_b_subset["is_zero_km"].sum()) / len(cell_b_subset) if len(cell_b_subset) else 0

    with c1:
        kpi_card("VIP CELL B RATE", f"{vip_cell_b_rate:.2f}%", "Partnership integrity")
    with c2:
        kpi_card("NON-VIP CELL B RATE", f"{nv_cell_b_rate:.2f}%", "Direct-customer reliability")
    with c3:
        kpi_card("ZERO-KM CELL B", f"{zero_km_share:.0f}%", "Customer-never-appeared signature", color=COLOR_WARN)
    with c4:
        n_lsps = accepted["LSP Name"].nunique()
        kpi_card("ACTIVE LSPs", f"{n_lsps:,}", f"{n_rejected:,} rejected bookings", color=COLOR_MUTED)

    # The Cell B split visualization
    section_header("Cell B — two operational stories", "Look closer: same label, different mechanisms, different levers")

    if len(cell_b_subset) > 0:
        c1, c2 = st.columns([1, 2])
        with c1:
            zero_km_n = int(cell_b_subset["is_zero_km"].sum())
            some_km_n = len(cell_b_subset) - zero_km_n
            st.markdown(f"""
                <div style="padding: 12px 16px; background: #FEF3C7; border-left: 4px solid {COLOR_WARN}; border-radius: 4px;">
                    <div style="font-size: 11px; color: {COLOR_WARN}; letter-spacing: 1px; font-weight: 600;">ZERO-KM CELL B</div>
                    <div style="font-size: 26px; font-weight: 700; color: {COLOR_INK}; font-feature-settings: 'tnum';">{zero_km_n:,} <span style="color:{COLOR_MUTED}; font-size: 16px; font-weight: 400;">({100*zero_km_n/len(cell_b_subset):.0f}%)</span></div>
                    <div style="font-size: 12px; color: {COLOR_INK}; margin-top: 4px;">Chauffeur waited, customer never appeared. <strong>Levers land cleanly here.</strong></div>
                </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.markdown(f"""
                <div style="padding: 12px 16px; background: {COLOR_PANEL}; border-left: 4px solid {COLOR_MUTED}; border-radius: 4px;">
                    <div style="font-size: 11px; color: {COLOR_MUTED}; letter-spacing: 1px; font-weight: 600;">SOME-KM CELL B</div>
                    <div style="font-size: 26px; font-weight: 700; color: {COLOR_INK}; font-feature-settings: 'tnum';">{some_km_n:,} <span style="color:{COLOR_MUTED}; font-size: 16px; font-weight: 400;">({100*some_km_n/len(cell_b_subset):.0f}%)</span></div>
                    <div style="font-size: 12px; color: {COLOR_INK}; margin-top: 4px;">Distance recorded but trip didn't complete. <strong>Needs qualitative work.</strong></div>
                </div>
            """, unsafe_allow_html=True)

        with c2:
            # Bar chart of Cell B rate by city
            city_cb = (
                accepted.groupby("Ride Bd")
                .agg(rides=("is_cell_b", "size"), cb=("is_cell_b", "sum"))
                .reset_index()
            )
            city_cb["cell_b_rate"] = 100 * city_cb["cb"] / city_cb["rides"]
            city_cb = city_cb[city_cb["rides"] >= 50].sort_values("cell_b_rate", ascending=True)

            fig = go.Figure()
            colors = [COLOR_BAD if r > 5 else (COLOR_WARN if r > 2 else COLOR_GOOD) for r in city_cb["cell_b_rate"]]
            fig.add_bar(
                y=city_cb["Ride Bd"],
                x=city_cb["cell_b_rate"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}%" for v in city_cb["cell_b_rate"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Cell B rate: %{x:.2f}%<br>Rides: %{customdata:,}<extra></extra>",
                customdata=city_cb["rides"],
            )
            fig.update_layout(
                title=dict(text="<b>Cell B rate by city</b>", font=dict(size=14, color=COLOR_INK)),
                height=380,
                margin=dict(l=80, r=20, t=40, b=20),
                xaxis=dict(title="Cell B rate (%)", showgrid=True, gridcolor="#E5E7EB"),
                yaxis=dict(title=""),
                plot_bgcolor=COLOR_BG,
                paper_bgcolor=COLOR_BG,
                font=dict(family="system-ui, sans-serif", color=COLOR_INK),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Cell B events in current filter selection.")

# ============================================================
# TAB: QUALITY PILLARS
# ============================================================

with tab_pillars:
    section_header("Three pillars", '"Reclaim your time" — what customers no longer have to think about')

    p1, p2, p3 = st.columns(3)

    # SAFETY pillar
    with p1:
        st.markdown(f"""
            <div style="padding: 16px; background: {COLOR_PANEL}; border-top: 4px solid {COLOR_BAD}; border-radius: 4px; min-height: 140px;">
                <div style="font-size: 11px; color: {COLOR_BAD}; letter-spacing: 2px; font-weight: 700;">FOUNDATIONAL</div>
                <div style="font-size: 22px; font-weight: 700; color: {COLOR_INK}; margin-top: 4px;">Safety</div>
                <div style="font-size: 12px; color: {COLOR_MUTED}; margin-top: 8px; font-style: italic;">"I felt safe in the car. I didn't worry."</div>
                <div style="font-size: 12px; color: {COLOR_INK}; margin-top: 10px;">Customer doesn't have to think about whether the car is safe.</div>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown("**Measurable today:**")
        st.caption("Most safety signals live outside this dataset. Operating telemetry placeholders below; legal-grade safety measurement waits on Q6 shift records.")

        cluster_size = (
            accepted.groupby("Driver First Last Name")
            .agg(rides=("is_cell_b", "size"), cb=("is_cell_b", "sum"))
            .reset_index()
        )
        cluster_size = cluster_size[cluster_size["rides"] >= 30]
        cluster_size["cb_rate"] = 100 * cluster_size["cb"] / cluster_size["rides"]
        anomalous = len(cluster_size[cluster_size["cb_rate"] >= 20])

        c1, c2 = st.columns(2)
        with c1:
            kpi_card("ANOMALOUS CLUSTER", f"{anomalous}", "Chauffeurs at ≥20% Cell B (≥30 rides)", color=COLOR_BAD)
        with c2:
            kpi_card("VEHICLE MODEL VARIETY", f"{accepted['Vehicle Model'].nunique()}", "Distinct vehicle models")

    # RELIABILITY pillar
    with p2:
        st.markdown(f"""
            <div style="padding: 16px; background: {COLOR_PANEL}; border-top: 4px solid {COLOR_WARN}; border-radius: 4px; min-height: 140px;">
                <div style="font-size: 11px; color: {COLOR_WARN}; letter-spacing: 2px; font-weight: 700;">MEASURABLE</div>
                <div style="font-size: 22px; font-weight: 700; color: {COLOR_INK}; margin-top: 4px;">Reliability</div>
                <div style="font-size: 12px; color: {COLOR_MUTED}; margin-top: 8px; font-style: italic;">"It went exactly as I expected."</div>
                <div style="font-size: 12px; color: {COLOR_INK}; margin-top: 10px;">Customer doesn't have to think about whether the ride happens.</div>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown("**Measurable today:**")
        st.caption("Punctuality, no-show split, completion rate — all computable directly on Q1 fields.")

        c1, c2 = st.columns(2)
        with c1:
            kpi_card("CELL B RATE", f"{cell_b_rate:.2f}%", "Operational no-show", color=COLOR_BAD if cell_b_rate > 3 else COLOR_WARN)
        with c2:
            cell_c_rate = 100 * n_cell_c / n_total_accepted if n_total_accepted else 0
            kpi_card("CELL C RATE", f"{cell_c_rate:.2f}%", "True chauffeur no-show", color=COLOR_BAD if cell_c_rate > 0.5 else COLOR_GOOD)

        c3, c4 = st.columns(2)
        with c3:
            airport_pickup_df = accepted[accepted["Transfer type"] == "airport pickup"]
            ap_rate = 100 * int(airport_pickup_df["is_cell_b"].sum()) / len(airport_pickup_df) if len(airport_pickup_df) else 0
            kpi_card("AIRPORT-PICKUP CELL B", f"{ap_rate:.2f}%", f"of {len(airport_pickup_df):,} airport pickups", color=COLOR_BAD if ap_rate > 3 else COLOR_WARN)
        with c4:
            kpi_card("COMPLETION RATE", f"{finished_rate:.1f}%", "All accepted rides", color=COLOR_GOOD)

    # PEACE OF MIND pillar
    with p3:
        st.markdown(f"""
            <div style="padding: 16px; background: {COLOR_PANEL}; border-top: 4px solid {COLOR_GOOD}; border-radius: 4px; min-height: 140px;">
                <div style="font-size: 11px; color: {COLOR_GOOD}; letter-spacing: 2px; font-weight: 700;">CUMULATIVE</div>
                <div style="font-size: 22px; font-weight: 700; color: {COLOR_INK}; margin-top: 4px;">Peace of mind</div>
                <div style="font-size: 12px; color: {COLOR_MUTED}; margin-top: 8px; font-style: italic;">"I don't have to plan for it going wrong."</div>
                <div style="font-size: 12px; color: {COLOR_INK}; margin-top: 10px;">Customer doesn't have to think about it at all.</div>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown("**Measurable today:**")
        st.caption("Ratings (gated by 17% coverage bias), low-rating tail share.")

        finished_subset = accepted[accepted["is_finished"]]
        rated_subset = finished_subset[finished_subset["is_rated"]]
        c1, c2 = st.columns(2)
        with c1:
            kpi_card("RATING COVERAGE", f"{rating_cov:.1f}%", "Coverage bias gate", color=COLOR_WARN)
        with c2:
            if len(rated_subset) > 0:
                avg_drv = rated_subset["Avg. Driver Rating"].mean()
                kpi_card("AVG CHAUFFEUR RATING", f"{avg_drv:.2f}", f"on {len(rated_subset):,} rated rides", color=COLOR_GOOD)
            else:
                kpi_card("AVG CHAUFFEUR RATING", "n/a", "No rated rides in filter")

        c3, c4 = st.columns(2)
        with c3:
            vip_rate_cov = 100 * int(finished_subset[finished_subset["Is VIP Airline? (Y/N)"] == "Y"]["is_rated"].sum()) / max(1, len(finished_subset[finished_subset["Is VIP Airline? (Y/N)"] == "Y"]))
            kpi_card("VIP RATING COVERAGE", f"{vip_rate_cov:.2f}%", "Structural bias", color=COLOR_BAD)
        with c4:
            if len(rated_subset) > 0:
                low_tail = 100 * (rated_subset["Avg. Driver Rating"] < 5).sum() / len(rated_subset)
                kpi_card("LOW-RATING TAIL", f"{low_tail:.1f}%", "Rated < 5 stars")
            else:
                kpi_card("LOW-RATING TAIL", "n/a", "No rated rides")

    # Rating coverage breakdown
    section_header("Rating bias diagnostic", "Coverage gaps reveal where the signal is structurally thin")

    rated_by_class = (
        accepted[accepted["is_finished"]]
        .groupby("Car Class")
        .agg(rides=("is_rated", "size"), rated=("is_rated", "sum"))
        .reset_index()
    )
    rated_by_class["coverage_pct"] = 100 * rated_by_class["rated"] / rated_by_class["rides"]

    rated_by_vip = (
        accepted[accepted["is_finished"]]
        .groupby("Is VIP Airline? (Y/N)")
        .agg(rides=("is_rated", "size"), rated=("is_rated", "sum"))
        .reset_index()
    )
    rated_by_vip["coverage_pct"] = 100 * rated_by_vip["rated"] / rated_by_vip["rides"]
    rated_by_vip["segment"] = rated_by_vip["Is VIP Airline? (Y/N)"].map({"Y": "VIP", "N": "Non-VIP"})

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(
            x=rated_by_class["Car Class"],
            y=rated_by_class["coverage_pct"],
            marker_color=[COLOR_BAD if c < 20 else COLOR_GOOD for c in rated_by_class["coverage_pct"]],
            text=[f"{v:.1f}%" for v in rated_by_class["coverage_pct"]],
            textposition="outside",
        )
        fig.update_layout(
            title="<b>Rating coverage by car class</b>",
            height=320,
            margin=dict(l=40, r=20, t=40, b=40),
            yaxis=dict(title="Coverage (%)", range=[0, max(40, rated_by_class["coverage_pct"].max() * 1.2)]),
            plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
            font=dict(family="system-ui", color=COLOR_INK),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_bar(
            x=rated_by_vip["segment"],
            y=rated_by_vip["coverage_pct"],
            marker_color=[COLOR_BAD, COLOR_GOOD] if rated_by_vip["coverage_pct"].iloc[0] < rated_by_vip["coverage_pct"].iloc[1] else [COLOR_GOOD, COLOR_BAD],
            text=[f"{v:.2f}%" for v in rated_by_vip["coverage_pct"]],
            textposition="outside",
        )
        fig.update_layout(
            title="<b>Rating coverage by segment</b>",
            height=320,
            margin=dict(l=40, r=20, t=40, b=40),
            yaxis=dict(title="Coverage (%)", range=[0, max(40, rated_by_vip["coverage_pct"].max() * 1.2)]),
            plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
            font=dict(family="system-ui", color=COLOR_INK),
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB: LSP SCORECARD
# ============================================================

with tab_scorecard:
    section_header("LSP Scorecard", "A · Standard · B · Monitor · C · Improve · D · Review")

    min_rides_input = st.slider("Minimum rides to include (filters out small LSPs)", min_value=50, max_value=1000, value=100, step=50)
    scorecard = build_lsp_scorecard(accepted, min_rides=min_rides_input)

    if len(scorecard) == 0:
        st.info("No LSPs meet the minimum-rides threshold in current filter selection.")
    else:
        # Tier summary
        c1, c2, c3, c4 = st.columns(4)
        for col, tier in zip([c1, c2, c3, c4], ["A", "B", "C", "D"]):
            count = (scorecard["tier"] == tier).sum()
            ride_share = 100 * scorecard.loc[scorecard["tier"] == tier, "rides"].sum() / scorecard["rides"].sum() if scorecard["rides"].sum() else 0
            tier_label = {
                "A": "TIER A · STANDARD",
                "B": "TIER B · MONITORING",
                "C": "TIER C · IMPROVEMENT",
                "D": "TIER D · REVIEW",
            }[tier]
            with col:
                kpi_card(tier_label, f"{count}", f"{ride_share:.1f}% of volume", color=TIER_COLORS[tier])

        section_header("Scorecard table", "Sort by any column · click a row to drill in")

        # Tier filter chips
        tier_filter = st.multiselect("Filter by tier", options=["A", "B", "C", "D"], default=["A", "B", "C", "D"])
        scorecard_view = scorecard[scorecard["tier"].isin(tier_filter)].copy()

        # Display table — formatted
        display = scorecard_view[[
            "LSP Name", "tier", "rides", "volume_share_pct", "vip_share_pct",
            "cell_b_rate_pct", "cell_c_rate_pct", "finished_rate_pct",
            "contribution_per_ride", "rating_coverage_pct", "avg_driver_rating",
        ]].rename(columns={
            "LSP Name": "LSP",
            "tier": "Tier",
            "rides": "Rides",
            "volume_share_pct": "Vol %",
            "vip_share_pct": "VIP %",
            "cell_b_rate_pct": "Cell B %",
            "cell_c_rate_pct": "Cell C %",
            "finished_rate_pct": "Finished %",
            "contribution_per_ride": "€/ride",
            "rating_coverage_pct": "Rated %",
            "avg_driver_rating": "Avg ★",
        })

        st.dataframe(
            display.style
                .format({
                    "Rides": "{:,.0f}",
                    "Vol %": "{:.1f}%",
                    "VIP %": "{:.0f}%",
                    "Cell B %": "{:.2f}%",
                    "Cell C %": "{:.2f}%",
                    "Finished %": "{:.1f}%",
                    "€/ride": "€{:.2f}",
                    "Rated %": "{:.1f}%",
                    "Avg ★": "{:.2f}",
                })
                .map(lambda v: value_gradient(v, 0, 20, "red"), subset=["Cell B %"])
                .map(lambda v: value_gradient(v, -10, 50, "green"), subset=["€/ride"])
                .map(lambda v: value_gradient(v, 0, 30, "blue"), subset=["Rated %"]),
            use_container_width=True,
            hide_index=True,
        )

        # Visual tier distribution
        section_header("Scorecard visual", "Cell B rate vs volume — bubble = revenue at risk")

        scorecard_view["revenue_at_risk"] = scorecard_view["cell_b_count"] * scorecard_view["avg_revenue"]
        fig = px.scatter(
            scorecard_view,
            x="volume_share_pct",
            y="cell_b_rate_pct",
            size="revenue_at_risk",
            color="tier",
            hover_name="LSP Name",
            color_discrete_map=TIER_COLORS,
            category_orders={"tier": ["A", "B", "C", "D"]},
            size_max=50,
            labels={
                "volume_share_pct": "Volume share of platform (%)",
                "cell_b_rate_pct": "Cell B rate (%)",
                "tier": "Tier",
            },
        )
        # Tier threshold lines
        fig.add_hline(y=TIER_B_THRESHOLD, line_dash="dot", line_color=COLOR_MUTED, opacity=0.4)
        fig.add_hline(y=TIER_C_THRESHOLD, line_dash="dot", line_color=COLOR_WARN, opacity=0.5)
        fig.add_hline(y=TIER_D_THRESHOLD, line_dash="dot", line_color=COLOR_BAD, opacity=0.6)

        fig.update_layout(
            height=480,
            plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
            font=dict(family="system-ui", color=COLOR_INK),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Drill-down
        section_header("LSP drill-down", "Pick any LSP to see its detailed scorecard")
        selected_lsp = st.selectbox(
            "Select LSP",
            options=scorecard_view["LSP Name"].tolist(),
        )

        if selected_lsp:
            lsp_row = scorecard_view[scorecard_view["LSP Name"] == selected_lsp].iloc[0]
            lsp_rides = accepted[accepted["LSP Name"] == selected_lsp]

            # Top metrics for selected LSP
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                tier_html = tier_pill(lsp_row['tier'])
                st.markdown(
                    f"""
                    <div style="background: {COLOR_BG}; border: 1px solid #E5E7EB; border-left: 4px solid {TIER_COLORS.get(lsp_row['tier'], COLOR_MUTED)}; padding: 14px 18px; border-radius: 4px;">
                        <div style="color: {COLOR_MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;">{selected_lsp[:30]}</div>
                        <div style="margin-top: 10px;">{tier_html}</div>
                        <div style="color: {COLOR_MUTED}; font-size: 12px; margin-top: 8px;">{int(lsp_row['rides']):,} rides · {lsp_row['volume_share_pct']:.1f}% of platform</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                kpi_card("CELL B RATE", f"{lsp_row['cell_b_rate_pct']:.2f}%", f"{int(lsp_row['cell_b_count']):,} events", color=COLOR_BAD if lsp_row['cell_b_rate_pct'] > 3 else COLOR_WARN)
            with c3:
                kpi_card("VIP SHARE", f"{lsp_row['vip_share_pct']:.0f}%", "Of this LSP's rides")
            with c4:
                kpi_card("€/RIDE", f"€{lsp_row['contribution_per_ride']:.2f}", "Avg contribution")

            # Airport-pickup specific failure rate
            ap_cb, ap_total, ap_rate = airport_pickup_failure_rate(accepted, selected_lsp)
            if ap_total > 0:
                st.write("")
                color = COLOR_BAD if ap_rate > 10 else (COLOR_WARN if ap_rate > 3 else COLOR_GOOD)
                st.markdown(
                    f"""
                    <div style="padding: 10px 14px; background: {COLOR_PANEL}; border-left: 4px solid {color}; border-radius: 3px; font-size: 13px;">
                        <strong>Airport-pickup specific Cell B:</strong>
                        <span style="color: {color}; font-weight: 700;">{ap_rate:.2f}%</span>
                        ({ap_cb:,} of {ap_total:,} airport pickups · peer median ≈ 2–3%)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Distribution by city
            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                city_dist = lsp_rides.groupby("Ride Bd").size().reset_index(name="rides").sort_values("rides", ascending=True).tail(10)
                fig = go.Figure()
                fig.add_bar(
                    y=city_dist["Ride Bd"],
                    x=city_dist["rides"],
                    orientation="h",
                    marker_color=COLOR_INK,
                    text=[f"{v:,}" for v in city_dist["rides"]],
                    textposition="outside",
                )
                fig.update_layout(
                    title="<b>Volume by city</b>",
                    height=300,
                    margin=dict(l=80, r=30, t=40, b=20),
                    plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                    font=dict(family="system-ui", color=COLOR_INK),
                    xaxis=dict(title="Rides"),
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                # Class mix
                class_dist = lsp_rides.groupby("Car Class").size().reset_index(name="rides")
                fig = px.pie(
                    class_dist,
                    values="rides",
                    names="Car Class",
                    color_discrete_sequence=[COLOR_INK, COLOR_GOOD, COLOR_WARN, COLOR_BAD],
                    hole=0.5,
                )
                fig.update_traces(textposition="outside", textinfo="label+percent")
                fig.update_layout(
                    title="<b>Class mix</b>",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                    font=dict(family="system-ui", color=COLOR_INK),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB: OPERATIONAL TELEMETRY
# ============================================================

with tab_telemetry:
    section_header(
        "Operational telemetry",
        "Measure customer experience directly — without asking the customer anything"
    )

    st.markdown(
        f"""
        <div style="background: {COLOR_PANEL}; border-left: 4px solid {COLOR_INK}; padding: 10px 14px; border-radius: 3px; font-size: 12px; color: {COLOR_INK};">
            Premium service brands measure quality through operational signals the system already generates, not by extracting time from customers. The four signals below proxy customer experience directly from existing timestamp and route fields — no new data required.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Signal 1: Pickup arrival delta ──────────────────────────────
    section_header(
        "1 · Pickup arrival delta",
        "Chauffeur arrival vs booked pickup time · negative = early · the cleanest reliability metric"
    )

    pd_data = accepted[accepted["pickup_delta_min"].notna() & accepted["pickup_delta_min"].between(-60, 60)].copy()

    if len(pd_data) > 0:
        c1, c2, c3, c4 = st.columns(4)
        median_delta = pd_data["pickup_delta_min"].median()
        on_time_pct = 100 * pd_data["pickup_delta_min"].between(-1, 5).mean()
        late_pct = 100 * (pd_data["pickup_delta_min"] > 5).mean()
        very_late_pct = 100 * (pd_data["pickup_delta_min"] > 15).mean()

        with c1:
            kpi_card("MEDIAN ARRIVAL", f"{median_delta:.0f} min", "Early = premium service behavior", color=COLOR_GOOD if median_delta < 0 else COLOR_WARN)
        with c2:
            kpi_card("ON-TIME (-1 to +5 min)", f"{on_time_pct:.1f}%", "Window of acceptable variance", color=COLOR_GOOD)
        with c3:
            kpi_card("LATE > 5 MIN", f"{late_pct:.1f}%", "Operational reliability tail", color=COLOR_WARN if late_pct < 15 else COLOR_BAD)
        with c4:
            kpi_card("VERY LATE > 15 MIN", f"{very_late_pct:.1f}%", "Customer-visible failure tier", color=COLOR_BAD)

        # Distribution histogram
        c1, c2 = st.columns([2, 3])
        with c1:
            fig = go.Figure()
            fig.add_histogram(
                x=pd_data["pickup_delta_min"],
                nbinsx=40,
                marker_color=COLOR_INK,
                marker_line_color=COLOR_BG,
                marker_line_width=1,
            )
            fig.add_vline(x=0, line_dash="solid", line_color=COLOR_BAD, opacity=0.6)
            fig.add_vline(x=5, line_dash="dash", line_color=COLOR_WARN, opacity=0.6)
            fig.update_layout(
                title="<b>Distribution (minutes vs booked pickup)</b>",
                height=320,
                margin=dict(l=40, r=20, t=40, b=40),
                xaxis=dict(title="Minutes (negative = early, positive = late)", showgrid=True, gridcolor="#E5E7EB"),
                yaxis=dict(title="Rides"),
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                font=dict(family="system-ui", color=COLOR_INK),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Per-city breakdown
            by_city = pd_data.groupby("Ride Bd").agg(
                rides=("pickup_delta_min", "size"),
                median_delta=("pickup_delta_min", "median"),
                late_pct=("pickup_delta_min", lambda s: 100 * (s > 5).mean()),
            ).reset_index()
            by_city = by_city[by_city["rides"] >= 50].sort_values("late_pct", ascending=True)

            fig = go.Figure()
            colors = [COLOR_BAD if r > 10 else (COLOR_WARN if r > 5 else COLOR_GOOD) for r in by_city["late_pct"]]
            fig.add_bar(
                y=by_city["Ride Bd"],
                x=by_city["late_pct"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}%" for v in by_city["late_pct"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Late >5 min: %{x:.1f}%<br>Rides: %{customdata:,}<extra></extra>",
                customdata=by_city["rides"],
            )
            fig.update_layout(
                title="<b>Share of rides arriving >5 min late, by city</b>",
                height=320,
                margin=dict(l=80, r=30, t=40, b=20),
                xaxis=dict(title="% rides late >5 min", showgrid=True, gridcolor="#E5E7EB"),
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                font=dict(family="system-ui", color=COLOR_INK),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timestamp data available for pickup-delta analysis in current filter.")

    # ── Signal 2: Booking-to-acceptance latency ─────────────────────
    section_header(
        "2 · Booking-to-acceptance latency",
        "Time between booking and an LSP claiming it · supply-tightness indicator"
    )

    bl_data = accepted[accepted["ba_latency_sec"].notna() & accepted["ba_latency_sec"].between(0, 86400)].copy()
    if len(bl_data) > 0:
        c1, c2, c3, c4 = st.columns(4)
        median_sec = bl_data["ba_latency_sec"].median()
        instant_pct = 100 * (bl_data["ba_latency_sec"] < 60).mean()
        fast_pct = 100 * (bl_data["ba_latency_sec"] < 300).mean()
        slow_pct = 100 * (bl_data["ba_latency_sec"] > 3600).mean()

        with c1:
            kpi_card("MEDIAN LATENCY", f"{median_sec/60:.1f} min", f"{median_sec:.0f} seconds")
        with c2:
            kpi_card("INSTANT (<60s)", f"{instant_pct:.1f}%", "Pre-assigned or hot supply", color=COLOR_GOOD)
        with c3:
            kpi_card("FAST (<5 min)", f"{fast_pct:.1f}%", "Healthy supply response")
        with c4:
            kpi_card("SLOW (>1 hour)", f"{slow_pct:.1f}%", "Supply-side latency tail", color=COLOR_BAD if slow_pct > 20 else COLOR_WARN)

        # Latency categories per city
        bl_data["category"] = pd.cut(
            bl_data["ba_latency_sec"],
            bins=[0, 60, 300, 1800, 3600, 86400],
            labels=["<1 min", "1-5 min", "5-30 min", "30-60 min", ">1 hour"],
            include_lowest=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            cat_share = bl_data["category"].value_counts(normalize=True).sort_index() * 100
            fig = go.Figure()
            colors = [COLOR_GOOD, "#0E7490", COLOR_MUTED, COLOR_WARN, COLOR_BAD]
            fig.add_bar(
                x=cat_share.index.astype(str),
                y=cat_share.values,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in cat_share.values],
                textposition="outside",
            )
            fig.update_layout(
                title="<b>Acceptance-latency distribution</b>",
                height=320,
                margin=dict(l=40, r=20, t=40, b=40),
                xaxis=dict(title=""),
                yaxis=dict(title="Share of bookings (%)"),
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                font=dict(family="system-ui", color=COLOR_INK),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Per-LSP latency (top 10 by volume)
            lsp_latency = bl_data.groupby("LSP Name").agg(
                rides=("ba_latency_sec", "size"),
                median_sec=("ba_latency_sec", "median"),
                slow_share=("ba_latency_sec", lambda s: 100 * (s > 3600).mean()),
            ).reset_index()
            lsp_latency = lsp_latency[lsp_latency["rides"] >= 500].sort_values("rides", ascending=False).head(10)
            lsp_latency = lsp_latency.sort_values("median_sec", ascending=True)
            lsp_latency["median_min"] = lsp_latency["median_sec"] / 60

            fig = go.Figure()
            fig.add_bar(
                y=lsp_latency["LSP Name"].str[:24],
                x=lsp_latency["median_min"],
                orientation="h",
                marker_color=[COLOR_BAD if v > 30 else (COLOR_WARN if v > 10 else COLOR_GOOD) for v in lsp_latency["median_min"]],
                text=[f"{v:.0f} min" for v in lsp_latency["median_min"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Median latency: %{x:.1f} min<br>Rides: %{customdata:,}<extra></extra>",
                customdata=lsp_latency["rides"],
            )
            fig.update_layout(
                title="<b>Median acceptance latency, top 10 LSPs</b>",
                height=320,
                margin=dict(l=160, r=30, t=40, b=20),
                xaxis=dict(title="Minutes", showgrid=True, gridcolor="#E5E7EB"),
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                font=dict(family="system-ui", color=COLOR_INK),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Signal 3: Lead time at booking ──────────────────────────────
    section_header(
        "3 · Lead-time distribution",
        "How far in advance customers book · short-lead is operationally hardest · direct link to Goal 2"
    )

    lt_data = accepted[accepted["lead_time_hr"].notna() & accepted["lead_time_hr"].between(0, 168)].copy()
    if len(lt_data) > 0:
        c1, c2, c3, c4 = st.columns(4)
        median_lt = lt_data["lead_time_hr"].median()
        short_pct = 100 * (lt_data["lead_time_hr"] < 2).mean()
        same_day = 100 * (lt_data["lead_time_hr"] < 24).mean()
        advance = 100 * (lt_data["lead_time_hr"] >= 24).mean()

        with c1:
            kpi_card("MEDIAN LEAD TIME", f"{median_lt:.0f} hours", f"{median_lt/24:.1f} days")
        with c2:
            kpi_card("SHORT LEAD (<2h)", f"{short_pct:.1f}%", "Goal 2's target segment", color=COLOR_BAD)
        with c3:
            kpi_card("SAME-DAY (<24h)", f"{same_day:.0f}%", "Within-day booking")
        with c4:
            kpi_card("ADVANCE (≥24h)", f"{advance:.0f}%", "Pre-planned bookings", color=COLOR_GOOD)

        # Short-lead acceptance by city — directly relevant to Goal 2
        short_lead = lt_data[lt_data["lead_time_hr"] < 2].copy()
        if len(short_lead) > 0:
            short_by_city = short_lead.groupby("Ride Bd").agg(
                short_lead_count=("lead_time_hr", "size"),
                finished=("is_finished", "sum"),
            ).reset_index()
            short_by_city["finish_rate"] = 100 * short_by_city["finished"] / short_by_city["short_lead_count"]
            short_by_city = short_by_city[short_by_city["short_lead_count"] >= 20].sort_values("finish_rate", ascending=True)

            if len(short_by_city) > 0:
                fig = go.Figure()
                colors = [COLOR_BAD if v < 90 else (COLOR_WARN if v < 95 else COLOR_GOOD) for v in short_by_city["finish_rate"]]
                fig.add_bar(
                    y=short_by_city["Ride Bd"],
                    x=short_by_city["finish_rate"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:.1f}%" for v in short_by_city["finish_rate"]],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Short-lead finish rate: %{x:.1f}%<br>Short-lead rides: %{customdata:,}<extra></extra>",
                    customdata=short_by_city["short_lead_count"],
                )
                fig.add_vline(x=98, line_dash="dash", line_color=COLOR_GOOD, opacity=0.6, annotation_text="Goal 2 target: 98%")
                fig.update_layout(
                    title="<b>Short-lead (<2h) completion rate by city · Goal 2's KPI target</b>",
                    height=400,
                    margin=dict(l=80, r=30, t=50, b=20),
                    xaxis=dict(title="Completion rate (%)", showgrid=True, gridcolor="#E5E7EB", range=[80, 102]),
                    plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                    font=dict(family="system-ui", color=COLOR_INK),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Signal 4: Trip speed (route-type proxy) ─────────────────────
    section_header(
        "4 · Trip speed · route-type signal",
        "Route Distance KM ÷ trip duration · reveals airport long-haul vs city short-haul mix"
    )

    sp_data = accepted[
        accepted["speed_kmh"].notna() & accepted["speed_kmh"].between(1, 200) & (accepted["trip_hr"] > 0.1)
    ].copy()
    if len(sp_data) > 0:
        c1, c2 = st.columns([2, 3])
        with c1:
            fig = go.Figure()
            fig.add_histogram(
                x=sp_data["speed_kmh"],
                nbinsx=40,
                marker_color=COLOR_INK,
                marker_line_color=COLOR_BG,
                marker_line_width=1,
            )
            fig.update_layout(
                title="<b>Trip-speed distribution (km/h)</b>",
                height=320,
                margin=dict(l=40, r=20, t=40, b=40),
                xaxis=dict(title="km/h", showgrid=True, gridcolor="#E5E7EB"),
                yaxis=dict(title="Rides"),
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                font=dict(family="system-ui", color=COLOR_INK),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            speed_by_city = sp_data.groupby("Ride Bd").agg(
                rides=("speed_kmh", "size"),
                median_speed=("speed_kmh", "median"),
                median_km=("Route Distance KM", "median"),
            ).reset_index()
            speed_by_city = speed_by_city[speed_by_city["rides"] >= 50].sort_values("median_speed", ascending=True)

            fig = go.Figure()
            fig.add_bar(
                y=speed_by_city["Ride Bd"],
                x=speed_by_city["median_speed"],
                orientation="h",
                marker_color=[
                    COLOR_BAD if v < 15 else ("#0E7490" if v < 25 else COLOR_GOOD)
                    for v in speed_by_city["median_speed"]
                ],
                text=[f"{v:.0f} km/h · median {k:.0f} km" for v, k in zip(speed_by_city["median_speed"], speed_by_city["median_km"])],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Median speed: %{x:.1f} km/h<br>Median trip: %{customdata:.0f} km<extra></extra>",
                customdata=speed_by_city["median_km"],
            )
            fig.update_layout(
                title="<b>Median trip speed by city · low speed = city-center, high = airport routes</b>",
                height=320,
                margin=dict(l=80, r=180, t=40, b=20),
                xaxis=dict(title="km/h", showgrid=True, gridcolor="#E5E7EB"),
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                font=dict(family="system-ui", color=COLOR_INK),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Closing framing
    st.markdown(
        f"""
        <div style="background: {COLOR_PANEL}; border-left: 4px solid {COLOR_GOOD}; padding: 14px 18px; border-radius: 3px; margin-top: 20px;">
            <div style="font-size: 12px; font-weight: 600; color: {COLOR_GOOD}; letter-spacing: 1px;">WHY THIS REPLACES RATING COVERAGE AS THE PRIMARY QUALITY MEASUREMENT</div>
            <div style="font-size: 13px; color: {COLOR_INK}; margin-top: 8px;">
                Premium customers value time. Asking them to rate the service costs them time — contradicting the brand promise.
                Operational telemetry measures customer experience directly from signals the operating system already generates,
                without taxing customer attention. The 17% rating coverage stays useful but smaller; airline-side NPS integration
                (Q6 wishlist) closes the VIP-experience gap from the partnership side, not the customer-time side.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# TAB: CHAUFFEUR AUDIT
# ============================================================

with tab_chauffeurs:
    section_header("Chauffeur cluster audit", "Individuals running ≥20% Cell B rate at ≥30 rides — Q1's anomalous cluster")
    st.caption("Caveat: 13.5% of multi-trip chauffeur-days show the same name across multiple vehicle models. The names below may include shared LSP-account labels rather than identifiable individuals. Verify identity before deactivation decisions.")

    chauffeur_stats = (
        accepted.groupby(["Driver First Last Name", "LSP Name"])
        .agg(
            rides=("is_cell_b", "size"),
            cb=("is_cell_b", "sum"),
            revenue=("Avg. Gross Revenue", "sum"),
            avg_rating=("Avg. Driver Rating", "mean"),
        )
        .reset_index()
    )
    chauffeur_stats["cb_rate"] = 100 * chauffeur_stats["cb"] / chauffeur_stats["rides"]

    min_rides_chauf = st.slider("Minimum rides per chauffeur", min_value=10, max_value=100, value=30, step=10)
    cb_threshold = st.slider("Cell B rate threshold (%)", min_value=10.0, max_value=50.0, value=20.0, step=5.0)

    flagged = chauffeur_stats[
        (chauffeur_stats["rides"] >= min_rides_chauf) &
        (chauffeur_stats["cb_rate"] >= cb_threshold)
    ].sort_values("cb_rate", ascending=False)

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("FLAGGED CHAUFFEURS", f"{len(flagged)}", f"≥{int(min_rides_chauf)} rides, ≥{cb_threshold:.0f}% Cell B", color=COLOR_BAD)
    with c2:
        total_failures = int(flagged["cb"].sum())
        kpi_card("TOTAL FAILURES", f"{total_failures:,}", "Cell B events from flagged group", color=COLOR_BAD)
    with c3:
        cluster_pct = 100 * total_failures / max(1, int(accepted["is_cell_b"].sum()))
        kpi_card("SHARE OF ALL CELL B", f"{cluster_pct:.0f}%", "Concentrated in this cluster", color=COLOR_BAD)

    if len(flagged) > 0:
        flagged_display = flagged.rename(columns={
            "Driver First Last Name": "Chauffeur Name",
            "LSP Name": "LSP",
            "rides": "Rides",
            "cb": "Cell B events",
            "cb_rate": "Cell B %",
            "avg_rating": "Avg ★ (when rated)",
            "revenue": "Revenue (€)",
        })[["Chauffeur Name", "LSP", "Rides", "Cell B events", "Cell B %", "Avg ★ (when rated)", "Revenue (€)"]]

        st.dataframe(
            flagged_display.style.format({
                "Rides": "{:,.0f}",
                "Cell B events": "{:,.0f}",
                "Cell B %": "{:.1f}%",
                "Avg ★ (when rated)": "{:.2f}",
                "Revenue (€)": "€{:,.0f}",
            }).map(lambda v: value_gradient(v, 10, 80, "red"), subset=["Cell B %"]),
            use_container_width=True,
            hide_index=True,
        )

        section_header("The audit anomaly", "Worst-performing chauffeurs sometimes have perfect customer ratings — investigate before action")
        if (flagged["avg_rating"] >= 4.9).any():
            perfect_rated = flagged[flagged["avg_rating"] >= 4.9]
            st.warning(f"⚠️ {len(perfect_rated)} flagged chauffeurs maintain ratings ≥4.9 stars despite high failure rates. This pattern (high Cell B + high rating) suggests gaming, LSP misreporting, or shared-account labels rather than genuine quality.")
    else:
        st.success("No chauffeurs meet the current threshold criteria in this filter scope.")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    f"""
    <div style='color: {COLOR_MUTED}; font-size: 11px; padding: 12px 0; text-align: center;'>
        Blacklane Quality Dashboard · Built on Q1 2019 EMEA accepted + rejected tours data ·
        Currency-corrected (Stockholm SEK→EUR ÷11) · Tier thresholds: A &lt;3% · B 3-8% · C 8-15% · D ≥15% Cell B
    </div>
    """,
    unsafe_allow_html=True,
)
