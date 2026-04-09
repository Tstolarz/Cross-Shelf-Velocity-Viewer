import streamlit as st
import pandas as pd
from datetime import timedelta
from pathlib import Path

from utils.data_loading import load_mbon_data, load_doppio_single_layer
from utils.plotting import (create_timeseries_base, update_selection,
                            create_base_map, create_arrow_overlay, create_map_plot)
from utils.ui_components import apply_custom_css

st.set_page_config(
    page_title="Cross-Shelf Velocity Viewer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# ── Location configuration ─────────────────────────────────────────────────────
LOCATIONS = {
    "Ocean City, MA": {
        "hfr_path":           "2007_2024_10_10_MARACOOS_uv_MBON_timeseries_v5_corrected.nc",
        "doppio_surface_path": "doppio_timeseries_surface.nc",
        "doppio_bottom_path":  "doppio_timeseries_bottom.nc",
        "map_label":           "MBON3",
    },
    "Atlantic City, NJ": {
        "hfr_path":           "2007_2025_MARACOOS_uv_MOTZ_timeseries_CLEANED.nc",
        "doppio_surface_path": "DOPPIO_surface_MOTZ_timeseries.nc",
        "doppio_bottom_path":  "DOPPIO_bottom_MOTZ_timeseries.nc",
        "map_label":           "MOTZ",
    },
}
# ──────────────────────────────────────────────────────────────────────────────


def show_data_interface(df, base_map_bytes, data_label, freq_label="Monthly",
                        arrow_color='darkblue', ylim=None):
    """Shared interface for displaying timeseries data with scrubbing and map."""

    st.sidebar.subheader("Time Range Selection")
    full_range = (df['time'].min(), df['time'].max())

    date_range = st.sidebar.date_input(
        "Select date range",
        value=(full_range[0].date(), full_range[1].date()),
        min_value=full_range[0].date(),
        max_value=full_range[1].date(),
        key=f"date_range_{data_label.lower().replace(' ', '_')}"
    )

    if len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + timedelta(days=31)
        df_filtered = df[(df['time'] >= start_date) & (df['time'] < end_date)]
        time_range = (start_date, end_date)
        base_ts_fig = create_timeseries_base(df, time_range, freq_label, ylim)
    else:
        df_filtered = df
        time_range = None
        base_ts_fig = create_timeseries_base(df, None, freq_label, ylim)

    st.sidebar.subheader(f"{freq_label} Selection")

    if len(df_filtered) > 0:
        slider_key = f"slider_{data_label.lower().replace(' ', '_')}"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = 0

        st.session_state[slider_key] = max(0, min(st.session_state[slider_key], len(df_filtered) - 1))

        col_left, col_slider, col_right = st.sidebar.columns([1, 6, 1])

        with col_left:
            if st.button("◀", key=f"left_{data_label.lower().replace(' ', '_')}",
                         help="Previous time step", use_container_width=True):
                if st.session_state[slider_key] > 0:
                    st.session_state[slider_key] -= 1

        with col_right:
            if st.button("▶", key=f"right_{data_label.lower().replace(' ', '_')}",
                         help="Next time step", use_container_width=True):
                if st.session_state[slider_key] < len(df_filtered) - 1:
                    st.session_state[slider_key] += 1

        with col_slider:
            selected_idx_filtered = st.slider(
                f"Select {freq_label.lower()}",
                min_value=0,
                max_value=len(df_filtered) - 1,
                help=f"Drag to explore {freq_label.lower()} patterns",
                key=slider_key,
                label_visibility="collapsed"
            )

        selected_time = df_filtered.iloc[selected_idx_filtered]['time']
        selected_idx = df[df['time'] == selected_time].index[0]
        current_row = df.iloc[selected_idx]

        time_format = '%Y-%m' if freq_label == "Monthly" else '%Y-%m-%d'

        st.sidebar.info(f"""
        **Current Selection ({data_label}):**
        - **{current_row['time'].strftime(time_format)}**
        - Along: {current_row['v_along_mean']:.1f} ± {current_row['v_along_stderr']:.1f} cm/s
        - Cross: {current_row['u_cross_mean']:.1f} ± {current_row['u_cross_stderr']:.1f} cm/s
        - N obs: {current_row['magnitude_count']:.0f}
        """)

        with st.sidebar.expander("⏱️ Performance Metrics", expanded=False):
            if 'timing_data' in st.session_state and st.session_state.timing_data:
                for func_name, duration in st.session_state.timing_data.items():
                    color = "🟢" if duration < 50 else "🟡" if duration < 200 else "🔴"
                    st.text(f"{color} {func_name}: {duration:.1f}ms")

        col1, col2 = st.columns([2, 1])

        with col1:
            ts_fig = update_selection(base_ts_fig, current_row['time'])
            st.plotly_chart(ts_fig, use_container_width=True)

        with col2:
            st.markdown(f"#### {freq_label} Current Details")
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Along-shelf", f"{current_row['v_along_mean']:.1f} cm/s")
                st.metric("Cross-shelf", f"{current_row['u_cross_mean']:.1f} cm/s")
                st.metric("Observations", f"{current_row['magnitude_count']:.0f}")
            with col2b:
                st.metric("U (East)", f"{current_row['u_mean']:.1f} cm/s")
                st.metric("V (North)", f"{current_row['v_mean']:.1f} cm/s")

        col_map1, col_map2, col_map3 = st.columns([1, 2, 1])
        with col_map2:
            map_img = create_map_plot(df, selected_idx, base_map_bytes, freq_label, arrow_color)
            st.image(map_img, use_container_width=True)

    else:
        st.warning("No data available for the selected time range.")


# ============================================================================
# DATA LOADING HELPERS
# ============================================================================

def load_hfr(loc, resample_freq):
    path = loc['hfr_path']
    if not Path(path).exists():
        st.error(f"HFR file not found: {path}")
        st.stop()
    return load_mbon_data(path, resample_freq=resample_freq)


def load_doppio(loc, layer, resample_freq='W'):
    key = f"doppio_{layer}_path"
    path = loc[key]
    if not Path(path).exists():
        st.error(f"DOPPIO {layer} file not found: {path}")
        st.stop()
    return load_doppio_single_layer(path, layer, resample_freq=resample_freq)


def make_base_map(df, loc):
    first_row = df.iloc[0]
    return create_base_map(
        float(first_row['lat_mean']),
        float(first_row['lon_mean']),
        primary_label=loc['map_label']
    )


# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def show_monthly_hfr_page(loc):
    st.markdown(f"## Monthly HFR Surface Currents")
    with st.spinner("Loading monthly HFR data..."):
        df = load_hfr(loc, 'MS')
        base_map_bytes = make_base_map(df, loc)
    show_data_interface(df, base_map_bytes, "Monthly HFR", freq_label="Monthly")


def show_weekly_hfr_page(loc):
    st.markdown(f"## Weekly HFR Surface Currents")
    with st.spinner("Loading weekly HFR data..."):
        df = load_hfr(loc, 'W')
        base_map_bytes = make_base_map(df, loc)
    st.success(f"✅ Loaded {len(df)} weekly means from "
               f"{df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}")
    show_data_interface(df, base_map_bytes, "Weekly HFR", freq_label="Weekly")


def show_doppio_surface_page(loc):
    st.markdown(f"## Weekly Surface DOPPIO")
    with st.spinner("Loading DOPPIO surface data..."):
        df = load_doppio(loc, 'surface', 'W')
        base_map_bytes = make_base_map(df, loc)
    st.success(f"✅ Loaded DOPPIO surface data — {len(df)} weeks")
    show_data_interface(df, base_map_bytes, "DOPPIO Surface", freq_label="Weekly",
                        arrow_color='darkblue', ylim=[-25, 25])


def show_doppio_bottom_page(loc):
    st.markdown(f"## Weekly Bottom DOPPIO")
    with st.spinner("Loading DOPPIO bottom data..."):
        df = load_doppio(loc, 'bottom', 'W')
        base_map_bytes = make_base_map(df, loc)
    st.success(f"✅ Loaded DOPPIO bottom data — {len(df)} weeks")
    show_data_interface(df, base_map_bytes, "DOPPIO Bottom", freq_label="Weekly",
                        arrow_color='darkred', ylim=[-25, 25])


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.sidebar.markdown("## Choose a Location")
    location = st.sidebar.selectbox(
        "Location:",
        list(LOCATIONS.keys()),
        key="velocity_location"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Choose a Dataset")

    dataset = st.sidebar.selectbox(
        "Dataset:",
        ["Monthly HFR Surface Currents", "Weekly HFR Surface Currents",
         "Weekly Surface DOPPIO", "Weekly Bottom DOPPIO"],
        key="velocity_dataset"
    )

    st.sidebar.markdown("---")

    loc = LOCATIONS[location]
    st.markdown(f"### 📍 {location}")

    if dataset == "Monthly HFR Surface Currents":
        show_monthly_hfr_page(loc)
    elif dataset == "Weekly HFR Surface Currents":
        show_weekly_hfr_page(loc)
    elif dataset == "Weekly Surface DOPPIO":
        show_doppio_surface_page(loc)
    elif dataset == "Weekly Bottom DOPPIO":
        show_doppio_bottom_page(loc)


if __name__ == "__main__":
    main()
