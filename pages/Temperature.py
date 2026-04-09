import streamlit as st
import pandas as pd
from datetime import timedelta
from pathlib import Path

import sys
sys.path.append('..')
from utils.data_loading import load_ghrsst_data, load_doppio_single_layer, load_ocim2_data
from utils.plotting import create_temperature_timeseries, update_selection, create_base_map
from utils.ui_components import apply_custom_css

st.set_page_config(
    page_title="Temperature Data",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# ── Location configuration ─────────────────────────────────────────────────────
LOCATIONS = {
    "Ocean City, MA": {
        "ghrsst_daily":   "ghrsst_MBON3_daily.nc",
        "ghrsst_weekly":  "ghrsst_MBON3_weekly.nc",
        "ghrsst_monthly": "ghrsst_MBON3_monthly.nc",
        "doppio_surface": "doppio_timeseries_surface.nc",
        "doppio_bottom":  "doppio_timeseries_bottom.nc",
        "ocim2_path":     "ocim2_combined_hourly.csv",
        "ocim2_lat":      38.328,
        "ocim2_lon":      -75.091,
        "buoy_label":     "OCIM2 Buoy",
        "map_label":      "MBON3",
    },
    "Atlantic City, NJ": {
        "ghrsst_daily":   "ghrsst_MOTZ_daily.nc",
        "ghrsst_weekly":  "ghrsst_MOTZ_weekly.nc",
        "ghrsst_monthly": "ghrsst_MOTZ_monthly.nc",
        "doppio_surface": "DOPPIO_surface_MOTZ_timeseries.nc",
        "doppio_bottom":  "DOPPIO_bottom_MOTZ_timeseries.nc",
        "ocim2_path":     "acyn4_combined_hourly.csv",
        "ocim2_lat":      39.357,
        "ocim2_lon":      -74.418,
        "buoy_label":     "ACYN4",
        "map_label":      "Target Location",
    },
}
# ──────────────────────────────────────────────────────────────────────────────

_EMPTY_TEMP_DF = pd.DataFrame(
    columns=['time', 'temp_mean', 'temp_std', 'temp_count', 'temp_stderr', 'lat_mean', 'lon_mean']
)


def show_temperature_interface(df_ghrsst, df_surface, df_bottom, df_ocim2,
                                base_map_bytes, temporal_res, loc, ghrsst_available):
    """Display temperature interface with scrubbing and plot."""

    df_ref = df_ghrsst if ghrsst_available else df_surface
    if df_ref.empty:
        st.warning("No data available.")
        return

    # Compute the date range from all available datasets so the widget
    # defaults to the full span rather than being clamped by the shortest series.
    all_dfs = [d for d in [df_ghrsst, df_surface, df_bottom, df_ocim2]
               if d is not None and not d.empty]
    min_time = min(d['time'].min() for d in all_dfs)
    max_time = max(d['time'].max() for d in all_dfs)

    st.sidebar.subheader("Time Range Selection")
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(min_time.date(), max_time.date()),
        min_value=min_time.date(),
        max_value=max_time.date(),
        key=f"temp_date_range_{temporal_res}_{loc['map_label']}"
    )

    if len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + timedelta(days=31)
        df_ghrsst_filt  = df_ghrsst[(df_ghrsst['time'] >= start_date) & (df_ghrsst['time'] < end_date)]
        df_surface_filt = df_surface[(df_surface['time'] >= start_date) & (df_surface['time'] < end_date)]
        df_bottom_filt  = df_bottom[(df_bottom['time'] >= start_date) & (df_bottom['time'] < end_date)]
        df_ocim2_filt   = df_ocim2[(df_ocim2['time'] >= start_date) & (df_ocim2['time'] < end_date)] if df_ocim2 is not None else None
        _ocim2_for_slider = [df_ocim2_filt] if df_ocim2_filt is not None and not df_ocim2_filt.empty else []
        _slider_candidates = [d for d in [df_ghrsst_filt, df_surface_filt, df_bottom_filt] + _ocim2_for_slider if not d.empty]
        df_ref_filt     = max(_slider_candidates, key=lambda d: d['time'].max()) if _slider_candidates else df_surface_filt
        time_range      = (start_date, end_date)
    else:
        df_ghrsst_filt  = df_ghrsst
        df_surface_filt = df_surface
        df_bottom_filt  = df_bottom
        df_ocim2_filt   = df_ocim2
        _ocim2_for_slider = [df_ocim2_filt] if df_ocim2_filt is not None and not df_ocim2_filt.empty else []
        _slider_candidates = [d for d in [df_ghrsst_filt, df_surface_filt, df_bottom_filt] + _ocim2_for_slider if not d.empty]
        df_ref_filt     = max(_slider_candidates, key=lambda d: d['time'].max()) if _slider_candidates else df_surface_filt
        time_range      = None

    ocim2_available = df_ocim2 is not None

    st.sidebar.subheader("Display Options")
    show_ghrsst  = st.sidebar.checkbox("Show GHRSST Satellite SST", value=ghrsst_available,
                                        disabled=not ghrsst_available,
                                        key=f"show_ghrsst_{temporal_res}_{loc['map_label']}")
    show_surface = st.sidebar.checkbox("Show DOPPIO Surface", value=True,
                                        key=f"show_surface_{temporal_res}_{loc['map_label']}")
    show_bottom  = st.sidebar.checkbox("Show DOPPIO Bottom", value=True,
                                        key=f"show_bottom_{temporal_res}_{loc['map_label']}")
    show_ocim2   = st.sidebar.checkbox(f"Show {loc['buoy_label']}", value=ocim2_available,
                                        disabled=not ocim2_available,
                                        key=f"show_ocim2_{temporal_res}_{loc['map_label']}")
    st.sidebar.markdown("---")

    base_ts_fig = create_temperature_timeseries(
        df_ghrsst_filt, df_surface_filt, df_bottom_filt,
        df_ocim2=df_ocim2_filt,
        time_range=time_range,
        freq_label=temporal_res,
        show_ghrsst=show_ghrsst,
        show_surface=show_surface,
        show_bottom=show_bottom,
        show_ocim2=show_ocim2,
        buoy_label=loc['buoy_label']
    )

    st.sidebar.subheader(f"{temporal_res} Selection")

    if len(df_ref_filt) > 0:
        slider_key = f"temp_slider_{temporal_res.lower()}_{loc['map_label']}"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = 0

        st.session_state[slider_key] = max(0, min(st.session_state[slider_key], len(df_ref_filt) - 1))

        col_left, col_slider, col_right = st.sidebar.columns([1, 6, 1])

        with col_left:
            if st.button("◀", key=f"temp_left_{temporal_res}_{loc['map_label']}",
                         help="Previous time step", use_container_width=True):
                if st.session_state[slider_key] > 0:
                    st.session_state[slider_key] -= 1

        with col_right:
            if st.button("▶", key=f"temp_right_{temporal_res}_{loc['map_label']}",
                         help="Next time step", use_container_width=True):
                if st.session_state[slider_key] < len(df_ref_filt) - 1:
                    st.session_state[slider_key] += 1

        with col_slider:
            selected_idx = st.slider(
                f"Select {temporal_res.lower()}",
                min_value=0,
                max_value=len(df_ref_filt) - 1,
                key=slider_key,
                label_visibility="collapsed"
            )

        selected_time = df_ref_filt.iloc[selected_idx]['time']

        def closest_row(df):
            if df is None or df.empty:
                return None
            return df.iloc[(df['time'] - selected_time).abs().argmin()]

        ghrsst_row  = closest_row(df_ghrsst_filt)
        surface_row = closest_row(df_surface_filt)
        bottom_row  = closest_row(df_bottom_filt)
        ocim2_row   = closest_row(df_ocim2_filt)

        ts_fig = update_selection(base_ts_fig, selected_time)
        st.plotly_chart(ts_fig, use_container_width=True)

        st.markdown("---")
        time_format = '%Y-%m' if temporal_res == "Monthly" else '%Y-%m-%d'
        st.markdown(f"### Data Values at {selected_time.strftime(time_format)}")

        visible = []
        if show_ghrsst and ghrsst_available and ghrsst_row is not None:
            visible.append(('GHRSST Satellite SST', ghrsst_row))
        if show_surface and surface_row is not None:
            visible.append(('DOPPIO Surface', surface_row))
        if show_bottom and bottom_row is not None:
            visible.append(('DOPPIO Bottom', bottom_row))
        if show_ocim2 and ocim2_available and ocim2_row is not None:
            visible.append((loc['buoy_label'], ocim2_row))

        if visible:
            cols = st.columns(len(visible))
            for col, (name, row) in zip(cols, visible):
                with col:
                    st.markdown(f"**{name}**")
                    st.metric("Temperature", f"{row['temp_mean']:.2f} °C",
                              help=f"± {row['temp_stderr']:.2f} °C")
                    st.caption(f"N = {row['temp_count']:.0f} observations")
        else:
            st.info("No datasets selected. Use the checkboxes in the sidebar.")

        st.markdown("---")
        st.markdown("### Measurement Location")
        col_map1, col_map2, col_map3 = st.columns([1, 2, 1])
        with col_map2:
            st.image(base_map_bytes, use_container_width=True)
            ref_row = ghrsst_row if (ghrsst_available and ghrsst_row is not None) else surface_row
            if ref_row is not None:
                caption = f"{loc['map_label']}: {ref_row['lat_mean']:.4f}°N, {abs(ref_row['lon_mean']):.4f}°W"
                if ocim2_available:
                    caption += f" | {loc['buoy_label']}: {loc['ocim2_lat']}°N, {abs(loc['ocim2_lon'])}°W"
                st.caption(caption)

    else:
        st.warning("No data available for the selected time range.")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_page(loc, ghrsst_path, resample_freq, temporal_res):
    """Load all data for a location + temporal resolution and render the interface."""
    ghrsst_available = Path(ghrsst_path).exists()

    if not ghrsst_available:
        st.info(
            f"**GHRSST data not yet available** (`{ghrsst_path}`). "
            f"DOPPIO temperature data will be shown. "
            f"Drop the GHRSST file in the app directory when ready and reload.",
            icon="ℹ️"
        )

    with st.spinner(f"Loading {temporal_res.lower()} temperature data..."):
        df_ghrsst = load_ghrsst_data(ghrsst_path) if ghrsst_available else _EMPTY_TEMP_DF.copy()

        if Path(loc['doppio_surface']).exists():
            df_surface = load_doppio_single_layer(loc['doppio_surface'], 'surface', resample_freq=resample_freq)
        else:
            st.warning(f"DOPPIO surface file not found: {loc['doppio_surface']}")
            df_surface = _EMPTY_TEMP_DF.copy()

        if Path(loc['doppio_bottom']).exists():
            df_bottom = load_doppio_single_layer(loc['doppio_bottom'], 'bottom', resample_freq=resample_freq)
        else:
            st.warning(f"DOPPIO bottom file not found: {loc['doppio_bottom']}")
            df_bottom = _EMPTY_TEMP_DF.copy()

        if loc['ocim2_path'] and Path(loc['ocim2_path']).exists():
            df_ocim2 = load_ocim2_data(loc['ocim2_path'], resample_freq=resample_freq,
                                        lat=loc['ocim2_lat'], lon=loc['ocim2_lon'])
        else:
            df_ocim2 = None

        # Align GHRSST to the common start date across all loaded datasets so the
        # plot doesn't show years of GHRSST data with no DOPPIO to compare against.
        doppio_starts = [df['time'].min() for df in [df_surface, df_bottom] if not df.empty]
        if ghrsst_available and doppio_starts:
            common_start = max(doppio_starts)
            df_ghrsst = df_ghrsst[df_ghrsst['time'] >= common_start].reset_index(drop=True)

        ref_df = df_ghrsst if (ghrsst_available and not df_ghrsst.empty) else df_surface
        if ref_df.empty:
            st.error("No data available to render the page.")
            st.stop()

        first_row = ref_df.iloc[0]
        base_map_bytes = create_base_map(
            float(first_row['lat_mean']),
            float(first_row['lon_mean']),
            ocim2_lat=loc['ocim2_lat'],
            ocim2_lon=loc['ocim2_lon'],
            primary_label=loc['map_label']
        )

    show_temperature_interface(df_ghrsst, df_surface, df_bottom, df_ocim2,
                                base_map_bytes, temporal_res, loc, ghrsst_available)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.sidebar.markdown("## Choose a Location")
    location = st.sidebar.selectbox(
        "Location:",
        list(LOCATIONS.keys()),
        key="temp_location"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Choose Temporal Resolution")

    temporal_resolution = st.sidebar.selectbox(
        "Temporal resolution:",
        ["Daily Temperature", "Weekly Temperature", "Monthly Temperature"],
        key="temp_resolution"
    )

    st.sidebar.markdown("---")

    loc = LOCATIONS[location]
    st.markdown(f"### 📍 {location}")

    if temporal_resolution == "Daily Temperature":
        st.markdown("## Daily Temperature Data")
        load_page(loc, loc['ghrsst_daily'], 'D', "Daily")
    elif temporal_resolution == "Weekly Temperature":
        st.markdown("## Weekly Temperature Data")
        load_page(loc, loc['ghrsst_weekly'], 'W', "Weekly")
    elif temporal_resolution == "Monthly Temperature":
        st.markdown("## Monthly Temperature Data")
        load_page(loc, loc['ghrsst_monthly'], 'MS', "Monthly")


if __name__ == "__main__":
    main()
