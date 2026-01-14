import streamlit as st
import pandas as pd
from datetime import timedelta
from pathlib import Path

# Import from parent directory utils
import sys
sys.path.append('..')
from utils.data_loading import load_ghrsst_data, load_doppio_single_layer, load_ocim2_data
from utils.plotting import create_temperature_timeseries, update_selection, create_base_map
from utils.ui_components import apply_custom_css

# Page config
st.set_page_config(
    page_title="Temperature Data",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS styling and keyboard navigation
apply_custom_css()

def show_temperature_interface(df_ghrsst, df_surface, df_bottom, df_ocim2, base_map_bytes, temporal_res):
    """Display temperature data interface with scrubbing and plots"""

    # Find common time range across all datasets
    min_time = max(df_surface['time'].min(), df_bottom['time'].min())
    max_time = min(df_surface['time'].max(), df_bottom['time'].max())

    # Sidebar: Time range selection
    st.sidebar.subheader("Time Range Selection")
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(min_time.date(), max_time.date()),
        min_value=df_surface['time'].min().date(),
        max_value=df_surface['time'].max().date(),
        key=f"temp_date_range_{temporal_res}"
    )

    if len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + timedelta(days=31)

        # Filter all datasets to common range
        df_ghrsst_filt = df_ghrsst[(df_ghrsst['time'] >= start_date) & (df_ghrsst['time'] < end_date)]
        df_surface_filt = df_surface[(df_surface['time'] >= start_date) & (df_surface['time'] < end_date)]
        df_bottom_filt = df_bottom[(df_bottom['time'] >= start_date) & (df_bottom['time'] < end_date)]
        df_ocim2_filt = df_ocim2[(df_ocim2['time'] >= start_date) & (df_ocim2['time'] < end_date)]

        time_range = (start_date, end_date)
    else:
        df_ghrsst_filt = df_ghrsst
        df_surface_filt = df_surface
        df_bottom_filt = df_bottom
        df_ocim2_filt = df_ocim2
        time_range = None

    # Dataset visibility controls
    st.sidebar.subheader("Display Options")
    show_ghrsst = st.sidebar.checkbox("Show GHRSST Satellite SST", value=True, key=f"show_ghrsst_{temporal_res}")
    show_surface = st.sidebar.checkbox("Show DOPPIO Surface", value=True, key=f"show_surface_{temporal_res}")
    show_bottom = st.sidebar.checkbox("Show DOPPIO Bottom", value=True, key=f"show_bottom_{temporal_res}")
    show_ocim2 = st.sidebar.checkbox("Show OCIM2 Buoy", value=True, key=f"show_ocim2_{temporal_res}")

    st.sidebar.markdown("---")

    # Create base time series (all four sources)
    base_ts_fig = create_temperature_timeseries(
        df_ghrsst_filt, df_surface_filt, df_bottom_filt, df_ocim2_filt,
        time_range=time_range,
        freq_label=temporal_res,
        show_ghrsst=show_ghrsst,
        show_surface=show_surface,
        show_bottom=show_bottom,
        show_ocim2=show_ocim2
    )

    # Time scrubbing: Use GHRSST as reference
    st.sidebar.subheader(f"{temporal_res} Selection")

    if len(df_ghrsst_filt) > 0:
        slider_key = f"temp_slider_{temporal_res.lower()}"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = 0

        st.session_state[slider_key] = max(0, min(st.session_state[slider_key], len(df_ghrsst_filt) - 1))

        # Arrow navigation
        col_left, col_slider, col_right = st.sidebar.columns([1, 6, 1])

        with col_left:
            if st.button("◀", key=f"temp_left_{temporal_res}",
                        help="Previous time step", use_container_width=True):
                if st.session_state[slider_key] > 0:
                    st.session_state[slider_key] -= 1

        with col_right:
            if st.button("▶", key=f"temp_right_{temporal_res}",
                        help="Next time step", use_container_width=True):
                if st.session_state[slider_key] < len(df_ghrsst_filt) - 1:
                    st.session_state[slider_key] += 1

        with col_slider:
            selected_idx = st.slider(
                f"Select {temporal_res.lower()}",
                min_value=0,
                max_value=len(df_ghrsst_filt) - 1,
                key=slider_key,
                label_visibility="collapsed"
            )

        # Get current selection
        selected_time = df_ghrsst_filt.iloc[selected_idx]['time']

        # Find matching rows in all datasets (closest time)
        ghrsst_row = df_ghrsst_filt.iloc[selected_idx]

        if len(df_surface_filt) > 0:
            surface_row = df_surface_filt.iloc[(df_surface_filt['time'] - selected_time).abs().argmin()]
        else:
            surface_row = None

        if len(df_bottom_filt) > 0:
            bottom_row = df_bottom_filt.iloc[(df_bottom_filt['time'] - selected_time).abs().argmin()]
        else:
            bottom_row = None

        if len(df_ocim2_filt) > 0:
            ocim2_row = df_ocim2_filt.iloc[(df_ocim2_filt['time'] - selected_time).abs().argmin()]
        else:
            ocim2_row = None

        # Time series plot (full width)
        ts_fig = update_selection(base_ts_fig, selected_time)
        st.plotly_chart(ts_fig, use_container_width=True)

        # Data values BELOW plot - only show for visible datasets
        st.markdown("---")
        time_format = '%Y-%m' if temporal_res == "Monthly" else '%Y-%m-%d'
        st.markdown(f"### Data Values at {selected_time.strftime(time_format)}")

        # Create columns dynamically based on which datasets are visible
        visible_datasets = []
        if show_ghrsst:
            visible_datasets.append(('GHRSST Satellite SST', ghrsst_row))
        if show_surface and surface_row is not None:
            visible_datasets.append(('DOPPIO Surface', surface_row))
        if show_bottom and bottom_row is not None:
            visible_datasets.append(('DOPPIO Bottom', bottom_row))
        if show_ocim2 and ocim2_row is not None:
            visible_datasets.append(('OCIM2 Buoy', ocim2_row))

        if len(visible_datasets) == 0:
            st.info("No datasets selected for display. Use the checkboxes in the sidebar to show data.")
        else:
            cols = st.columns(len(visible_datasets))

            for idx, (col, (name, row)) in enumerate(zip(cols, visible_datasets)):
                with col:
                    st.markdown(f"**{name}**")
                    if name == 'GHRSST Satellite SST':
                        st.metric("Temperature", f"{row['temp_mean']:.2f} °C",
                                 help=f"± {row['temp_stderr']:.2f} °C" if row['temp_stderr'] > 0 else "Pre-aggregated data")
                    else:
                        st.metric("Temperature", f"{row['temp_mean']:.2f} °C",
                                 help=f"± {row['temp_stderr']:.2f} °C")
                        st.caption(f"Time: {row['time'].strftime(time_format)}")
                    st.caption(f"N = {row['temp_count']:.0f} observations")

        # Static location map
        st.markdown("---")
        st.markdown("### Measurement Locations")

        col_map1, col_map2, col_map3 = st.columns([1, 2, 1])
        with col_map2:
            st.image(base_map_bytes, use_container_width=True)
            st.caption(f"MBON3: {ghrsst_row['lat_mean']:.4f}°N, {abs(ghrsst_row['lon_mean']):.4f}°W | OCIM2 Buoy: 38.328°N, 75.091°W")

    else:
        st.warning("No data available for the selected time range.")

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def show_daily_temperature_page():
    st.markdown("## Daily Temperature Data")

    with st.spinner("Loading daily temperature data..."):
        ghrsst_path = "ghrsst_MBON3_daily.nc"
        doppio_surface_path = "doppio_timeseries_surface.nc"
        doppio_bottom_path = "doppio_timeseries_bottom.nc"
        ocim2_path = "ocim2_combined_hourly.csv"

        if not Path(ghrsst_path).exists():
            st.error(f"GHRSST daily file not found: {ghrsst_path}")
            st.stop()

        # Load GHRSST data
        df_ghrsst = load_ghrsst_data(ghrsst_path)

        # Load and resample DOPPIO data to daily
        if Path(doppio_surface_path).exists():
            df_surface = load_doppio_single_layer(doppio_surface_path, 'surface', resample_freq='D')
        else:
            st.warning("DOPPIO surface file not found")
            df_surface = pd.DataFrame(columns=['time', 'temp_mean', 'temp_std', 'temp_count', 'temp_stderr'])

        if Path(doppio_bottom_path).exists():
            df_bottom = load_doppio_single_layer(doppio_bottom_path, 'bottom', resample_freq='D')
        else:
            st.warning("DOPPIO bottom file not found")
            df_bottom = pd.DataFrame(columns=['time', 'temp_mean', 'temp_std', 'temp_count', 'temp_stderr'])

        # Load OCIM2 buoy data (resample to daily)
        if Path(ocim2_path).exists():
            df_ocim2 = load_ocim2_data(ocim2_path, resample_freq='D')
        else:
            st.warning("OCIM2 buoy file not found")
            df_ocim2 = pd.DataFrame(columns=['time', 'temp_mean', 'temp_std', 'temp_count', 'temp_stderr'])

        # Create static base map with both MBON3 and OCIM2 locations
        first_row = df_ghrsst.iloc[0]
        base_map_bytes = create_base_map(
            float(first_row['lat_mean']),
            float(first_row['lon_mean']),
            ocim2_lat=38.328,
            ocim2_lon=-75.091
        )

    show_temperature_interface(df_ghrsst, df_surface, df_bottom, df_ocim2, base_map_bytes, "Daily")

def show_weekly_temperature_page():
    st.markdown("## Weekly Temperature Data")

    with st.spinner("Loading weekly temperature data..."):
        ghrsst_path = "ghrsst_MBON3_weekly.nc"
        doppio_surface_path = "doppio_timeseries_surface.nc"
        doppio_bottom_path = "doppio_timeseries_bottom.nc"
        ocim2_path = "ocim2_combined_hourly.csv"

        if not Path(ghrsst_path).exists():
            st.error(f"GHRSST weekly file not found: {ghrsst_path}")
            st.stop()

        # Load GHRSST data
        df_ghrsst = load_ghrsst_data(ghrsst_path)

        # Load and resample DOPPIO data to weekly
        if Path(doppio_surface_path).exists():
            df_surface = load_doppio_single_layer(doppio_surface_path, 'surface', resample_freq='W')
        else:
            st.warning("DOPPIO surface file not found")
            df_surface = pd.DataFrame(columns=['time', 'temp_mean', 'temp_std', 'temp_count', 'temp_stderr'])

        if Path(doppio_bottom_path).exists():
            df_bottom = load_doppio_single_layer(doppio_bottom_path, 'bottom', resample_freq='W')
        else:
            st.warning("DOPPIO bottom file not found")
            df_bottom = pd.DataFrame(columns=['time', 'temp_mean', 'temp_std', 'temp_count', 'temp_stderr'])

        # Load OCIM2 buoy data (resample to weekly)
        if Path(ocim2_path).exists():
            df_ocim2 = load_ocim2_data(ocim2_path, resample_freq='W')
        else:
            st.warning("OCIM2 buoy file not found")
            df_ocim2 = pd.DataFrame(columns=['time', 'temp_mean', 'temp_std', 'temp_count', 'temp_stderr'])

        # Create static base map with both MBON3 and OCIM2 locations
        first_row = df_ghrsst.iloc[0]
        base_map_bytes = create_base_map(
            float(first_row['lat_mean']),
            float(first_row['lon_mean']),
            ocim2_lat=38.328,
            ocim2_lon=-75.091
        )

    show_temperature_interface(df_ghrsst, df_surface, df_bottom, df_ocim2, base_map_bytes, "Weekly")

def show_monthly_temperature_page():
    st.markdown("## Monthly Temperature Data")

    with st.spinner("Loading monthly temperature data..."):
        ghrsst_path = "ghrsst_MBON3_monthly.nc"
        doppio_surface_path = "doppio_timeseries_surface.nc"
        doppio_bottom_path = "doppio_timeseries_bottom.nc"
        ocim2_path = "ocim2_combined_hourly.csv"

        if not Path(ghrsst_path).exists():
            st.error(f"GHRSST monthly file not found: {ghrsst_path}")
            st.stop()

        # Load GHRSST data
        df_ghrsst = load_ghrsst_data(ghrsst_path)

        # Load and resample DOPPIO data to monthly
        if Path(doppio_surface_path).exists():
            df_surface = load_doppio_single_layer(doppio_surface_path, 'surface', resample_freq='MS')
        else:
            st.warning("DOPPIO surface file not found")
            df_surface = pd.DataFrame(columns=['time', 'temp_mean', 'temp_std', 'temp_count', 'temp_stderr'])

        if Path(doppio_bottom_path).exists():
            df_bottom = load_doppio_single_layer(doppio_bottom_path, 'bottom', resample_freq='MS')
        else:
            st.warning("DOPPIO bottom file not found")
            df_bottom = pd.DataFrame(columns=['time', 'temp_mean', 'temp_std', 'temp_count', 'temp_stderr'])

        # Load OCIM2 buoy data (resample to monthly)
        if Path(ocim2_path).exists():
            df_ocim2 = load_ocim2_data(ocim2_path, resample_freq='MS')
        else:
            st.warning("OCIM2 buoy file not found")
            df_ocim2 = pd.DataFrame(columns=['time', 'temp_mean', 'temp_std', 'temp_count', 'temp_stderr'])

        # Create static base map with both MBON3 and OCIM2 locations
        first_row = df_ghrsst.iloc[0]
        base_map_bytes = create_base_map(
            float(first_row['lat_mean']),
            float(first_row['lon_mean']),
            ocim2_lat=38.328,
            ocim2_lon=-75.091
        )

    show_temperature_interface(df_ghrsst, df_surface, df_bottom, df_ocim2, base_map_bytes, "Monthly")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Navigation in sidebar
    st.sidebar.markdown("## Choose Temporal Resolution")

    temporal_resolution = st.sidebar.selectbox(
        "Select temporal resolution:",
        ["Daily Temperature", "Weekly Temperature", "Monthly Temperature"],
        help="Choose the temporal aggregation for temperature data"
    )

    st.sidebar.markdown("---")

    # Route to appropriate page
    if temporal_resolution == "Daily Temperature":
        show_daily_temperature_page()
    elif temporal_resolution == "Weekly Temperature":
        show_weekly_temperature_page()
    elif temporal_resolution == "Monthly Temperature":
        show_monthly_temperature_page()

if __name__ == "__main__":
    main()
