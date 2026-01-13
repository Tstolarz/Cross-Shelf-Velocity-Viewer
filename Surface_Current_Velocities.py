import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Import from utils modules
from utils.data_loading import load_mbon_data, load_doppio_single_layer
from utils.plotting import (create_timeseries_base, update_selection,
                            create_base_map, create_arrow_overlay, create_map_plot)
from utils.ui_components import time_it, apply_custom_css

# Set page config
st.set_page_config(
    page_title="Cross-Shelf Velocity Viewer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS styling and keyboard navigation
apply_custom_css()

# ============================================================================
# Data loading and plotting functions are imported from utils modules
# ============================================================================


def show_monthly_mbon_page():
    st.markdown("## Monthly HFR Surface Current Data")
    
    # File upload or default path
    uploaded_file = st.sidebar.file_uploader(
        "Upload a surface current time series NetCDF file", 
        type=['nc'],
        help="Upload a U & V time series dataset at a single location",
        key="mbon_monthly_upload"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_mbon_data.nc", "wb") as f:
            f.write(uploaded_file.getbuffer())
        nc_path = "temp_mbon_data.nc"
    else:
        # Use default path
        default_path = "2007_2024_10_10_MARACOOS_uv_MBON_timeseries_v5_corrected.nc"
        if Path(default_path).exists():
            nc_path = default_path
        else:
            st.error("Please upload a NetCDF file or ensure the default file exists.")
            st.stop()
    
    # Load monthly data
    with st.spinner("Loading and processing monthly HFR Surface Current data..."):
        df_monthly = load_mbon_data(nc_path, resample_freq='M')
        
        # Create base map once (cached)
        first_row = df_monthly.iloc[0]
        base_map_bytes = create_base_map(float(first_row['lat_mean']), float(first_row['lon_mean']))
    
    # Success message
    st.success(f"✅ Loaded {len(df_monthly)} monthly means from {df_monthly['time'].min().strftime('%Y-%m')} to {df_monthly['time'].max().strftime('%Y-%m')}")
    
    show_data_interface(df_monthly, base_map_bytes, "Monthly", freq_label="Monthly")

def show_weekly_mbon_page():
    st.markdown("## Weekly HFR Surface Current Data")
    
    # File upload or default path
    uploaded_file = st.sidebar.file_uploader(
        "Upload HFR Surface Current Timeseries NetCDF file", 
        type=['nc'],
        help="Upload your HFR Surface Current dataset",
        key="mbon_weekly_upload"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_mbon_weekly_data.nc", "wb") as f:
            f.write(uploaded_file.getbuffer())
        nc_path = "temp_mbon_weekly_data.nc"
    else:
        # Use default path
        default_path = "2007_2024_10_10_MARACOOS_uv_MBON_timeseries_v5_corrected.nc"
        if Path(default_path).exists():
            nc_path = default_path
        else:
            st.error("Please upload a NetCDF file or ensure the default file exists.")
            st.stop()
    
    # Load weekly data
    with st.spinner("Loading and processing weekly HFR Surface Current data..."):
        df_weekly = load_mbon_data(nc_path, resample_freq='W')
        
        # Create base map once (cached)
        first_row = df_weekly.iloc[0]
        base_map_bytes = create_base_map(float(first_row['lat_mean']), float(first_row['lon_mean']))
    
    # Success message
    st.success(f"✅ Loaded {len(df_weekly)} weekly means from {df_weekly['time'].min().strftime('%Y-%m-%d')} to {df_weekly['time'].max().strftime('%Y-%m-%d')}")
    
    show_data_interface(df_weekly, base_map_bytes, "Weekly", freq_label="Weekly")

def show_doppio_surface_page():
    st.markdown("## Weekly Surface DOPPIO Model Data")
    
    # Default DOPPIO files
    default_surface = "doppio_timeseries_surface.nc"
    default_bottom = "doppio_timeseries_bottom.nc"
    
    surface_file = None
    bottom_file = None
    
    # Check for default files
    if Path(default_surface).exists() and Path(default_bottom).exists():
        st.info(f"✅ Using default DOPPIO files: {default_surface} and {default_bottom}", )
        surface_file = default_surface
        bottom_file = default_bottom
    else:
        st.warning("Default DOPPIO files not found. Please upload both surface and bottom NetCDF files.")
        
        # File uploads
        col1, col2 = st.columns(2)
        with col1:
            uploaded_surface = st.file_uploader(
                "Upload DOPPIO Surface NetCDF", 
                type=['nc'],
                help="Upload DOPPIO surface current dataset",
                key="doppio_surface_upload"
            )
        
        with col2:
            uploaded_bottom = st.file_uploader(
                "Upload DOPPIO Bottom NetCDF", 
                type=['nc'],
                help="Upload DOPPIO bottom current dataset",
                key="doppio_bottom_upload"
            )
        
        if uploaded_surface is not None and uploaded_bottom is not None:
            # Save uploaded files temporarily
            with open("temp_doppio_surface.nc", "wb") as f:
                f.write(uploaded_surface.getbuffer())
            with open("temp_doppio_bottom.nc", "wb") as f:
                f.write(uploaded_bottom.getbuffer())
            surface_file = "temp_doppio_surface.nc"
            bottom_file = "temp_doppio_bottom.nc"
        else:
            st.error("Please upload both surface and bottom NetCDF files to continue.")
            st.stop()
    
    # Load DOPPIO surface data only
    with st.spinner("Loading and processing DOPPIO surface data..."):
        df_surface = load_doppio_single_layer(surface_file, 'surface', resample_freq='W')
        
        # Create base map using surface data
        first_surface_row = df_surface.iloc[0]
        base_map_bytes = create_base_map(
            float(first_surface_row['lat_mean']), 
            float(first_surface_row['lon_mean'])
        )
    
    st.success(f"✅ Loaded DOPPIO surface data - {len(df_surface)} weeks")
    arrow_color = 'darkblue'
    
    show_data_interface(df_surface, base_map_bytes, "DOPPIO Surface", 
                       freq_label="Weekly", arrow_color=arrow_color, ylim=[-25, 25])

def show_doppio_bottom_page():
    st.markdown("## Weekly Bottom DOPPIO Model Data")
    
    # Default DOPPIO files
    default_surface = "doppio_timeseries_surface.nc"
    default_bottom = "doppio_timeseries_bottom.nc"
    
    surface_file = None
    bottom_file = None
    
    # Check for default files
    if Path(default_surface).exists() and Path(default_bottom).exists():
        st.info(f"✅ Using default DOPPIO files: {default_surface} and {default_bottom}")
        surface_file = default_surface
        bottom_file = default_bottom
    else:
        st.warning("Default DOPPIO files not found. Please upload both surface and bottom NetCDF files.")
        
        # File uploads
        col1, col2 = st.columns(2)
        with col1:
            uploaded_surface = st.file_uploader(
                "Upload DOPPIO Surface NetCDF", 
                type=['nc'],
                help="Upload DOPPIO surface current dataset",
                key="doppio_surface_upload_bottom_page"
            )
        
        with col2:
            uploaded_bottom = st.file_uploader(
                "Upload DOPPIO Bottom NetCDF", 
                type=['nc'],
                help="Upload DOPPIO bottom current dataset",
                key="doppio_bottom_upload_bottom_page"
            )
        
        if uploaded_surface is not None and uploaded_bottom is not None:
            # Save uploaded files temporarily
            with open("temp_doppio_surface_bottom_page.nc", "wb") as f:
                f.write(uploaded_surface.getbuffer())
            with open("temp_doppio_bottom_bottom_page.nc", "wb") as f:
                f.write(uploaded_bottom.getbuffer())
            surface_file = "temp_doppio_surface_bottom_page.nc"
            bottom_file = "temp_doppio_bottom_bottom_page.nc"
        else:
            st.error("Please upload both surface and bottom NetCDF files to continue.")
            st.stop()
    
    # Load DOPPIO bottom data only
    with st.spinner("Loading and processing DOPPIO bottom data..."):
        df_bottom = load_doppio_single_layer(bottom_file, 'bottom', resample_freq='W')
        
        # Create base map using bottom data
        first_bottom_row = df_bottom.iloc[0]
        base_map_bytes = create_base_map(
            float(first_bottom_row['lat_mean']), 
            float(first_bottom_row['lon_mean'])
        )
    
    st.success(f"✅ Loaded DOPPIO bottom data - {len(df_bottom)} weeks")
    arrow_color = 'darkred'
    
    show_data_interface(df_bottom, base_map_bytes, "DOPPIO Bottom", 
                       freq_label="Weekly", arrow_color=arrow_color, ylim=[-25, 25])

def show_data_interface(df, base_map_bytes, data_label, freq_label="Monthly", arrow_color='darkblue', ylim=None):
    """Shared interface for displaying timeseries data"""
    
    # Time range selection for zoom
    st.sidebar.subheader("Time Range Selection")
    
    # Full range by default
    full_range = (df['time'].min(), df['time'].max())
    
    # Date range selector
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(full_range[0].date(), full_range[1].date()),
        min_value=full_range[0].date(),
        max_value=full_range[1].date(),
        key=f"date_range_{data_label.lower().replace(' ', '_')}"
    )
    
    # Convert back to datetime for filtering
    if len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + timedelta(days=31)  # Buffer
        
        # Filter dataframe
        df_filtered = df[(df['time'] >= start_date) & (df['time'] < end_date)]
        time_range = (start_date, end_date)
        
        # Create base time series
        base_ts_fig = create_timeseries_base(df, time_range, freq_label, ylim)
    else:
        df_filtered = df
        time_range = None
        base_ts_fig = create_timeseries_base(df, None, freq_label, ylim)
    
    # Time point slider
    st.sidebar.subheader(f"{freq_label} Selection")
    
    if len(df_filtered) > 0:
        # Initialize slider value from session state if it exists
        slider_key = f"slider_{data_label.lower().replace(' ', '_')}"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = 0
        
        # Ensure the session state value is within bounds
        st.session_state[slider_key] = max(0, min(st.session_state[slider_key], len(df_filtered) - 1))
        
        # Arrow navigation buttons
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
        
        # Get the actual index in the full dataset
        selected_time = df_filtered.iloc[selected_idx_filtered]['time']
        selected_idx = df[df['time'] == selected_time].index[0]
        
        # Get current row data
        current_row = df.iloc[selected_idx]
        
        # Display current selection info
        time_format = '%Y-%m' if freq_label == "Monthly" else '%Y-%m-%d'

        st.sidebar.info(f"""
        **Current Selection ({data_label}):**
        - **{current_row['time'].strftime(time_format)}**
        - Along: {current_row['v_along_mean']:.1f} ± {current_row['v_along_stderr']:.1f} cm/s
        - Cross: {current_row['u_cross_mean']:.1f} ± {current_row['u_cross_stderr']:.1f} cm/s
        - N obs: {current_row['magnitude_count']:.0f}
        """)
        
        # Performance metrics - collapsible
        with st.sidebar.expander("⏱️ Performance Metrics", expanded=False):
            if 'timing_data' in st.session_state and st.session_state.timing_data:
                for func_name, duration in st.session_state.timing_data.items():
                    color = "🟢" if duration < 50 else "🟡" if duration < 200 else "🔴"
                    st.text(f"{color} {func_name}: {duration:.1f}ms")
        
        # Main layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Time series plots
            ts_fig = update_selection(base_ts_fig, current_row['time'])
            st.plotly_chart(ts_fig, use_container_width=True)
        
        with col2:
            # Current vector information
            st.markdown(f"#### {freq_label} Current Details")

            # 2 columns layout (temperature removed)
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Along-shelf", f"{current_row['v_along_mean']:.1f} cm/s")
                st.metric("Cross-shelf", f"{current_row['u_cross_mean']:.1f} cm/s")
                st.metric("Observations", f"{current_row['magnitude_count']:.0f}")
            with col2b:
                st.metric("U (East)", f"{current_row['u_mean']:.1f} cm/s")
                st.metric("V (North)", f"{current_row['v_mean']:.1f} cm/s")

        # Map section
        col_map1, col_map2, col_map3 = st.columns([1, 2, 1])
        
        with col_map2:
            map_img = create_map_plot(df, selected_idx, base_map_bytes, freq_label, arrow_color)
            st.image(map_img, use_container_width=True)
        
    else:
        st.warning("No data available for the selected time range.")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # st.markdown("# 🌊 Ocean Current Multi-Viewer")
    
    # Navigation in sidebar
    st.sidebar.markdown("## Choose a Dataset")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Monthly HFR Surface Currents", "Weekly HFR Surface Currents", "Weekly Surface DOPPIO", "Weekly Bottom DOPPIO"],
        help="Navigate between different data views"
    )
    
    st.sidebar.markdown("---")
    
    # Route to appropriate page
    if page == "Monthly HFR Surface Currents":
        show_monthly_mbon_page()
    elif page == "Weekly HFR Surface Currents":
        show_weekly_mbon_page()
    elif page == "Weekly Surface DOPPIO":
        show_doppio_surface_page()
    elif page == "Weekly Bottom DOPPIO":
        show_doppio_bottom_page()

if __name__ == "__main__":
    main()