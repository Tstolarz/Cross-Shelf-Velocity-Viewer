import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import time

# Performance timing decorator
def time_it(func_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Store timing in session state for display
            if 'timing_data' not in st.session_state:
                st.session_state.timing_data = {}
            st.session_state.timing_data[func_name] = duration
            
            print(f"⏱️ {func_name}: {duration:.1f}ms")
            return result
        return wrapper
    return decorator

# Set page config
st.set_page_config(
    page_title="Ocean Current Multi-Viewer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to reduce padding and margins
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        margin-top: -80px;
    }
    h1 {
        padding-top: 0rem;
        margin-top: 0rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SHARED DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_mbon_data(nc_path, resample_freq='M'):
    """Load and preprocess the MBON dataset to specified frequency (M=monthly, W=weekly)"""
    ds = xr.open_dataset(nc_path)
    
    # Convert to pandas for easier handling
    df = pd.DataFrame({
        'time': pd.to_datetime(ds.time.values),
        'u': ds.u.values * 100,  # Convert to cm/s
        'v': ds.v.values * 100,  # Convert to cm/s
        'lon': ds.lon.values if ds.lon.dims else [ds.lon.values] * len(ds.time),
        'lat': ds.lat.values if ds.lat.dims else [ds.lat.values] * len(ds.time)
    })
    
    # Add along-shelf and cross-shelf components (using region4 rotation)
    rot_angle = 360 - 37  # degrees
    uv_complex = df['u'] + 1j * df['v']
    uv_rotated = uv_complex * np.exp(-1j * np.deg2rad(rot_angle))
    
    df['u_cross'] = np.real(uv_rotated)  # Cross-shelf (Ur)
    df['v_along'] = np.imag(uv_rotated)  # Along-shelf (Vr)
    
    # Calculate magnitude for each hourly record
    df['magnitude'] = np.sqrt(df['u']**2 + df['v']**2)
    
    # Resample to specified frequency with additional statistics
    df.set_index('time', inplace=True)
    resampled = df.resample(resample_freq).agg({
        'u': ['mean', 'std', 'count'],
        'v': ['mean', 'std', 'count'],  
        'u_cross': ['mean', 'std', 'count'],
        'v_along': ['mean', 'std', 'count'],
        'magnitude': ['mean', 'std', 'count'],
        'lon': 'mean',
        'lat': 'mean'
    })
    
    # Flatten column names
    resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]
    
    # Reset index to get time as column
    resampled.reset_index(inplace=True)
    
    # Calculate standard errors
    resampled['u_stderr'] = resampled['u_std'] / np.sqrt(resampled['u_count'])
    resampled['v_stderr'] = resampled['v_std'] / np.sqrt(resampled['v_count'])
    resampled['u_cross_stderr'] = resampled['u_cross_std'] / np.sqrt(resampled['u_cross_count'])
    resampled['v_along_stderr'] = resampled['v_along_std'] / np.sqrt(resampled['v_along_count'])
    resampled['magnitude_stderr'] = resampled['magnitude_std'] / np.sqrt(resampled['magnitude_count'])
    
    # Normalize vectors for direction display
    u_mean = resampled['u_mean']
    v_mean = resampled['v_mean']
    magnitude_mean = np.sqrt(u_mean**2 + v_mean**2)
    resampled['u_norm'] = u_mean / (magnitude_mean + 1e-10)
    resampled['v_norm'] = v_mean / (magnitude_mean + 1e-10)
    
    return resampled

@st.cache_data
def load_doppio_single_layer(nc_path, layer_name, resample_freq='W'):
    """Load and preprocess a single DOPPIO layer (surface or bottom)"""
    ds = xr.open_dataset(nc_path)
    
    # Create dataframe
    df = pd.DataFrame({
        'time': pd.to_datetime(ds.time.values),
        'u': ds.u.values * 100,  # Convert to cm/s
        'v': ds.v.values * 100,  # Convert to cm/s
        'lon': ds.lon.values if ds.lon.dims else [ds.lon.values] * len(ds.time),
        'lat': ds.lat.values if ds.lat.dims else [ds.lat.values] * len(ds.time)
    })
    
    # Add along-shelf and cross-shelf components
    rot_angle = 360 - 37  # degrees
    uv_complex = df['u'] + 1j * df['v']
    uv_rotated = uv_complex * np.exp(-1j * np.deg2rad(rot_angle))
    
    df['u_cross'] = np.real(uv_rotated)  # Cross-shelf (Ur)
    df['v_along'] = np.imag(uv_rotated)  # Along-shelf (Vr)
    df['magnitude'] = np.sqrt(df['u']**2 + df['v']**2)
    
    # Resample to specified frequency
    df.set_index('time', inplace=True)
    resampled = df.resample(resample_freq).agg({
        'u': ['mean', 'std', 'count'],
        'v': ['mean', 'std', 'count'],  
        'u_cross': ['mean', 'std', 'count'],
        'v_along': ['mean', 'std', 'count'],
        'magnitude': ['mean', 'std', 'count'],
        'lon': 'mean',
        'lat': 'mean'
    })
    
    # Flatten column names
    resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]
    resampled.reset_index(inplace=True)
    
    # Calculate standard errors
    resampled['u_stderr'] = resampled['u_std'] / np.sqrt(resampled['u_count'])
    resampled['v_stderr'] = resampled['v_std'] / np.sqrt(resampled['v_count'])
    resampled['u_cross_stderr'] = resampled['u_cross_std'] / np.sqrt(resampled['u_cross_count'])
    resampled['v_along_stderr'] = resampled['v_along_std'] / np.sqrt(resampled['v_along_count'])
    resampled['magnitude_stderr'] = resampled['magnitude_std'] / np.sqrt(resampled['magnitude_count'])
    
    # Normalize vectors
    u_mean = resampled['u_mean']
    v_mean = resampled['v_mean']
    magnitude_mean = np.sqrt(u_mean**2 + v_mean**2)
    resampled['u_norm'] = u_mean / (magnitude_mean + 1e-10)
    resampled['v_norm'] = v_mean / (magnitude_mean + 1e-10)
    
    return resampled

# ============================================================================
# SHARED PLOTTING FUNCTIONS
# ============================================================================

@time_it("create_timeseries_base")
def create_timeseries_base(df, time_range=None, freq_label="Monthly", ylim=None):
    """Create base time series plots - works for both monthly and weekly"""
    if time_range is not None:
        df_plot = df[(df['time'] >= time_range[0]) & (df['time'] <= time_range[1])]
    else:
        df_plot = df
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[f'{freq_label} Along-shelf Velocity', f'{freq_label} Cross-shelf Velocity'],
        vertical_spacing=0.1
    )
    
    # Along-shelf plot with error bars
    fig.add_trace(
        go.Scatter(
            x=df_plot['time'],
            y=df_plot['v_along_mean'],
            mode='lines+markers',
            name='Along-shelf (Vr)',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            error_y=dict(
                type='data',
                array=df_plot['v_along_stderr'],
                visible=True,
                color='rgba(0,0,255,0.3)',
                thickness=1
            ),
            hovertemplate='%{x}<br>%{y:.2f} ± %{error_y.array:.2f} cm/s<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Cross-shelf plot with error bars
    fig.add_trace(
        go.Scatter(
            x=df_plot['time'],
            y=df_plot['u_cross_mean'],
            mode='lines+markers',
            name='Cross-shelf (Ur)',
            line=dict(color='red', width=2),
            marker=dict(size=4),
            error_y=dict(
                type='data',
                array=df_plot['u_cross_stderr'],
                visible=True,
                color='rgba(255,0,0,0.3)',
                thickness=1
            ),
            hovertemplate='%{x}<br>%{y:.2f} ± %{error_y.array:.2f} cm/s<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add overall means as horizontal lines
    overall_along = df_plot['v_along_mean'].mean()
    overall_cross = df_plot['u_cross_mean'].mean()
    
    fig.add_hline(y=overall_along, line_dash="dash", line_color="blue", 
                  line_width=1, opacity=0.7, row=1, col=1)
    fig.add_hline(y=overall_cross, line_dash="dash", line_color="red", 
                  line_width=1, opacity=0.7, row=2, col=1)
    
    # Layout updates
    fig.update_layout(
        height=380,
        showlegend=False,
        title_text=" ",
        title_x=0.5,
        uirevision='constant',
        hovermode='x unified',
        margin=dict(t=40, b=20, l=40, r=20)
    )
    
    fig.update_yaxes(title_text="Velocity (cm/s)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (cm/s)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    # Set y-axis limits if provided
    if ylim is not None:
        fig.update_yaxes(range=ylim, row=1, col=1)
        fig.update_yaxes(range=ylim, row=2, col=1)
    
    # Add annotations for overall means
    fig.add_annotation(x=0.02, y=0.95, xref="paper", yref="paper",
                      text=f"Mean: {overall_along:.1f} cm/s", 
                      showarrow=False, font=dict(size=10, color="blue"),
                      bgcolor="rgba(255,255,255,0.8)", bordercolor="blue", borderwidth=1)
    
    fig.add_annotation(x=0.02, y=0.45, xref="paper", yref="paper",
                      text=f"Mean: {overall_cross:.1f} cm/s", 
                      showarrow=False, font=dict(size=10, color="red"),
                      bgcolor="rgba(255,255,255,0.8)", bordercolor="red", borderwidth=1)
    
    return fig

@time_it("update_selection")
def update_selection(base_fig, selected_time):
    """Add selection line to base figure"""
    import copy
    
    # Deep copy to avoid modifying cached figure
    fig = copy.deepcopy(base_fig)
    
    # Add vertical lines for current selection
    for i in range(1, 3):  # Both subplots
        fig.add_vline(
            x=selected_time,
            line_dash="solid",
            line_color="orange",
            line_width=3,
            row=i, col=1
        )
    
    return fig

@st.cache_data
@time_it("create_base_map")
def create_base_map(lat, lon):
    """Create base map once and cache it"""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from io import BytesIO
    
    # Create figure with PlateCarree projection for consistency
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Set extent around the location
    extent = [lon - 1.5, lon + 1.5, lat - 1.0, lat + 1.0]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.7)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=False, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot location point (static)
    ax.scatter(lon, lat, color='red', s=100, zorder=5, 
               transform=ccrs.PlateCarree())
    
    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf.getvalue()

@time_it("create_arrow_overlay")
def create_arrow_overlay(lat, lon, u_norm, v_norm, magnitude, timestamp, stderr, 
                        freq_label="Monthly", arrow_color='darkblue', fig_size=(10, 8)):
    """Create arrow overlay with uncertainty info"""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from io import BytesIO
    
    dpi = 100
    plt.switch_backend('Agg')
    
    # Create transparent figure with same projection as base map
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi, 
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    extent = [lon - 1.5, lon + 1.5, lat - 1.0, lat + 1.0]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Make background transparent
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    ax.axis('off')
    
    # Arrow parameters based on frequency
    scale = 6 if freq_label == "Monthly" else 8
    headwidth = 7 if freq_label == "Monthly" else 6
    
    quiver_kwargs = {
        'scale': scale,
        'headwidth': headwidth,
        'headlength': 6,
        'headaxislength': 6,
        'minshaft': 2,
        'minlength': 6,
        'width': 0.02,
        'color': arrow_color,
        'alpha': 0.9,
        'rasterized': True,
        'pivot': 'tail',
        'transform': ccrs.PlateCarree()
    }
    
    ax.quiver(lon, lat, u_norm, v_norm, **quiver_kwargs)
    
    # Text with uncertainty
    if freq_label == "Monthly":
        time_str = timestamp.strftime('%Y-%m')
    else:  # Weekly
        time_str = timestamp.strftime('%Y-%m-%d')
    
    ax.text(0.5, 0.95, f'{time_str}\n± {stderr:.1f} cm/s',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                     edgecolor=arrow_color, linewidth=1))
    
    plt.tight_layout(pad=0.1)
    
    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                transparent=True, facecolor='none')
    buf.seek(0)
    plt.close()
    
    return buf

@time_it("create_map_plot")
def create_map_plot(df, selected_idx, base_map_bytes, freq_label="Monthly", arrow_color='darkblue'):
    """Combine base map with arrow overlay"""
    from PIL import Image
    from io import BytesIO
    
    row = df.iloc[selected_idx]
    
    # Convert to regular Python types
    lat = float(row['lat_mean'])
    lon = float(row['lon_mean'])
    u_norm = float(row['u_norm'])
    v_norm = float(row['v_norm'])
    magnitude = float(row['magnitude_mean'])
    stderr = float(row['magnitude_stderr'])
    
    # Load base image and get its size
    base_img = Image.open(BytesIO(base_map_bytes))
    base_width, base_height = base_img.size
    
    # Create arrow overlay
    arrow_buf = create_arrow_overlay(lat, lon, u_norm, v_norm, magnitude, 
                                   row['time'], stderr, freq_label, arrow_color,
                                   fig_size=(base_width/100, base_height/100))
    
    # Load and resize arrow image if needed
    arrow_img = Image.open(arrow_buf)
    if arrow_img.size != base_img.size:
        arrow_img = arrow_img.resize(base_img.size, Image.Resampling.LANCZOS)

    # Composite the images
    base_rgba = base_img.convert('RGBA')
    arrow_rgba = arrow_img.convert('RGBA')
    
    if base_rgba.size != arrow_rgba.size:
        arrow_rgba = arrow_rgba.resize(base_rgba.size, Image.Resampling.LANCZOS)
    
    combined = Image.alpha_composite(base_rgba, arrow_rgba)
    
    # Convert back to bytes
    final_buf = BytesIO()
    combined.convert('RGB').save(final_buf, format='PNG')
    final_buf.seek(0)
    
    return final_buf

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def show_monthly_mbon_page():
    st.markdown("## 📅 Monthly HFR Surface Current Data")
    
    # File upload or default path
    uploaded_file = st.sidebar.file_uploader(
        "Upload MBON NetCDF file", 
        type=['nc'],
        help="Upload your MBON current dataset",
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
    st.markdown("## 📊 Weekly HFR Surface Current Data")
    
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
    st.markdown("## 🌊 Weekly Surface DOPPIO Model Data")
    
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
    st.markdown("## ⬇️ Weekly Bottom DOPPIO Model Data")
    
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
        selected_idx_filtered = st.sidebar.slider(
            f"Select {freq_label.lower()}",
            min_value=0,
            max_value=len(df_filtered) - 1,
            value=0,
            help=f"Drag to explore {freq_label.lower()} patterns",
            key=f"slider_{data_label.lower().replace(' ', '_')}"
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

            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Along-shelf", f"{current_row['v_along_mean']:.1f} cm/s")
                st.metric("U (East)", f"{current_row['u_mean']:.1f} cm/s")
            with col2b:
                st.metric("Cross-shelf", f"{current_row['u_cross_mean']:.1f} cm/s")
                st.metric("V (North)", f"{current_row['v_mean']:.1f} cm/s")
            with col2c:
                st.metric("Observations", f"{current_row['magnitude_count']:.0f}")

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
    st.markdown("# 🌊 Ocean Current Multi-Viewer")
    
    # Navigation in sidebar
    st.sidebar.markdown("## 📋 Navigation")
    
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