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
    page_title="Monthly Ocean Current Viewer",
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

@st.cache_data
def load_data(nc_path):
    """Load and preprocess the MBON dataset to monthly means"""
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
    
    # Resample to monthly means with additional statistics
    df.set_index('time', inplace=True)
    monthly = df.resample('M').agg({
        'u': ['mean', 'std', 'count'],
        'v': ['mean', 'std', 'count'],  
        'u_cross': ['mean', 'std', 'count'],
        'v_along': ['mean', 'std', 'count'],
        'magnitude': ['mean', 'std', 'count'],
        'lon': 'mean',
        'lat': 'mean'
    })
    
    # Flatten column names
    monthly.columns = ['_'.join(col).strip() for col in monthly.columns.values]
    
    # Reset index to get time as column
    monthly.reset_index(inplace=True)
    
    # Calculate monthly standard errors
    monthly['u_stderr'] = monthly['u_std'] / np.sqrt(monthly['u_count'])
    monthly['v_stderr'] = monthly['v_std'] / np.sqrt(monthly['v_count'])
    monthly['u_cross_stderr'] = monthly['u_cross_std'] / np.sqrt(monthly['u_cross_count'])
    monthly['v_along_stderr'] = monthly['v_along_std'] / np.sqrt(monthly['v_along_count'])
    monthly['magnitude_stderr'] = monthly['magnitude_std'] / np.sqrt(monthly['magnitude_count'])
    
    # Normalize monthly mean vectors for direction display
    u_mean = monthly['u_mean']
    v_mean = monthly['v_mean']
    magnitude_mean = np.sqrt(u_mean**2 + v_mean**2)
    monthly['u_norm'] = u_mean / (magnitude_mean + 1e-10)
    monthly['v_norm'] = v_mean / (magnitude_mean + 1e-10)
    
    return monthly

@st.cache_data
@time_it("create_monthly_timeseries_cached")
def create_monthly_timeseries_base(df, time_range=None):
    """Create base monthly time series plots - CACHED"""
    if time_range is not None:
        df_plot = df[(df['time'] >= time_range[0]) & (df['time'] <= time_range[1])]
    else:
        df_plot = df
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=['Monthly Along-shelf Velocity', 'Monthly Cross-shelf Velocity'],
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

@time_it("update_monthly_selection")
def update_monthly_selection(base_fig, selected_time):
    """Just add/update the selection line - FAST"""
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
    
    # Set extent around the location - SAME as arrow overlay
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
    
    # Plot location point (static) - SAME COORDINATE SYSTEM
    ax.scatter(lon, lat, color='red', s=100, zorder=5, 
               transform=ccrs.PlateCarree())
    
    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf.getvalue()

@time_it("create_monthly_arrow_overlay")
def create_monthly_arrow_overlay(lat, lon, u_norm, v_norm, magnitude, timestamp, stderr, fig_size=(10, 8)):
    """Create monthly arrow overlay with uncertainty info"""
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
    
    # Monthly arrow - slightly larger and more prominent
    quiver_kwargs = {
        'scale': 6,  # Larger arrows for monthly data
        'headwidth': 7,
        'headlength': 6,
        'headaxislength': 6,
        'minshaft': 2,
        'minlength': 6,
        'width': 0.02,
        'color': 'darkblue',
        'alpha': 0.9,
        'rasterized': True,
        'pivot': 'tail',
        'transform': ccrs.PlateCarree()
    }
    
    ax.quiver(lon, lat, u_norm, v_norm, **quiver_kwargs)
    
    # Monthly data text with uncertainty
    time_str = timestamp.strftime('%Y-%m')
    ax.text(0.5, 0.95, f'{time_str}\n± {stderr:.1f} cm/s',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                     edgecolor='darkblue', linewidth=1))
    
    plt.tight_layout(pad=0.1)
    
    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                transparent=True, facecolor='none')
    buf.seek(0)
    plt.close()
    
    return buf

@time_it("create_monthly_map_plot")
def create_monthly_map_plot(df, selected_idx, base_map_bytes):
    """Combine base map with monthly arrow overlay"""
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
    arrow_buf = create_monthly_arrow_overlay(lat, lon, u_norm, v_norm, magnitude, 
                                           row['time'], stderr,
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

def main():
    st.markdown("# 🌊 Monthly Ocean Current Viewer")
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # File upload or default path
    uploaded_file = st.sidebar.file_uploader(
        "Upload NetCDF file", 
        type=['nc'],
        help="Upload your MBON current dataset"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_data.nc", "wb") as f:
            f.write(uploaded_file.getbuffer())
        nc_path = "temp_data.nc"
    else:
        # Use default path
        default_path = "2007_2024_10_10_MARACOOS_uv_MBON_timeseries_v5_corrected.nc"
        if Path(default_path).exists():
            nc_path = default_path
        else:
            st.error("Please upload a NetCDF file or ensure the default file exists.")
            st.stop()
    
    # Load monthly data
    with st.spinner("Loading and processing monthly data..."):
        df_monthly = load_data(nc_path)
        
        # Create base map once (cached)
        first_row = df_monthly.iloc[0]
        base_map_bytes = create_base_map(float(first_row['lat_mean']), float(first_row['lon_mean']))
    
    # Success message that disappears
    success_placeholder = st.empty()
    # success_placeholder.success(f"✅ Loaded {len(df_monthly)} monthly means from {df_monthly['time'].min().strftime('%Y-%m')} to {df_monthly['time'].max().strftime('%Y-%m')}")
    
    # Clear success message after 3 seconds
    import threading
    def clear_message():
        import time
        time.sleep(3)
        success_placeholder.empty()
    
    threading.Thread(target=clear_message, daemon=True).start()
    
    # Time range selection for zoom
    st.sidebar.subheader("Time Range Selection")
    
    # Full range by default
    full_range = (df_monthly['time'].min(), df_monthly['time'].max())
    
    # Date range selector (year-month level)
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(full_range[0].date(), full_range[1].date()),
        min_value=full_range[0].date(),
        max_value=full_range[1].date()
    )
    
    # Convert back to datetime for filtering
    if len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + timedelta(days=31)  # Month buffer
        
        # Filter dataframe
        df_filtered = df_monthly[(df_monthly['time'] >= start_date) & (df_monthly['time'] < end_date)]
        time_range = (start_date, end_date)
        
        # Recreate base time series if time range changed
        if time_range != getattr(st.session_state, 'last_time_range', None):
            st.session_state.last_time_range = time_range
            base_ts_fig = create_monthly_timeseries_base(df_monthly, time_range)
        else:
            base_ts_fig = st.session_state.get('base_ts_fig', create_monthly_timeseries_base(df_monthly, time_range))
        
        st.session_state.base_ts_fig = base_ts_fig
    else:
        df_filtered = df_monthly
        time_range = None
        
        # Use full dataset base figure
        if not hasattr(st.session_state, 'base_ts_fig_full'):
            st.session_state.base_ts_fig_full = create_monthly_timeseries_base(df_monthly, None)
        base_ts_fig = st.session_state.base_ts_fig_full
    
    # Month slider
    st.sidebar.subheader("Month Selection")
    
    if len(df_filtered) > 0:
        selected_idx_filtered = st.sidebar.slider(
            "Select month",
            min_value=0,
            max_value=len(df_filtered) - 1,
            value=0,
            help="Drag to explore monthly patterns"
        )
        
        # Get the actual index in the full dataset
        selected_time = df_filtered.iloc[selected_idx_filtered]['time']
        selected_idx = df_monthly[df_monthly['time'] == selected_time].index[0]
        
        # Get current row data
        current_row = df_monthly.iloc[selected_idx]
        
        # Display current selection info - more compact
        st.sidebar.info(f"""
        **Current Selection:**
        - **{current_row['time'].strftime('%Y-%m')}** 
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
        
        # Main layout - adjusted proportions
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Monthly time series plots
            ts_fig = update_monthly_selection(base_ts_fig, current_row['time'])
            st.plotly_chart(ts_fig, use_container_width=True)
        
        with col2:
            # Current monthly vector information
            st.markdown("#### Monthly Current Details")

            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Along-shelf", f"{current_row['v_along_mean']:.1f} cm/s", 
                         delta=None)
                st.metric("U (East)", f"{current_row['u_mean']:.1f} cm/s",
                         delta=None)
            with col2b:
                st.metric("Cross-shelf", f"{current_row['u_cross_mean']:.1f} cm/s",
                         delta=None)
                st.metric("V (North)", f"{current_row['v_mean']:.1f} cm/s",
                         delta=None)
            with col2c:
                # st.metric("Average Velocity", f"{current_row['magnitude_mean']:.1f} cm/s",
                        #  delta=None)
                st.metric("Observations", f"{current_row['magnitude_count']:.0f} hrs")

        # Map section - reduced width and better positioned
        col_map1, col_map2, col_map3 = st.columns([1, 2, 1])
        
        with col_map2:
            # st.markdown("#### Monthly Mean Current Vector")
            map_img = create_monthly_map_plot(df_monthly, selected_idx, base_map_bytes)
            st.image(map_img, use_container_width=True)
        
    else:
        st.warning("No data available for the selected time range.")

if __name__ == "__main__":
    main()