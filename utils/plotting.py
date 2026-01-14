import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy
from utils.ui_components import time_it

@time_it("create_timeseries_base")
def create_timeseries_base(df, time_range=None, freq_label="Monthly", ylim=None):
    """Create base time series plots for velocity data (2-row subplots, no temperature)"""
    if time_range is not None:
        df_plot = df[(df['time'] >= time_range[0]) & (df['time'] <= time_range[1])]
    else:
        df_plot = df

    # Always create 2 rows: along-shelf, cross-shelf
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

    # Add annotations for mean values
    fig.add_annotation(
        x=1.0, y=1.0, xref="x domain", yref="y domain",
        text=f"Mean: {overall_along:.1f} cm/s",
        showarrow=False,
        xanchor='right', yanchor='top',
        bgcolor='rgba(173, 216, 230, 0.8)',
        bordercolor='blue', borderwidth=1,
        font=dict(size=10), row=1, col=1
    )

    fig.add_annotation(
        x=1.0, y=1.0, xref="x2 domain", yref="y2 domain",
        text=f"Mean: {overall_cross:.1f} cm/s",
        showarrow=False,
        xanchor='right', yanchor='top',
        bgcolor='rgba(255, 182, 193, 0.8)',
        bordercolor='red', borderwidth=1,
        font=dict(size=10), row=2, col=1
    )

    # Update layout
    fig.update_yaxes(title_text="Along-shelf Velocity (cm/s)", row=1, col=1)
    fig.update_yaxes(title_text="Cross-shelf Velocity (cm/s)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    if ylim:
        fig.update_yaxes(range=ylim, row=1, col=1)
        fig.update_yaxes(range=ylim, row=2, col=1)

    fig.update_layout(
        height=520,
        showlegend=True,
        hovermode='x unified',
        margin=dict(t=40, b=40, l=60, r=20)
    )

    return fig

@time_it("update_selection")
def update_selection(base_fig, selected_time):
    """Add selection line to base figure"""
    # Deep copy to avoid modifying cached figure
    fig = copy.deepcopy(base_fig)

    # Determine number of subplots based on figure structure
    # Count the number of y-axes to determine subplot count
    num_subplots = len([key for key in fig.layout if key.startswith('yaxis')])

    # Add vertical lines for current selection
    for i in range(1, num_subplots + 1):
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
def create_base_map(lat, lon, ocim2_lat=None, ocim2_lon=None):
    """Create base map once and cache it

    Args:
        lat: Latitude of primary location (MBON3)
        lon: Longitude of primary location (MBON3)
        ocim2_lat: Optional latitude of OCIM2 buoy
        ocim2_lon: Optional longitude of OCIM2 buoy
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from io import BytesIO

    # Create figure with PlateCarree projection for consistency
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set extent to include both locations if OCIM2 is provided
    if ocim2_lat is not None and ocim2_lon is not None:
        # Calculate extent to show both points
        all_lats = [lat, ocim2_lat]
        all_lons = [lon, ocim2_lon]
        center_lat = (max(all_lats) + min(all_lats)) / 2
        center_lon = (max(all_lons) + min(all_lons)) / 2
        extent = [center_lon - 1.5, center_lon + 1.5, center_lat - 1.0, center_lat + 1.0]
    else:
        # Original extent around single location
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

    # Plot primary location point (MBON3 - red)
    ax.scatter(lon, lat, color='red', s=100, zorder=5,
               transform=ccrs.PlateCarree(), label='MBON3')

    # Plot OCIM2 buoy location (lime) if provided
    if ocim2_lat is not None and ocim2_lon is not None:
        ax.scatter(ocim2_lon, ocim2_lat, color='lime', s=100, zorder=5,
                   transform=ccrs.PlateCarree(), label='OCIM2 Buoy')

        # Add legend
        ax.legend(loc='lower center', framealpha=0.9)

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

@time_it("create_temperature_timeseries")
def create_temperature_timeseries(df_ghrsst, df_surface, df_bottom, df_ocim2=None, time_range=None, freq_label="Daily",
                                  show_ghrsst=True, show_surface=True, show_bottom=True, show_ocim2=True):
    """Create single overlay plot with all temperature time series

    Args:
        df_ghrsst: GHRSST satellite SST dataframe
        df_surface: DOPPIO surface temperature dataframe
        df_bottom: DOPPIO bottom temperature dataframe
        df_ocim2: OCIM2 buoy water temperature dataframe (optional)
        time_range: Optional (start, end) tuple
        freq_label: Label for frequency (Daily, Weekly, Monthly)
        show_ghrsst: Whether to show GHRSST satellite trace
        show_surface: Whether to show DOPPIO surface trace
        show_bottom: Whether to show DOPPIO bottom trace
        show_ocim2: Whether to show OCIM2 buoy trace

    Returns:
        Plotly figure with single plot and up to 4 overlaid traces
    """
    # Filter dataframes if time_range is provided
    if time_range is not None:
        df_ghrsst_plot = df_ghrsst[(df_ghrsst['time'] >= time_range[0]) & (df_ghrsst['time'] <= time_range[1])]
        df_surface_plot = df_surface[(df_surface['time'] >= time_range[0]) & (df_surface['time'] <= time_range[1])]
        df_bottom_plot = df_bottom[(df_bottom['time'] >= time_range[0]) & (df_bottom['time'] <= time_range[1])]
        if df_ocim2 is not None:
            df_ocim2_plot = df_ocim2[(df_ocim2['time'] >= time_range[0]) & (df_ocim2['time'] <= time_range[1])]
        else:
            df_ocim2_plot = None
    else:
        df_ghrsst_plot = df_ghrsst
        df_surface_plot = df_surface
        df_bottom_plot = df_bottom
        df_ocim2_plot = df_ocim2

    # Create single plot figure
    fig = go.Figure()

    # Add GHRSST Satellite trace (orange)
    if show_ghrsst and len(df_ghrsst_plot) > 0:
        fig.add_trace(
            go.Scatter(
                x=df_ghrsst_plot['time'],
                y=df_ghrsst_plot['temp_mean'],
                mode='lines+markers',
                name='GHRSST Satellite SST',
                line=dict(color='orange', width=2),
                marker=dict(size=4),
                error_y=dict(
                    type='data',
                    array=df_ghrsst_plot['temp_stderr'],
                    visible=True,
                    color='rgba(255,165,0,0.3)',
                    thickness=1
                ),
                hovertemplate='<b>GHRSST Satellite</b><br>%{x}<br>%{y:.2f} ± %{error_y.array:.2f} °C<extra></extra>'
            )
        )

    # Add DOPPIO Surface trace (blue)
    if show_surface and len(df_surface_plot) > 0:
        fig.add_trace(
            go.Scatter(
                x=df_surface_plot['time'],
                y=df_surface_plot['temp_mean'],
                mode='lines+markers',
                name='DOPPIO Surface',
                line=dict(color='blue', width=2),
                marker=dict(size=4),
                error_y=dict(
                    type='data',
                    array=df_surface_plot['temp_stderr'],
                    visible=True,
                    color='rgba(0,0,255,0.3)',
                    thickness=1
                ),
                hovertemplate='<b>DOPPIO Surface</b><br>%{x}<br>%{y:.2f} ± %{error_y.array:.2f} °C<extra></extra>'
            )
        )

    # Add DOPPIO Bottom trace (red)
    if show_bottom and len(df_bottom_plot) > 0:
        fig.add_trace(
            go.Scatter(
                x=df_bottom_plot['time'],
                y=df_bottom_plot['temp_mean'],
                mode='lines+markers',
                name='DOPPIO Bottom',
                line=dict(color='red', width=2),
                marker=dict(size=4),
                error_y=dict(
                    type='data',
                    array=df_bottom_plot['temp_stderr'],
                    visible=True,
                    color='rgba(255,0,0,0.3)',
                    thickness=1
                ),
                hovertemplate='<b>DOPPIO Bottom</b><br>%{x}<br>%{y:.2f} ± %{error_y.array:.2f} °C<extra></extra>'
            )
        )

    # Add OCIM2 Buoy trace (lime)
    if show_ocim2 and df_ocim2_plot is not None and len(df_ocim2_plot) > 0:
        fig.add_trace(
            go.Scatter(
                x=df_ocim2_plot['time'],
                y=df_ocim2_plot['temp_mean'],
                mode='lines+markers',
                name='OCIM2 Buoy',
                line=dict(color='lime', width=2),
                marker=dict(size=4),
                error_y=dict(
                    type='data',
                    array=df_ocim2_plot['temp_stderr'],
                    visible=True,
                    color='rgba(0,255,0,0.3)',
                    thickness=1
                ),
                hovertemplate='<b>OCIM2 Buoy</b><br>%{x}<br>%{y:.2f} ± %{error_y.array:.2f} °C<extra></extra>'
            )
        )

    # Add mean lines for visible traces
    if show_ghrsst and len(df_ghrsst_plot) > 0:
        overall_ghrsst = df_ghrsst_plot['temp_mean'].mean()
        fig.add_hline(y=overall_ghrsst, line_dash="dash", line_color="orange",
                      line_width=1, opacity=0.5,
                      annotation_text=f"GHRSST Mean: {overall_ghrsst:.1f}°C",
                      annotation_position="right")

    if show_surface and len(df_surface_plot) > 0:
        overall_surface = df_surface_plot['temp_mean'].mean()
        fig.add_hline(y=overall_surface, line_dash="dash", line_color="blue",
                      line_width=1, opacity=0.5,
                      annotation_text=f"Surface Mean: {overall_surface:.1f}°C",
                      annotation_position="right")

    if show_bottom and len(df_bottom_plot) > 0:
        overall_bottom = df_bottom_plot['temp_mean'].mean()
        fig.add_hline(y=overall_bottom, line_dash="dash", line_color="red",
                      line_width=1, opacity=0.5,
                      annotation_text=f"Bottom Mean: {overall_bottom:.1f}°C",
                      annotation_position="right")

    if show_ocim2 and df_ocim2_plot is not None and len(df_ocim2_plot) > 0:
        overall_ocim2 = df_ocim2_plot['temp_mean'].mean()
        fig.add_hline(y=overall_ocim2, line_dash="dash", line_color="lime",
                      line_width=1, opacity=0.5,
                      annotation_text=f"OCIM2 Mean: {overall_ocim2:.1f}°C",
                      annotation_position="right")

        # Set x-axis range to the Doppio bottom temperature data range
    doppio_times = df_bottom_plot['time'].tolist()
    

    if doppio_times:
        xaxis_range = [min(doppio_times), max(doppio_times)]
    else:
        xaxis_range = None

    # Update layout
    fig.update_layout(
        title=f"{freq_label} Temperature Data",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        height=600,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='center',
            yanchor='bottom',
            bgcolor='rgba(255,255,255,0.1)',
            bordercolor='gray',
            borderwidth=1
        ),
        hovermode='x unified',
        margin=dict(t=60, b=40, l=60, r=20)
    )

    # Set x-axis range if we have data
    if xaxis_range:
        fig.update_xaxes(range=xaxis_range)

    return fig
