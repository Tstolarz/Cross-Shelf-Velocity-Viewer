import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr

@st.cache_data
def load_mbon_data(nc_path, resample_freq='MS'):
    """Load and preprocess the MBON dataset to specified frequency (MS=monthly, W=weekly)"""
    ds = xr.open_dataset(nc_path)

    # Convert to pandas for easier handling
    df = pd.DataFrame({
        'time': pd.to_datetime(ds.time.values),
        'u': ds.u.values * 100,  # Convert to cm/s
        'v': ds.v.values * 100,  # Convert to cm/s
        'lon': ds.lon.values if ds.lon.dims else [ds.lon.values] * len(ds.time),
        'lat': ds.lat.values if ds.lat.dims else [ds.lat.values] * len(ds.time)
    })

    # Add temperature if available
    if 'temp' in ds.variables:
        df['temp'] = ds.temp.values  # Temperature in degrees C

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

    # Build aggregation dictionary dynamically
    agg_dict = {
        'u': ['mean', 'std', 'count'],
        'v': ['mean', 'std', 'count'],
        'u_cross': ['mean', 'std', 'count'],
        'v_along': ['mean', 'std', 'count'],
        'magnitude': ['mean', 'std', 'count'],
        'lon': 'mean',
        'lat': 'mean'
    }

    # Add temperature aggregation if available
    if 'temp' in df.columns:
        agg_dict['temp'] = ['mean', 'std', 'count']

    resampled = df.resample(resample_freq).agg(agg_dict)

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

    # Add temperature standard error if available
    if 'temp_std' in resampled.columns:
        resampled['temp_stderr'] = resampled['temp_std'] / np.sqrt(resampled['temp_count'])

    # Normalize vectors for direction display
    u_mean = resampled['u_mean']
    v_mean = resampled['v_mean']
    magnitude_mean = np.sqrt(u_mean**2 + v_mean**2)
    resampled['u_norm'] = u_mean / (magnitude_mean + 1e-10)
    resampled['v_norm'] = v_mean / (magnitude_mean + 1e-10)

    return resampled

@st.cache_data
def load_doppio_single_layer(nc_path, layer_name, resample_freq='W'):
    """Load and preprocess a single DOPPIO layer (surface or bottom)

    Args:
        resample_freq: 'D' (daily), 'W' (weekly), 'MS' (monthly-start)
    """
    ds = xr.open_dataset(nc_path)

    # Create dataframe
    df = pd.DataFrame({
        'time': pd.to_datetime(ds.time.values),
        'u': ds.u.values * 100,  # Convert to cm/s
        'v': ds.v.values * 100,  # Convert to cm/s
        'lon': ds.lon.values if ds.lon.dims else [ds.lon.values] * len(ds.time),
        'lat': ds.lat.values if ds.lat.dims else [ds.lat.values] * len(ds.time)
    })

    # Add temperature if available
    if 'temp' in ds.variables:
        df['temp'] = ds.temp.values  # Temperature in degrees C

    # Add along-shelf and cross-shelf components
    rot_angle = 360 - 37  # degrees
    uv_complex = df['u'] + 1j * df['v']
    uv_rotated = uv_complex * np.exp(-1j * np.deg2rad(rot_angle))

    df['u_cross'] = np.real(uv_rotated)  # Cross-shelf (Ur)
    df['v_along'] = np.imag(uv_rotated)  # Along-shelf (Vr)
    df['magnitude'] = np.sqrt(df['u']**2 + df['v']**2)

    # Resample to specified frequency
    df.set_index('time', inplace=True)

    # Build aggregation dictionary dynamically
    agg_dict = {
        'u': ['mean', 'std', 'count'],
        'v': ['mean', 'std', 'count'],
        'u_cross': ['mean', 'std', 'count'],
        'v_along': ['mean', 'std', 'count'],
        'magnitude': ['mean', 'std', 'count'],
        'lon': 'mean',
        'lat': 'mean'
    }

    # Add temperature aggregation if available
    if 'temp' in df.columns:
        agg_dict['temp'] = ['mean', 'std', 'count']

    resampled = df.resample(resample_freq).agg(agg_dict)

    # Flatten column names
    resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]
    resampled.reset_index(inplace=True)

    # Calculate standard errors
    resampled['u_stderr'] = resampled['u_std'] / np.sqrt(resampled['u_count'])
    resampled['v_stderr'] = resampled['v_std'] / np.sqrt(resampled['v_count'])
    resampled['u_cross_stderr'] = resampled['u_cross_std'] / np.sqrt(resampled['u_cross_count'])
    resampled['v_along_stderr'] = resampled['v_along_std'] / np.sqrt(resampled['v_along_count'])
    resampled['magnitude_stderr'] = resampled['magnitude_std'] / np.sqrt(resampled['magnitude_count'])

    # Add temperature standard error if available
    if 'temp_std' in resampled.columns:
        resampled['temp_stderr'] = resampled['temp_std'] / np.sqrt(resampled['temp_count'])

    # Normalize vectors
    u_mean = resampled['u_mean']
    v_mean = resampled['v_mean']
    magnitude_mean = np.sqrt(u_mean**2 + v_mean**2)
    resampled['u_norm'] = u_mean / (magnitude_mean + 1e-10)
    resampled['v_norm'] = v_mean / (magnitude_mean + 1e-10)

    return resampled

@st.cache_data
def load_ghrsst_data(nc_path):
    """Load pre-aggregated GHRSST satellite SST data

    Args:
        nc_path: Path to GHRSST NetCDF file (daily, weekly, or monthly)

    Returns:
        DataFrame with columns: time, temp_mean, temp_std, temp_count,
                                temp_stderr, lat_mean, lon_mean
    """
    ds = xr.open_dataset(nc_path)

    # Create dataframe from GHRSST data
    df = pd.DataFrame({
        'time': pd.to_datetime(ds.time.values),
        'temp': ds.temp.values,  # Temperature in degrees C
        'lon': float(ds.longitude.values),
        'lat': float(ds.latitude.values)
    })

    df.set_index('time', inplace=True)

    # Since GHRSST files are pre-aggregated, we still need to calculate statistics
    # for consistency with other datasets. For pre-aggregated data, std and count
    # will represent the aggregated period.
    # We'll treat each point as a single observation with no within-period variation
    df['temp_mean'] = df['temp']
    df['temp_std'] = 0.0  # Pre-aggregated data, no within-period std available
    df['temp_count'] = 1.0  # Each point is one aggregated value
    df['temp_stderr'] = 0.0  # No stderr for pre-aggregated data
    df['lon_mean'] = df['lon']
    df['lat_mean'] = df['lat']

    # Reset index
    df.reset_index(inplace=True)

    # Keep only the columns we need
    df = df[['time', 'temp_mean', 'temp_std', 'temp_count', 'temp_stderr', 'lat_mean', 'lon_mean']]

    return df

@st.cache_data
def load_ocim2_data(csv_path, resample_freq='D'):
    """Load OCIM2 buoy water temperature data

    Args:
        csv_path: Path to processed OCIM2 hourly CSV file
        resample_freq: 'D' (daily), 'W' (weekly), 'MS' (monthly-start)

    Returns:
        DataFrame with columns: time, temp_mean, temp_std, temp_count,
                                temp_stderr, lat_mean, lon_mean
    """
    # Load hourly buoy data
    df = pd.read_csv(csv_path, parse_dates=['datetime'], index_col='datetime')

    # Extract water temperature column (WTMP in NDBC data)
    if 'WTMP' not in df.columns:
        raise ValueError("WTMP (water temperature) column not found in OCIM2 data")

    # Keep only temperature column
    df_temp = df[['WTMP']].copy()

    # Remove NaN values
    df_temp = df_temp.dropna()

    # Resample to desired frequency
    agg_dict = {
        'WTMP': ['mean', 'std', 'count']
    }

    resampled = df_temp.resample(resample_freq).agg(agg_dict)

    # Flatten column names
    resampled.columns = ['temp_mean', 'temp_std', 'temp_count']
    resampled.reset_index(inplace=True)
    resampled.rename(columns={'datetime': 'time'}, inplace=True)

    # Calculate standard error
    resampled['temp_stderr'] = resampled['temp_std'] / np.sqrt(resampled['temp_count'])

    # Add location (OCIM2 buoy in Delaware Bay)
    resampled['lat_mean'] = 38.328
    resampled['lon_mean'] = -75.091

    # Keep only rows with data
    resampled = resampled[resampled['temp_count'] > 0].copy()

    return resampled
