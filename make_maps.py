from pathlib import Path

import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import verde as vd
#from tqdm import tqdm
import pyproj

try:
    from .parse_tec import get_tec_data
except ImportError:
    from parse_tec import get_tec_data



def plot_map(
    date,
    dataset_path,
    target="vTEC",
    region=[-75, -50, -55, -18],
    spacing=2,
    maxdist=3,
    cmap="jet",
    v_min=None,
    v_max=None,
    colormap_label=None,
    operation="mean",
    time_delta="35s",
):
    """
    Make a TEC (Total Electron Content) or ROTI map figure for the given date.

    Parameters
    ----------
    date : pd.Timestamp
        Date for which to generate and save the figure
    datast_path : str | Path
        Path of dataset
    target : str
        Data variable to plot (e.g., 'vTEC', 'roti')
    region : list
        Map region as [west, east, south, north] in degrees
    spacing : float
        Grid spacing in degrees
    maxdist : float
        The maximum distance that a point can be from the closest data point
    cmap : str
        Matplotlib colormap name
    v_min : float
        Minimum value for colorbar scale
    v_max : float
        Maximum value for colorbar scale
    colormap_label : str
        Label for the colorbar
    operation : str, optional
        Reduction operation ('mean' or 'q90'), default is 'mean'
    time_delta: str, optional
        Time delta to load data for. Default is "35s"
    """
    print(f"Processing {date}...")

    # 1. Load Data
    try:
        data = get_tec_data(
            dataset_path, "all", date, date + pd.to_timedelta(time_delta)
        ).dropna()
    except Exception as e:
        print(f"Skipping {date}: No data or error ({e})")
        return None

    # Skip empty dataframes to prevent errors
    if data.empty:
        print(f"Skipping {date}: No data available")
        return None

    coordinates = (data.lon.values, data.lat.values)

    # 2. Interpolation / Gridding
    # Use a Mercator projection for our Cartesian gridder
    projection = pyproj.Proj(proj="merc", lat_ts=data.lat.mean())

    # Chain blocked mean and spline
    if operation == "q90":
        first_operation = (
            "q90",
            vd.BlockReduce(lambda x: np.quantile(x, q=0.9), spacing=spacing * 111e3),
        )
    elif operation == "mean":
        first_operation = ("blockmean", vd.BlockMean(spacing=spacing * 111e3))
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    chain = vd.Chain(
        [
            first_operation,
            ("spline", vd.Spline(damping=1e-10)),
        ]
    )

    # Fit the model
    # Note: We skip train/test split here to speed up the loop for plotting
    # unless you explicitly need the score printed for every frame.
    chain.fit(projection(*coordinates), data[target])

    # Create grid using the MANUAL_REGION
    grid_full = chain.grid(
        region=region,
        spacing=spacing,
        projection=projection,
        dims=["latitude", "longitude"],
        data_names="vTEC",
    )

    # Masking
    grid = vd.distance_mask(
        coordinates, maxdist=maxdist * spacing * 111e3, grid=grid_full, projection=projection
    )

    # 3. Plotting
    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.Mercator())

    # Add timestamp to title so we know which frame is which in the video
    ax.set_title(f"vTEC: {date}")

    # Plot original data points
    ax.plot(*coordinates, ".k", markersize=1, transform=ccrs.PlateCarree())

    # Plot the grid WITH FIXED SCALES
    tmp = grid.vTEC.plot.pcolormesh(
        ax=ax,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        vmin=v_min,  # <--- Fixes the lower limit (e.g., 0)
        vmax=v_max,  # <--- Fixes the upper limit (e.g., 60)
    )

    # Colorbar needs to know the mappable 'tmp'
    plt.colorbar(tmp).set_label(colormap_label)

    # Fixed Map Extent
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # Map Features
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=1, linestyle="-")

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(f"UT {date}")


    return ax 

def save_map(date,
    target="vTEC",
    region=[-75, -50, -55, -18],
    spacing=2,
    cmap="jet",
    v_min=None,
    v_max=None,
    colormap_label=None,
    dataset_path="TEC_dataset",
    operation="mean",
    folder_path=None,
    time_delta="35s",
    dpi=300,
):
    """
    Save a TEC (Total Electron Content) or ROTI map figure for the given date.

    Parameters
    ----------
    date : pd.Timestamp
        Date for which to generate and save the figure
    target : str
        Data variable to plot (e.g., 'vTEC', 'roti')
    region : list
        Map region as [west, east, south, north] in degrees
    spacing : float
        Grid spacing in degrees
    cmap : str
        Matplotlib colormap name
    v_min : float
        Minimum value for colorbar scale
    v_max : float
        Maximum value for colorbar scale
    colormap_label : str
        Label for the colorbar
    operation : str, optional
        Reduction operation ('mean' or 'q90'), default is 'mean'
    folder_path : Path, optional
        Directory to save the figure. If None, uses 'figures/{target}/{freq}/'
    time_delta: str, optional
        Time delta to load data for. Default is "35s"
    """

    ax = plot_map(target,
    region,
    spacing,
    cmap,
    v_min,
    v_max,
    colormap_label,
    dataset_path,
    operation,
    time_delta)

    # Save
    if folder_path is None:
        folder_path = Path(f"figures/{target}/")
    elif isinstance(folder_path, str):
        folder_path = Path(folder_path)
    folder_path.mkdir(exist_ok=True, parents=True)

    filename = (
        f"{target}map_{operation}_{str(date).replace(':', '.').replace(' ', '_')}.png"
    )


    plt.savefig(folder_path / filename, dpi=dpi)  # dpi=100 is usually enough for video

    # 4. CRITICAL: Close the figure to free memory
    plt.close()