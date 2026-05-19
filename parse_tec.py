# %%
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# %%


def parse_filename(path):
    """
    Extracts station metadata from a specific filename format.

    Assumes the filename follows Format 3 with fixed width format
    (e.g., 'tucuF3_316.25A').

    Parameters
    ----------
    path : str or pathlib.Path
        The file path or filename to process.

    Returns
    -------
    station : str
        The 4-character station identifier.
    doy : int
        The Day of Year (1-366) extracted from the filename.
    year : int
        The 4-digit year. This currently assumes the 21st century
        (prepending '20' to the 2-digit year found in the extension).

    Examples
    --------
    >>> station, doy, year = scrape_filename("tucuF3_316.25A")
    >>> print(station, doy, year)
    'tucu', 316, 2025
    """
    path = Path(path)

    station = path.name[:4]
    doy = int(path.name[7:10])
    year = int("20" + path.name[-3:-1])
    return station, doy, year


def parse_filename_gopi(path):
    """
    Extracts station metadata from Gopi .Cmn filenames.

    Expected format: station + DOY + '-' + YYYY-MM-DD (e.g., 'jbal130-2024-05-09.Cmn').

    Parameters
    ----------
    path : str or pathlib.Path
        The file path or filename to process.

    Returns
    -------
    station : str
        The 4-character station identifier.
    date : pd.Timestamp
        Date parsed from the filename (UTC date, no time).
    """
    path = Path(path)
    name = path.stem

    station = name[:4]
    parts = name.split("-")
    if len(parts) >= 4:
        date_str = "-".join(parts[1:4])
        date = pd.Timestamp(date_str)
    else:
        raise ValueError(f"Unsupported Gopi filename format: {path.name}")

    return station, date


def parse_file_ciraolo_f3(path, web_version=False):
    """
    Parses a GNSS/Satellite data file into a Pandas DataFrame with a time index.

    The function reads a whitespace-separated file, cleans numeric columns
    (handling European decimal commas), and constructs a precise timestamp
    based on the filename's year/DOY and the file's seconds-of-day column.

    Parameters
    ----------
    path : str or pathlib.Path
        The location of the file to parse.

    web_version: bool
        Web version of TEC that includes header

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by 'Time' (pd.Timestamp).
        Columns include: 'PRN', 'PP_Az', 'El', 'Lon', 'Lat', 'Slant', 'vTEC', 'station'.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the filename format causes scrape_filename to fail.
    """
    columns = ["Time", "PRN", "PP_Az", "elevation", "lon", "lat", "sTEC", "vTEC"]
    relevant_columns = ["station", "PRN", "lon", "lat", "PP_Az", "elevation", "sTEC", "vTEC"]
    # List of columns that MUST be numbers
    numeric_cols = ["PP_Az", "elevation", "lon", "lat", "sTEC", "vTEC"]

    station, doy, year = parse_filename(path)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        names=columns,
        decimal=",",
        skiprows=5 if web_version else 0, # skip header for web version
    )

    # fix possible fails in float conversion
    for col in numeric_cols:
        # Check if the column is NOT numeric (i.e., it's stuck as 'object' string)
        if df[col].dtype == "object":
            # Replace commas with dots explicitly (just in case read_csv missed them)
            df[col] = df[col].astype(str).str.replace(",", ".")

            # Coerce: Turn "Bad Data" into NaN (Not a Number) instead of crashing
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Negative TEC values do not exist
    # if len(df.loc[df.vTEC < 0]) > 0:
    # print(f"Negative TEC values found in {path} file")
    df.loc[df.vTEC < 0, "vTEC"] = pd.NA

    # 999.99 TEC is not valid TEC
    df.loc[df.vTEC > 999, "vTEC"] = pd.NA

    # compute datetime from doy and seconds
    base_date = pd.Timestamp(f"{year}-01-01")
    doy_offset = pd.to_timedelta(doy - 1, unit="D")
    df["Time"] = base_date + doy_offset + pd.to_timedelta(df["Time"], unit="s")
    df["station"] = station

    return df.set_index("Time").loc[:, relevant_columns]


def parse_file_gopi(path):
    """
    Parses a Gopi .Cmn GNSS file into a DataFrame with the Ciraolo schema.

    This reads the data table from the .Cmn file, ignores S4, drops negative
    Time rows, and constructs timestamps using the date encoded in the filename
    plus the decimal-hour Time column rounded to the nearest second.

    Parameters
    ----------
    path : str or pathlib.Path
        The location of the .Cmn file to parse.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by 'Time' (pd.Timestamp).
        Columns: 'station', 'PRN', 'lon', 'lat', 'sTEC', 'vTEC'.
    """
    columns = [
        "MJdatet",
        "Time",
        "PRN",
        "PP_Az",
        "elevation",
        "lat",
        "lon",
        "sTEC",
        "vTEC",
        "S4",
    ]
    relevant_columns = ["station", "PRN", "lon", "lat", "PP_Az", "elevation", "sTEC", "vTEC"]
    numeric_cols = ["PP_Az", "elevation", "lon", "lat", "sTEC", "vTEC", "Time"]

    station, base_date = parse_filename_gopi(path)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        names=columns,
        skiprows=5,
    )

    for col in numeric_cols:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.replace(",", ".")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[df.vTEC < 0, "vTEC"] = pd.NA
    df.loc[df.vTEC > 999, "vTEC"] = pd.NA

    df = df.loc[df["Time"] >= 0]
    hours = df["Time"].fillna(0)
    seconds = (hours * 3600.0).round().astype("Int64")
    df["Time"] = base_date + pd.to_timedelta(seconds, unit="s")
    df["station"] = station

    return df.set_index("Time").loc[:, relevant_columns]


def parse_file(path, web_version=False, format="auto"):
    """
    Parses a GNSS/Satellite data file with format selection.

    Parameters
    ----------
    path : str or pathlib.Path
        The location of the file to parse.
    web_version : bool
        Only applies to Ciraolo F3 parsing.
    format : str
        One of "auto", "ciraolo_f3", or "gopi".
    """
    format = format.lower()
    if format == "auto":
        suffix = Path(path).suffix.lower()
        name = Path(path).name.lower()
        if suffix == ".cmn":
            format = "gopi"
        elif name.endswith("a"):
            format = "ciraolo_f3"
        else:
            raise ValueError(f"Unsupported file extension for auto format: {Path(path).name}")

    if format == "ciraolo_f3":
        return parse_file_ciraolo_f3(path, web_version=web_version)
    if format == "gopi":
        return parse_file_gopi(path)

    raise ValueError(f"Invalid format: {format}")


def convert_to_parquet(
    source_folder,
    output_folder,
    format="auto",
    freq="30s",
    overwrite=False,
    compute_roti=True,
    web_version=False,
    **roti_kwargs,
):
    """
    Converts GNSS data to Parquet, skipping files that already exist.

    Parameters
    ----------
    source_folder : str or pathlib.Path
        The directory containing the raw text/CSV files (e.g., .25A files).
    output_folder : str or pathlib.Path
        The root directory where the partitioned Parquet dataset will be created.
    format : str
        One of "auto", "ciraolo_f3", or "gopi".
    freq : str
        Frequency for resampling the data. Default is "30s". This is passed to
        the `resample` method of Pandas.
    overwrite : bool
        If True, re-processes files even if the Parquet file exists.
        If False (default), skips files that have already been processed.
    compute_roti : bool
        If True, computes ROT and ROTI columns for each file. Default is True.
    **roti_kwargs : dict
        Additional keyword arguments passed to the `compute_rot_and_roti` function.

    Returns
    -------
    None
        Files are written directly to disk.

    Notes
    -----
    - This function uses the 'station' column to create folder partitions
      and subsequently drops that column from the individual files to
      save space (schema optimization).
    - Files failing to parse are skipped, and the error is printed.
    """
    source_path = Path(source_folder)
    output_path = Path(output_folder)
    format = format.lower()

    if format not in {"auto", "ciraolo_f3", "gopi"}:
        raise ValueError(f"Invalid format: {format}")

    all_files = [f for f in source_path.iterdir() if f.is_file()]
    gopi_files = [f for f in all_files if f.suffix.lower() == ".cmn"]
    ciraolo_files = [f for f in all_files if f.name.lower().endswith("a")]

    if format == "auto":
        if gopi_files and ciraolo_files:
            raise ValueError(
                "Mixed .Cmn and .A files found. Set format to 'gopi' or 'ciraolo_f3'."
            )
        if gopi_files:
            format = "gopi"
            files = gopi_files
        elif ciraolo_files:
            format = "ciraolo_f3"
            files = ciraolo_files
        else:
            print("Scanning 0 files...")
            return
    elif format == "gopi":
        files = gopi_files
    else:
        files = ciraolo_files

    if not files:
        print("Scanning 0 files...")
        return

    print(f"Scanning {len(files)} files...")

    # We use a counter to see how many were actually processed vs skipped
    processed_count = 0

    for file in tqdm(files):
        try:
            # --- STEP 1: Calculate the Target Path FIRST ---
            # We don't need to read the CSV content to know where it goes.
            # We just need the filename.
            if format == "gopi":
                station, date = parse_filename_gopi(file)
                year = date.year
            else:
                station, doy, year = parse_filename(file)

            partition_dir = output_path / f"station={station}" / f"year={year}"
            save_name = partition_dir / f"{file.stem}.parquet"

            # --- STEP 2: The Check ---
            if save_name.exists() and not overwrite:
                # File exists and we are not forcing an update -> Skip
                continue

            # --- STEP 3: File parsing ---
            # If we are here, the file doesn't exist (or we are overwriting)

            # Create dir if it's the first time for this station/year
            partition_dir.mkdir(parents=True, exist_ok=True)

            df = parse_file(file, web_version=web_version, format=format)
            # Resample to fixed frequency
            df = (
                df.groupby(["PRN", "station"])
                .resample(freq, include_groups=False)
                .asfreq()
            )
            df = df.reset_index(level=["PRN", "station"])
            if compute_roti:
                # roti computation
                df = compute_rot_and_roti(df, **roti_kwargs)

            # Optimization: Drop 'station' column
            if "station" in df.columns:
                df = df.drop(columns=["station"])

            df.to_parquet(save_name)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    print(
        f"Job Complete. Processed {processed_count} new files (Skipped {len(files) - processed_count})."
    )


def compute_rot_and_roti(
    df: pd.DataFrame,
    target: str = "sTEC",
    max_delta_time: float = 1.0,
    roti_window: str = "5min",
    rot_treshold: float = 10.0,
) -> pd.DataFrame:
    """
    Computes Rate of TEC (ROT) and Rate of TEC Index (ROTI) for GNSS data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a DatetimeIndex and columns ["PRN", "station", target].
    target : str
        The TEC column name. Default "sTEC".
    max_delta_time : float
        Max allowed gap in minutes. Default 1.0.
    roti_window : str
        Rolling window size. Default "5min".
    rot_treshold : float
        Max Abs value in TECu/min. Values greater that this are considered cycle slips
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'rot' and 'roti' columns.
    """
    # 1. Sort Data: Essential for .diff() to work logically
    # We rely on the existing index being the time.
    df = df.sort_values(by=["PRN", "station", df.index.name or "index"])

    # 2. Create a temporary time column
    # We need this because we cannot run .transform() directly on the index
    # if we want to group by PRN/Station simultaneously.
    df["_temp_time"] = df.index

    # 3. Define the grouper
    g = df.groupby(["PRN", "station"])

    # 4. Calculate Differentials using transform()
    # keeps the original index alignment

    # Calculate dTEC
    d_tec = g[target].transform(lambda x: x.diff())

    # Calculate dt in minutes
    dt_minutes = g["_temp_time"].transform(lambda x: x.diff().dt.total_seconds() / 60.0)

    # 5. Compute ROT
    df["rot"] = d_tec / dt_minutes

    # 6. Filter Invalid Data
    # Remove large gaps or negative time diffs (reordered packets)
    mask_invalid = (dt_minutes > max_delta_time) | (dt_minutes <= 0)
    mask_cycle_slips = df.rot.abs() > rot_treshold
    df.loc[mask_invalid | mask_cycle_slips, "rot"] = np.nan

    # Clean up infinite values (division by zero dt)
    df["rot"] = df["rot"].replace([np.inf, -np.inf], np.nan)

    # 7. Compute ROTI
    df["roti"] = g["rot"].transform(
        lambda x: x.rolling(window=roti_window, center=True, min_periods=6).std()
    )

    # 8. Cleanup
    df = df.drop(columns=["_temp_time"])

    return df


def get_tec_data(
    dataset_path, stations="all", start_date=None, end_date=None, include_end_date=False
):
    """
    Loads TEC data from parquet dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the root folder (e.g., 'GNSS_Dataset')
    stations : str, list[str] or "all", optional
        - "all" (default): Loads data from ALL stations (scans all folders).
        - "tucu": Loads a single station.
        - ["tucu", "cord"]: Loads a specific list of stations.
    start_date : str or datetime, optional
        Filter for data on or after this date.
    end_date : str or datetime, optional
        Filter for data on or before this date.
    include_end_date : bool, optional
        If True, includes data on the end_date (<=). Default is False (<).
    """

    filters = []

    # 1. Station Filter Logic
    # We only add a filter if the user requests specific stations.
    # If stations is "all" or None, we skip this block, and PyArrow reads everything.
    if stations is not None and stations != "all":
        if isinstance(stations, str):
            stations = [stations]  # Convert single string to list

        # Add the filter: only read folders matching these names
        filters.append(("station", "in", stations))

    # 2. Date Filters
    if start_date:
        filters.append(("Time", ">=", pd.Timestamp(start_date)))
    if end_date:
        operation = "<=" if include_end_date else "<"
        filters.append(("Time", operation, pd.Timestamp(end_date)))

    # If filters is empty, PyArrow reads everything (behavior for "all")
    filter_arg = filters if filters else None

    df = pd.read_parquet(dataset_path, filters=filter_arg, engine="pyarrow")

    return df


def list_available_stations(dataset_path):
    """
    Helper to see what stations are in the dataset without loading data.
    """
    p = Path(dataset_path)
    # Looks for folders starting with "station="
    stations = [
        x.name.split("=")[1]
        for x in p.iterdir()
        if x.is_dir() and x.name.startswith("station=")
    ]
    return sorted(stations)


# %%
# %%
if __name__ == "__main__":
    source_folder = Path("/home/atuel/Downloads/palm-09.11.2024-09.15.2024") # Path("D:/Mis Datos/Pesados/TEC_ISEA17/rinex_data/obs/OutPut1")
    convert_to_parquet(
        source_folder, "TEC_yami", overwrite=False, compute_roti=True,
        web_version=True,
    )
#%%
    tec = get_tec_data(
       "TEC_yami",
    )

# %%
