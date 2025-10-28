# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import xarray as xr

# %%
url = "https://data-cbr.csiro.au/thredds/dodsC/catch_all/CMAR_CAWCR-Wave_archive/CAWCR_Wave_Hindcast_aggregate/gridded/ww3.pac_4m.201503.nc"
ds = xr.open_dataset(url)

# %%
# Example usage
locations = [
    {"name": "Funafuti, Tuvalu", "lat": -8.5155, "lon": 179.113, "max": "2015-03-10"},
    {"name": "Tarawa, Kiribati", "lat": 1.1940, "lon": 172.5837, "max": "2015-03-10"},
    {"name": "Nanumea, Tuvalu", "lat": -5.6612, "lon": 176.1035, "max": "2015-03-09"},
    {
        "name": "Nukulaelae, Tuvalu",
        "lat": -9.3872,
        "lon": 179.8424,
        "max": "2015-03-13",
    },
    {"name": "Nui, Tuvalu", "lat": -7.2258, "lon": 177.1546, "max": "2015-03-12"},
]


# %%
def plot_wave_data(ds, locations, start_date=None, end_date=None):
    """
    Plot HS and LM for multiple locations.

    Parameters
    ----------
        ds (xarray.Dataset): Dataset containing 'hs' and 'tm'.
        locations (list of dict): Each dict must have 'name', 'lat', 'lon'.
        start_date (str, optional): e.g. "2015-03-01"
        end_date (str, optional): e.g. "2015-03-10"
    """
    for loc in locations:
        name = loc["name"]
        lat = loc["lat"]
        lon = loc["lon"]

        # Extract data at nearest grid point
        hs_sel = ds["hs"].sel(latitude=lat, longitude=lon, method="nearest")
        lm_sel = ds["lm"].sel(latitude=lat, longitude=lon, method="nearest")

        # Apply date filter if given
        if start_date and end_date:
            hs_sel = hs_sel.sel(time=slice(start_date, end_date))
            lm_sel = lm_sel.sel(time=slice(start_date, end_date))

        # Plot
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_title(f"Wave Data at {name} ({lat:.2f}, {lon:.2f})")

        ax1.plot(hs_sel["time"], hs_sel, "b-", label="HS (m)")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Significant Wave Height (m)", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        ax2 = ax1.twinx()
        ax2.plot(lm_sel["time"], lm_sel, "r-", label="LM (m)")
        ax2.set_ylabel("Mean Wave Length (m)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        plt.show()


# Call the function (assuming ds is your xarray dataset)
plot_wave_data(ds, locations)


# %%
def get_max_wave_event(ds, locations, start_date=None, end_date=None):
    """
    Select nearest grid point for each site, filter by date range,
    find max significant wave height (HS) and matching mean wavelength (LM).

    Parameters
    ----------
        ds (xarray.Dataset): Dataset with 'hs' and 'lm'.
        locations (list of dict): List of {"name": str, "lat": float, "lon": float}.
        start_date (str, optional): e.g. "2015-03-01".
        end_date (str, optional): e.g. "2015-03-31".

    Returns
    -------
        list of dict: Summary table with max HS, matching LM, and time.
    """
    results = []

    for loc in locations:
        # Extract nearest grid point
        hs_sel = ds["hs"].sel(
            latitude=loc["lat"], longitude=loc["lon"], method="nearest"
        )
        lm_sel = ds["lm"].sel(
            latitude=loc["lat"], longitude=loc["lon"], method="nearest"
        )

        # Filter by time range
        if start_date and end_date:
            hs_sel = hs_sel.sel(time=slice(start_date, end_date))
            lm_sel = lm_sel.sel(time=slice(start_date, end_date))

        # Find index of maximum HS
        max_idx = hs_sel.argmax(dim="time")

        # Retrieve values
        max_hs = float(hs_sel.isel(time=max_idx).values)
        matching_lm = float(lm_sel.isel(time=max_idx).values)
        event_time = str(hs_sel["time"].isel(time=max_idx).values)

        results.append(
            {
                "location": loc["name"],
                "lat": loc["lat"],
                "lon": loc["lon"],
                "time": event_time,
                "max_HS_m": round(max_hs, 3),
                "LM_at_max_HS_m": round(matching_lm, 3),
            }
        )

    # Print summary
    print("\nMaximum Wave Events:")
    for r in results:
        print(
            f"{r['location']} ({r['lat']:.2f}, {r['lon']:.2f}) | "
            f"Date: {r['time']} | HS: {r['max_HS_m']} m | LM: {r['LM_at_max_HS_m']} m"
        )

    return results


summary = get_max_wave_event(
    ds, locations, start_date="2015-03-06", end_date="2015-03-20"
)

# %%
