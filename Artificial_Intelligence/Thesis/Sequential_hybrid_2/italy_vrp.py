#!/usr/bin/env python3
"""
generate_italy_vrp_autocities.py

Generates a synthetic VRP instance for Italy:
 - Detects major Italian cities automatically using Natural Earth metadata
 - Creates clustered customer points near those cities using a distance gradient
 - Adds white noise for rural customers beyond city influence
 - Places depots at cluster centers using K-Means (multiple depots per cluster)
 - Outputs a CVRPLIB-compatible .vrp file

Requires: geopandas, shapely, numpy, scikit-learn
"""

import random
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import cm

BASE_DIR = Path(__file__).parent

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------
COUNTRIES_SHP = BASE_DIR / "data/ne_10m_admin_0_countries_ita/ne_10m_admin_0_countries_ita.shp"
POPULATED_PLACES_SHP = BASE_DIR / "data/ne_10m_populated_places/ne_10m_populated_places.shp"

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
N_DEPOTS = 100         # number of depots, now multiple per cluster
N_CLUSTERED = 4000
N_RURAL = 20_000
SIGMA = 0.5           # city influence spread
NOISE_STD = 0.02      # jitter around points
PROB_THRESHOLD = 1e-2 # below this probability, switch to white noise
OUTPUT_FILE = "italy_auto_cities.vrp"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# Load Italy polygon
# ------------------------------------------------------------
countries = gpd.read_file(COUNTRIES_SHP)
italy_geom = countries[countries["ADMIN"] == "Italy"].geometry.unary_union

# ------------------------------------------------------------
# Load populated places and auto-select major cities
# ------------------------------------------------------------
places = gpd.read_file(POPULATED_PLACES_SHP)
italy_cities = places[places["ADM0NAME"] == "Italy"]

MAJOR_CITIES_DF = italy_cities[
    (italy_cities["FEATURECLA"].isin(["Admin-0 capital", "Admin-1 capital"])) |
    (italy_cities["SCALERANK"] <= 5) |
    (italy_cities.get("MEGACITY", 0) == 1)
]

MAJOR_CITIES = {row["NAME"]: (row.geometry.y, row.geometry.x) for _, row in MAJOR_CITIES_DF.iterrows()}
print(f"Detected {len(MAJOR_CITIES)} major cities:", list(MAJOR_CITIES.keys()))

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def random_point_in_italy():
    minx, miny, maxx, maxy = italy_geom.bounds
    while True:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = Point(x, y)
        if italy_geom.contains(p):
            return (y, x)

def distance_to_nearest_city(lat, lon):
    if not MAJOR_CITIES:
        return float("inf")
    return min(np.hypot(lat - city_lat, lon - city_lon) for city_lat, city_lon in MAJOR_CITIES.values())

def generate_customers(n_customers):
    customers = []
    attempts = 0
    max_attempts = n_customers * 100
    while len(customers) < n_customers and attempts < max_attempts:
        attempts += 1
        minx, miny, maxx, maxy = italy_geom.bounds
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        pnt = Point(x, y)
        if not italy_geom.contains(pnt):
            continue

        dist = distance_to_nearest_city(y, x)
        prob = np.exp(-dist / SIGMA)

        if prob > PROB_THRESHOLD:
            if random.random() >= prob:
                continue  # gradient phase
        # else white noise phase

        lat_noisy = y + np.random.normal(0, NOISE_STD)
        lon_noisy = x + np.random.normal(0, NOISE_STD)
        if italy_geom.contains(Point(lon_noisy, lat_noisy)):
            customers.append((lat_noisy, lon_noisy))
    return customers

# ------------------------------------------------------------
# Generate customers
# ------------------------------------------------------------
deliveries = generate_customers(N_CLUSTERED + N_RURAL)
customer_coords = np.array(deliveries)

# ------------------------------------------------------------
# Place depots using K-Means on customer coordinates
# ------------------------------------------------------------
kmeans = KMeans(n_clusters=N_DEPOTS, random_state=SEED)
kmeans.fit(customer_coords)
cluster_centers = kmeans.cluster_centers_

depots = []
for lat, lon in cluster_centers:
    p = Point(lon, lat)
    if italy_geom.contains(p):
        depots.append((lat, lon))
    else:
        # small adjustment to move inside Italy
        for _ in range(50):
            lat2 = lat + np.random.normal(0, 0.01)
            lon2 = lon + np.random.normal(0, 0.01)
            if italy_geom.contains(Point(lon2, lat2)):
                depots.append((lat2, lon2))
                break

# ------------------------------------------------------------
# Assign customer data
# ------------------------------------------------------------
def random_demand(): 
    return random.randint(1, 30)

customers = [
    {"id": i + 1, "lat": lat, "lon": lon, "demand": random_demand()}
    for i, (lat, lon) in enumerate(deliveries)
]

# ------------------------------------------------------------
# Write CVRPLIB .vrp file
# ------------------------------------------------------------
def write_vrp_file(filename):
    with open(filename, "w") as f:
        f.write("NAME : Italy_AutoCities\n")
        f.write("COMMENT : Synthetic VRP with auto-detected Italian city clusters\n")
        f.write("TYPE : CVRP\n")
        f.write(f"DIMENSION : {len(customers) + len(depots)}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("CAPACITY : 100\n")

        f.write("\nNODE_COORD_SECTION\n")
        for i, (lat, lon) in enumerate(depots, start=1):
            f.write(f"{i} {lon} {lat}\n")
        for i, c in enumerate(customers, start=len(depots)+1):
            f.write(f"{i} {c['lon']} {c['lat']}\n")

        f.write("\nDEMAND_SECTION\n")
        for _ in depots:
            f.write("0\n")
        for c in customers:
            f.write(f"{c['demand']}\n")

        f.write("\nDEPOT_SECTION\n")
        for i in range(1, len(depots)+1):
            f.write(f"{i}\n")
        f.write("-1\nEOF\n")

    print(f"✔ Wrote file: {filename}")
    print(f"  {len(customers)} customers + {len(depots)} depot(s)")

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    write_vrp_file(OUTPUT_FILE)

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------
    distances = np.array([distance_to_nearest_city(c["lat"], c["lon"]) for c in customers])
    dist_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-9)
    colors = cm.get_cmap("coolwarm_r")(1 - dist_norm)

    plt.figure(figsize=(10, 12))
    gpd.GeoSeries([italy_geom]).plot(color="lightgrey", edgecolor="black", linewidth=0.5)

    # depots
    plt.scatter([lon for lat, lon in depots], [lat for lat, lon in depots],
                color="red", s=50, label="Depots")

    # customers
    plt.scatter([c["lon"] for c in customers], [c["lat"] for c in customers],
                c=colors, s=10, alpha=0.7, label="Customers (gradient)")

    # major cities
    plt.scatter([lon for lat, lon in MAJOR_CITIES.values()],
                [lat for lat, lon in MAJOR_CITIES.values()],
                color="black", s=30, marker="x", label="Major cities")

    plt.title("Generated VRP Customers with Distance Gradient and Cluster Depots")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.show()
