from geopy.distance import geodesic
import numpy as np
from scipy.cluster.hierarchy import fclusterdata

# Sample site coordinates (latitude, longitude)
sites = [
    (30.033333, 31.233334),  # Example: Cairo, Egypt
    (29.976480, 31.131302),  # Giza Pyramids
    # Add your site coordinates here
]

# Distance threshold in kilometers (10 km in this case)
min_distance_km = 10

# Convert coordinates to distances (in kilometers)
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).km

# Use fclusterdata from scipy to cluster based on minimum distance
clusters = fclusterdata(sites, t=min_distance_km, criterion='distance', metric=lambda u, v: geodesic(u, v).km)

# Group sites by clusters
from collections import defaultdict
batches = defaultdict(list)
for site, cluster_id in zip(sites, clusters):
    batches[cluster_id].append(site)

# Output the batches
for cluster_id, batch in batches.items():
    print(f"Batch {cluster_id}:")
    for site in batch:
        print(f" - {site}")
