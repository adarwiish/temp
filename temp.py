import pandas as pd
from geopy.distance import geodesic
import numpy as np

# Sample DataFrame
data = {
    'longitude': [106.780781, 106.780781, 106.780781, 106.780781, 106.780781, 106.780781, 106.780781, 106.780781],
    'latitude': [-6.223201, -6.223201, -6.223201, -6.223201, -6.223201, -6.223201, -6.223201, -6.223201],
}
df = pd.DataFrame(data)

# Convert DataFrame to list of tuples (lat, long)
coords = df[['latitude', 'longitude']].to_numpy()

def calculate_distance_matrix(coords):
    """Calculate the distance matrix between all pairs of coordinates."""
    num_sites = len(coords)
    distance_matrix = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        for j in range(num_sites):
            if i != j:
                distance_matrix[i, j] = geodesic(coords[i], coords[j]).km
    return distance_matrix

def create_batches(coords, min_distance_km):
    """Create batches based on the minimum distance constraint."""
    distance_matrix = calculate_distance_matrix(coords)
    remaining_sites = list(range(len(coords)))
    batches = []
    
    while remaining_sites:
        batch = [remaining_sites.pop(0)]  # Start with the first site
        while True:
            # Find the nearest site to the last added site in the batch that is at least min_distance_km away
            last_site = batch[-1]
            valid_sites = [site for site in remaining_sites if distance_matrix[last_site, site] >= min_distance_km]
            
            if not valid_sites:
                break
            
            distances_to_last_site = {site: distance_matrix[last_site, site] for site in valid_sites}
            next_site = min(distances_to_last_site, key=distances_to_last_site.get)
            batch.append(next_site)
            remaining_sites.remove(next_site)
        
        batches.append(batch)
    
    return batches

# Create batches
min_distance_km = 10
batches = create_batches(coords, min_distance_km)

# Add cluster labels to DataFrame
df['cluster'] = -1
for cluster_id, batch in enumerate(batches):
    for site in batch:
        df.at[site, 'cluster'] = cluster_id

# Display the results
for cluster_id, batch in df.groupby('cluster'):
    print(f"Batch {cluster_id}:")
    print(batch[['longitude', 'latitude']])
    print()
