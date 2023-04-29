#!/bin/env python3

import geopandas as gpd
from pyrosm import OSM

# osm = OSM('germany-latest.osm.pbf')
osm = OSM('bremen-latest.osm.pbf')
buildings = osm.get_buildings()

# Calculate the area for each building
buildings['area'] = buildings['geometry'].area

# Sort the buildings by area in descending order
sorted_buildings = buildings.sort_values(by='area', ascending=False)

# Print the sorted buildings
print(sorted_buildings)

# Print the biggest building
print("Biggest building:")
print(sorted_buildings.iloc[0])

print("finished")
