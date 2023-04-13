# tum.ai Solar Challenge

![solar](https://user-images.githubusercontent.com/24369532/231741472-fd22977d-10a5-4715-aaf2-be472c7aaaaf.png)

## Motivation

The future of energy is green! To accelerate the clean energy revolution, we must use the resources we have in the most effective manner. Last year, 50% more solar panels were installed on German roofs than the year before. While this is great news, it does not paint the full picture. Due to supply chain limitations and skill shortage, most home owners in Germany have to wait up to 1 year for a consultation appointment with solar roof experts. For an installation, the waiting periods are even crazier: Waiting times of 2-3 years before a solar roof can be installed have become the new normal. The current solar market does not scale with the demand!

That's why we must ensure that the limited solar panel resources we have are used in the most efficient way possible! Sadly, the current system is quite the opposite. Solar construction companies serve their customers on a first-come-first-serve basis: Whoever requests the solar panel first gets the panel installed first. This inherently means that solar installations on large roofs at locations with optimal sun conditions are often postponed in favor of installation locations that are less well-suited, simply due to the fact that one request came in before another. 

To optimize the wellfare accross Germany, we should turn this process around and first evaluate which houses in Germany are best suited for solar panels. Then, we should panel the best suited houses first, getting the most efficiency out of each newly installed panel, before moving on to less optimal houses. With limited solar panels available, maximizing the efficiency of each panel is crucial and the only way to maximize the clean energy produced in the country. 

Luckily, we as Data Scientists can solve this challenge using openly available data! Using datasets from the German Weather Insitution [Deutscher Wetterdienst - DWD](https://www.dwd.de/), we can find out which regions in Germany get the highest amount of sunlight. With [Open Street Maps - OSM](https://www.openstreetmap.org/about/en), we have a crowd sourced dataset available containing detailed information for any building accross the country, allowing us to run efficient analysis on individual buildings at scale. Lastly, using Satelite Image Data from commercial vendors like [Google Maps Platform - GCP](https://developers.google.com/maps), we can even run computer vision algorithms on high quality satelite imagery to run fine grained analysis on individual houses. 

## Key Questions

With so much data publicly available, we can answer many solar related questions that are important for the clean energy future of Germany, like:

- How many German roofs must be equipped with solar panels to subtitute all of Germany's focile energy sources?
- In which German regions should solar be subsidized the most?
- Where are the 100 buildings in Germany with the largest roofs and the best efficiency per square meter?
- How much electricity can the home owner of the house at location long/lat produce per year?

And maybe even:
- Is the roof of a candidate building even suitable for solar panels, or are windows or chimneys blocking the roof area?
- Are other buildings, vegetation or even mountains casting large shadows onto a candiate building?

## Background Knowledge

### Sun Radiations dependence on geographic location

The more sun radiation a solar panel receives, the more energy it produces. Although this curve might not be linear, it is still monotonic. Hence, to find out how efficient a rooftop is, knowing how much sun radiation the roof receives is an important step. The amount of sunlight hitting a unit squared area is measured in kWh/m^2. Solar panels can not convert 100% of this energy into power, though. They typically have an efficiency factor of 0.15 - 0.2. How much solar energy lands on a square meter depends on the geographic location of the house. Generally, the closer a house is to the equator, the more energy does the sun have per square meter due to the angle between the ground and the sunrays.

<img src="https://upload.wikimedia.org/wikipedia/commons/5/55/Seasons.too.png" width="400">

[Source](https://en.wikipedia.org/wiki/Solar_irradiance#Projection_effect).

### Panel efficiency dependence on roof Azimuth and Tilt

The azimuth and tilt of a roof influences the efficiency of the installed solar panels. Positioning a solar panel on a roof that faces the true south, with a roof that is slightly tilted, yields the most efficient conversion between incoming sun radiation and generated electricity. 

<img src="https://i0.wp.com/www.prostarsolar.net/wp-content/uploads/2020/10/Solar-Panel-angle.jpg?resize=1024%2C454&ssl=1" width="400">

[Source](https://www.prostarsolar.net/article/how-to-set-solar-panel-angle-to-sun.html)

### Roof Size

Installing a multitude of solar panels on one roof is much cheaper and labor efficient than installing single solar panels on multiple roofs. Panels must not only be installed, but also cleaned and maintained regularly, which is currently a limiting factor due to labor shortage. Hence, installing as many panels on a single roof as possible reduces the time and energy needed for maintenance workers to travel between locations. Therefore, the bigger the roof, the better. 

## Datasets

### Sun Radiation

The German Weather Institution has a large collection of open source datasets, including one that includes the total radiation: [Gridded annual sum of incoming shortwave radiation (global radiation) on the
horizontal plain for Germany based on ground and satellite measurements](https://opendata.dwd.de/climate_environment/CDC/grids_germany/annual/radiation_global/DESCRIPTION_gridsgermany_annual_radiation_global_en.pdf)

Their [Dataset for 2022](https://opendata.dwd.de/climate_environment/CDC/grids_germany/annual/radiation_global/grids_germany_annual_radiation_global_2022.zip) describes a grid of 1 km^2 resolution containing radiation measurements accross Germany for the year 2022. The PDF linked above contains detailed descriptions how the data can be interpreted. Note that the data is stored in the [Gauss-Krueger Coordiante System](https://gfzpublic.gfz-potsdam.de/rest/items/item_8827_5/component/file_130038/content). You can refer to the coding hints below to find out how to convert between regular longitude-latitude coordinates and the Gaus-Krueger coordinates. 

### Building Information

[Open Street Map](https://www.openstreetmap.org/about/en) is a community driven map data provider that also contains [3D Building Information](https://osmbuildings.org/?lat=48.14907&lon=11.56744&zoom=16.0&tilt=30). 
In OSM Buildings, buildings are defined in various Level of Details (LOD). For some buildings, only their base outline is available as a 2D Polygon. However, most buildings also contain height information such that the 3D baseline polygon can be extruded to a 3D cubic form. Only for some buildings, more detail is given, such as detailed information about balconies or windows. However, such details are only given for famous buildings that OSM Contributors mostly modelled by hand. 
We can determine the size of a rooftop even from the LOD0 simply by calculating the area of the polygonial building outline. It might not be a perfect representation, but a good approximation. 
All OSM Data is Open to the public and can be requested through various open APIs, such as the [OSM Buildings API](https://osmbuildings.org/documentation/data/). 
However, you can also download all OSM Data for offline use rom providers like [Geofabrik](https://download.geofabrik.de/europe.html) and process the data offline. This is extremely useful for our case, as spaming the OSM API for millions of buildings accross Germany would certainly result in rate limits or even IP blocks. All available OSM Data accross [Germany](https://download.geofabrik.de/europe/germany.html) is 3.8 GB in size, and includes not just Building Data but all data OSM has about Germany, including addresses, streets, and much more. Note that you can also just download and process the data for a certain region to speed up your development. The smallest region is [Bremen](https://download.geofabrik.de/europe/germany/bremen.html) with only 18 MB in size.
To find out how to work with the OSM data in Python, refer to the coding hints section below.

<img src="https://osmbuildings.org/blog/2018-02-28_level_of_detail/lod.png" width="600">

### Satelite Information

From the data we have collected so far, we can locate the rooftops of millions of buildings in Germany, their size in square meters and the radiation it absorbs in kWh/m^2. However, as for most buildings only LOD0 or LOD1 are available, we can not determine whether the buildings has an flat or angled rooftop nor the azimuth of the roof. To achieve this task, we can apply computer vision on satelite data. 

Unfortunately, there are no high resolution open source datasets with satelite imagery available. While some government organizations do release public satelite imagery, their resolution is by no means high enough to analyze rooftops, as it is often 1 km^2 per pixel or worse. 

Luckily, commercial providers like [Google Maps](https://developers.google.com/maps/documentation/maps-static/start#MapTypes) allow us to request high resolution satelite imagery of any point on earth. The Google Cloud Platform Service [Static Maps API](https://developers.google.com/maps/documentation/maps-static/start) is the simplest one to use: By providing longitude latitude coordinates as URL parameters, this service returns a PNG image containing a clenaed satelite image at the given location. The [setup](https://developers.google.com/maps/documentation/maps-static/cloud-setup) is easy and can be done with anyone with a Google Cloud account. **However:** This service comes at a cost. According to Google's [Pricing](https://developers.google.com/maps/documentation/maps-static/usage-and-billing), fetching one satelite image costs 0.002 USD. Hence, fetching satelite images of 1000 buildings costs 2.00 USD. Given that Germany has millions of houses, fetching a high resolution image of each and every house is financially unfeasable. 

By setting up a new Google Cloud Account, you receive a [starting budget](https://cloud.google.com/free?hl=de) of 300 USD. This should be enough to fetch as many satelite images as you need over the course of this hackaton. However, handle and monitor your cloud budget with care! 

![Google Maps Imagery](https://maps.googleapis.com/maps/api/staticmap?center=40.714728,-73.998672&zoom=12&maptype=satellite&size=400x400&key=AIzaSyA3kg7YWugGl1lTXmAmaBGPNhDW9pEh5bo&signature=5tyWj9NAOGlFz33nroLk6sV4ASk=)

## Coding Hints

### Converting Gauss Krueger Zone 3 Coordinates to WGS 84 (long/lat)

To convert between coordinate systems in Python, a library called [pyproj](https://pyproj4.github.io/pyproj/stable/) does a lot of the heavy lifting. 

Converting from longitude/latitdude to Gauss Krueger can be done with just a few lines of code:

```python
from pyproj import Transformer

# Coordinates of TU Munich
latitude, longitude = 48.1496636, 11.5656715

# Define coordinate systems
from_crs = "EPSG:4326"  # WGS 84
to_crs = "EPSG:31467"  # Gauss Krüger Zone 3

# Create transformer object
transformer = Transformer.from_crs(from_crs, to_crs)

# Convert latitude and longitude to Gauss Krüger coordinates
h, r = transformer.transform(latitude, longitude)
```

To query the given `h, r` coordiantes against the DWD dataset, we must take the dataset boundaries into consideration.

```python
import numpy as np

# Information extracted from the dataset header
XLLCORNER = 3280500
YLLCORNER = 5237500
NROWS = 866
CELLSIZE = 1000
NODATA_VALUE = -999

# Load data as 2d array
data = np.loadtxt("grids_germany_annual_radiation_global_2022.asc", skiprows=28)
data[data == -999] = np.nan

y, x = math.floor((r - XLLCORNER) / CELLSIZE), NROWS - math.ceil((h - YLLCORNER) / CELLSIZE)
radiance = data[x, y]
```

### Load OSM Buildings data

To deal with OSM data, we can use a handy library called [Pyrosm](https://pyrosm.readthedocs.io/en/latest/) that loads OSM data as [GeoPandas](https://geopandas.org/en/stable/) data, including the polygonial building outline as a [Shapely](https://shapely.readthedocs.io/en/stable/manual.html) Polygon. 

If you are only interested in the building information contained in the OSM data, you can use tools like [OSM Convert](https://wiki.openstreetmap.org/wiki/Osmconvert) and [OSM Filter](https://wiki.openstreetmap.org/wiki/Osmfilter) to generate new files that contain only a subset of the OSM data, which might speed up your development.

```python
from pyrosm import OSM

osm = OSM('bremen-latest.osm.pbf')
buildings = osm.get_buildings()
```

### Apply Computer Vision to Satelite Imagery

Getting a satelite image from Google Maps is as simple as making a GET request to a carefully constructed URL:

```python
import requests
import cv2

url = "https://maps.googleapis.com/maps/api/staticmap?center=40.714728,-73.998672&zoom=12&maptype=satellite&size=400x400&key=AIzaSyA3kg7YWugGl1lTXmAmaBGPNhDW9pEh5bo&signature=5tyWj9NAOGlFz33nroLk6sV4ASk="
response = requests.get(url).content
image = cv2.imdecode(np.frombuffer(response, np.uint8), cv2.IMREAD_UNCHANGED)

cv2.imshow("Satelite Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

