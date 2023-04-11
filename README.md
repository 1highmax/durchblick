# tum.ai Solar Challenge

## Motivation

The future of energy is green! To accelerate the clean energy revolution, we must use the resources we have in the most effective manner. Last year, 50% more solar panels were installed on german roofs than the year before. While this is great news, it does not paint the full picture. Due to supply chain limitations and skill shortage, most home owners in germany have to wait up to 1 year for a consultation appointment with solar roof experts. For an installation, the waiting periods are even crazier: Waiting times for 2-3 years before a solar roof can be installed has become the new normal. The current solar market does not scale with the demand!

That's why we must ensure that the limited solar panel resources we have are used in the most efficient way possible! Sadly, the current system is not at all efficient. Solar construction companies serve their customers on a first-come-first-serve basis: Whoever requests the solar panel first gets the panel installed first. This inherently means that solar installations on large roofs at location with optimal sun conditions are often postponed in favor of installations locations that are less well-suited, simply due to the fact that one requests came in before the other. 

To optimize the wellfare accross germany, we should turn this process around and first evaluate which houses in germany are best suited for solar panels. Then, we should panel the best suited houses first, getting the most efficiency out of each newly installed panel, before moving on to less optimal houses. With limited solar panels available, maximizing the efficiency of each panel is crucial and the only way to maximize the clean energy produced in the country. 

Luckily, we as Data Scientists can solve this challenge using openly available data! Using datasets from the german weather insitution [Deutscher Wetterdienst - DWD](https://www.dwd.de/), we can find out which regions in germany have the most sun. With [Püem Street Maps - OSM)[https://www.openstreetmap.org/about], we have a crowd sourced dataset available containing detailed information for any building accross the country, allowing us to run efficiently analysis on individual buildings at scale. Lastly, using Satelite Image Data from commercial vendors like [Google Maps Platform - GMP](https://developers.google.com/maps), we can even run computer vision algorithms on high quality satelite imagery to run fine grained analysis on individual houses. 

## Key Questions

With so much data publicly available, we can answer many solar related questions that are important for the clean energy future of germany, like:

- How many german roofs must be equipped with solar panels to subtitute all of germanies focile energy sources?
- In which regions of germany should solar be substituted the most?
- Where are the 100 buildings in germany with the largest roofs and the best efficiency per square meter?
- How much electricity can the home owner of the house at location long/lat produce per year?

## Background Knowledge



