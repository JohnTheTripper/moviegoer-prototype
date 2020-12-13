# Unifying Features
## Putting it all together
With features extracted from the vision, audio, and subtitle data streams, we can use them in tandem to get closer to *Moviegoer* goals.

## Repository Files
The directory contains the following files:

1. *scene_boundary_partitioning.ipynb* - breakdown of how individual scenes are identified
2. *film_details_io.py* - functions for analyses and details of the overall film
3. *scene_identification_io.py* - functions for identification of two-character dialogue scenes
4. *scene_details_io.py* - functions for analyses and details at the individual scene level
5. *character_identification_io.py* - functions for tracking characters
6. *character_details_io.py* - functions for analyses and details of individual characters
7. *time_reference_io.py* - functions for manipulating datetime.time objects as well as converting between vision/frames and subtitles, which use different systems for time
