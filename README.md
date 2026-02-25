LiDAR Multi-Hillshade Animation
A tool for animating multi-directional hillshade visualizations created from LiDAR data.

Overview
This tool takes a multi-band hillshade raster as input and produces a smooth animated visualization by cycling through each band (hillshade direction) sequentially. It works best with rasters generated using the RVT (Relief Visualization Toolbox) multi-hillshade tool in QGIS.

Input
Your input file should be a multi-band hillshade raster created using the QGIS RVT Multi-hillshade tool.
The recommended number of bands is 16–64, where more bands produce smoother animations at the cost of longer processing times.

Animation Timing
Frame duration scales with the number of bands. As a guide, a 32-band hillshade works well with a frame duration of around 200ms.

Export Options
By default the tool exports an animated GIF, but you can also choose from the following output formats:

PNG — exports each band as an individual PNG file
MP4 — exports as a video file
RGB mode — combines 3 hillshade bands simultaneously into a single RGB-composite frame, with an option to control the spacing between the selected bands (band distance)
