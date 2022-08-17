# SL2lab_Delgado_et_al_2022
Data analysis methods for manuscript "Universal Pre-Mixing Dry-Film Stickers Capable of Retrofitting Existing Microfluidics"

Script dot_quant_intensity is designed to analyze intensity of interconnected regions of signal from fluoresence microscopy video data over time. This script relies heavily on morphological operations performed using library OpenCV and region analysis performed using library Scikit-image.

Script created for the Sensors for Living Systems lab (SL2) lead by David Myers, PhD, based at the Wallace H. Coulter department of Biomedical Engineering at Georgia Institute of Technology and Emory University

My contribution to this manuscript: conceived of data analysis methods, wrote script, performed analysis.

## Inputs, methods, outputs

Users are guided to choose a single .avi video file using a file dialog window

Users must edit the following parameters based on their own individual data files:

```
# USER SHOULD EDIT THESE IMPORTANT VARIABLES
kernel_erode = (51, 51)  # Kernel for morphological opening operation (3 iterations)
kernel_dilate = (101, 101)  # Kernel for morphological closing operation (3 iterations)
noise_size = 50
threshold = 20  # Threshold value for applied binary threshold, needs to work best for brightest image
graph_threshold = 4000  # Minimum summed intensity for graph of n regions above threshold
channel = 'r'  # Indicate channel to analyze = r: red, b: blue, g: green, gs: greyscale
```

Script reads all video frames to locate the highest-intensity frame. From this frame, a "map" is built using the following operations:
- A binary threshold is applied using the specified threshold variable
- Erosion is applied to this binary image using the specified kernel for erosion to remove noise
- Dilation is applied to the binary image with erosion using the specified kernel for dilation to better connect regions of signal, e.g. the sticker dots

Region analysis then finds coordinates of all interconnected regions of signal.
Pixel intensity values of coordinates of each interconnected region of signal are summed for every frame.

Outputs include:
- Numerical data saved within an excel sheet, containing the intensity of each located region at each time point. Includes the mean intensity of all located regions and the number of regions with summed intensity greater than the graphing_threshold at each time point.
- Graphical data saved as .png files, including a line graph indicating each region/mean intensity over time and a line graph indicating number of regions with intensity greater than the provided threshold over time.
- Each frame labeled with indices corresponding to numerical data.
- Final "map" labeled with indices corresponding to numerical data.

These outputs are organized in a new analysis directory, located in the directory of the file selected.

## Help and contributing
Contributions are always welcome! Submit a pull request, contact me directly at meredith.e.fay@gmail.com, or contact the SL2 lab coding team directly at myers.sl2.lab@gmail.com. Please also feel free to reach out via email for assistance. The SL2 lab is happy to provide detailed experimental methods upon request.

## References
OpenCV:
- Bradski, G. "The Opencv Library." Dr. Dobb’s Journal of Software Tools 2000  (2000).

Scikit-image:
- van der Walt, Stéfan, Johannes L. Schönberger, Juan Nunez-Iglesias, François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle Gouillart, Tony Yu, and contributors the scikit-image. "Scikit-Image: Image Processing in Python." PeerJ 2 (2014/06/19 2014): e453. https://doi.org/10.7717/peerj.453. https://doi.org/10.7717/peerj.453.

