"""Script created for the Sensors for Living Systems Lab (SL2), lead by David Myers, PhD,
based at Georgia Institute of Technology and Emory University

Script used for data analysis in manuscript:
"Universal Pre-Mixing Dry-Film Stickers Capable of Retrofitting Existing Microfluidics"
(Delgado, 2022)

Script author: Meredith Fay, Lam Lab, Georgia Institute of Technology and Emory University
Last updated: 2022-08-17

Script measures summed raw intensity of individual regions of signal within a field of view over time.
A binary threshold is applied to the image with the greatest total intensity using OpenCV.
--Line to apply adaptive threshold instead is provided in comments
Region analysis is performed on this binary image using sci-kit image.
Raw pixel intensity of each region is summed for each frame.

This script analyzes a single video file.

Inputs include:
--Single video file

--Channel to analyze
----Choose 'r', 'b', 'g', or 'gs'
--Kernel size for erosion, dilation morphological operations, input as a tuple
----See: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
----First operation "erosion" removes noise from image
----Second operation "dilation" connects all pixels within regions of signal
----Kernel for opening should be smaller than kernel for closing for best results
----Designed to work with 3 iterations, could easily change this within code
--Noise size: the area below which an interconnected region is considered noise
--Graphing threshold: for the graph showing n regions above a summed threshold

The brightest image, the image with threshold applied, the image after erosion, and the
image after dilation are all shown to assist user in selecting parameters
--Could easily comment out these sections of code

Outputs include:
--Numerical data
----Excel file of intensity of each located region at each time point
----First column: mean intensity of all located regions
----Second column: number of regions with summed intensity above threshold
----Each region is given an index
--Excel sheet contains parameters used

--Labeled images
----Each frame labeled with the indices for each located region, corresponding to numerical data
----The binary threshold "map" labeled with indices

--Graphical data
----Line graph where each light-colored line indicates the intensity of an individual region over time
-----The dark-colored line indicates the mean intensity of all regions over time
----Line graph indicating how many regions are more intense than the provided "graphing threshold"

Please see github repository README.md for additional information.

"""

# Import libraries
#   File management
import os  # For directory management
from tkinter import filedialog
import datetime  # Time stamping results

#   Number, file, and image management
import cv2  # Computer vision/image processing
from PIL import Image, ImageDraw  # Labeling regions of signal with index
from skimage import measure  # For region analysis
import numpy as np  # For array management
import pandas as pd  # For database management

#   Plotting results
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# USER SHOULD EDIT THESE IMPORTANT VARIABLES
kernel_erode = (51, 51)  # Kernel for morphological opening operation (3 iterations)
kernel_dilate = (101, 101)  # Kernel for morphological closing operation (3 iterations)
noise_size = 50
threshold = 20  # Value for applied binary threshold, needs to work best for brightest image
graph_threshold = 4000  # Minimum summed intensity for graph of n regions above threshold
channel = 'r'  # Indicate channel to analyze = r: red, b: blue, g: green, gs: greyscale


# Choose file, determine video name for saving results
filepath = filedialog.askopenfilename()
dir = os.path.dirname(filepath)
filename = os.path.basename(filepath).split(".")[0]

# Create results directory
now = datetime.datetime.now()  # Results are timestamped
current_dir = os.getcwd()

output_folder = os.path.join(dir, 'Results, ' + now.strftime("%m_%d_%Y, %H_%M_%S"))
os.mkdir(output_folder)
os.chdir(output_folder)


# Read video
vid = cv2.VideoCapture(filepath)
length = vid.get(cv2.CAP_PROP_FRAME_COUNT)
fps = vid.get(cv2.CAP_PROP_FPS)

# Create lists to save frames, modified frames to
frames = []

count = 1
success, image = vid.read()  # Initial read

while success:  # While remaining frames exist
    success, image = vid.read()
    if image is not None:

        # Index channel to create one layer image suitable for threshold
        # Here use an 'RGB' color scheme
        if channel is 'r':
            img_layer = image[:, :, 2]
        elif channel is 'g':
            img_layer = image[:, :, 1]
        elif channel is 'b':
            img_layer = image[:, :, 0]
        elif channel is 'gs':
            img_layer = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Save
        frames.append(img_layer.copy())
        count += 1

# Find frame with max value and use as map
max_index = np.argmax([np.sum(f, axis=None) for f in frames])

# Clean up using morphological operations
img_map = frames[max_index]

cv2.imshow("Brightest image", img_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Binary threshold, option for Gaussian filtering and Otsu's binarization
ret, img_threshold = cv2.threshold(img_map, threshold, 255, cv2.THRESH_BINARY)
# kernel_blur = (5, 5)  # May need to edit for use
# blur = cv2.GaussianBlur(img_layer, kernel_blur, 0)
# ret, img_threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("Image with threshold applied", img_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_map_erode = cv2.erode(img_threshold, kernel_erode, iterations=3)

cv2.imshow("Image after erosion operation", img_map_erode)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_map_dilate = cv2.dilate(img_map_erode, kernel_dilate, iterations=3)

cv2.imshow("Image after dilation operation", img_map_dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create a labeled image for regionprops input from brightest frame
labels = measure.label(img_map_dilate)
prop_table = measure.regionprops_table(labels, properties=('centroid', 'area', 'coords'))  # Region analysis
prop_table_df = pd.DataFrame(prop_table)  # Convert to dataframe

# Filter out any values below noise_size
prop_table_df = prop_table_df[prop_table_df['area'] > noise_size]

prop_table_df['ID'] = np.arange(len(prop_table_df)) + 1  # Add cell index



df = pd.DataFrame()  # Data to save
for i in range(len(prop_table_df)):  # For each blob
    region_int = []  # List to add region of intensity to
    coords_i = prop_table_df['coords'].iloc[i]  # Find where region is located
    for j in range(len(frames)):  # For each image
        frame_j = frames[j]
        int = np.sum(frame_j[coords_i[:, 0], coords_i[:, 1]])  # Find summed intensity of this region
        region_int.append(int)  # Save this value

    # Create a new column in the final dataframe
    col_name = 'Region ' + str(prop_table_df['ID'].iloc[i])
    df[col_name] = region_int  # Save vector of per-frame intensity values

# Find number of regions with summed intensity above threshold
above_threshold = []
for k in range(len(df)):
    df_t = df.iloc[k, :]
    above_threshold.append(len(df_t[df_t > graph_threshold]))

# Add to dataframe
df.insert(loc=0, column='Values > %d' % graph_threshold, value=above_threshold)

# Find mean summed intensity of a region in each frame
means = df.mean(axis=1)
df.insert(loc=0, column='Mean int. (a.u.)', value=means)

# Add time values to numerical data
times = np.arange(count - 1)/fps
df.insert(loc=0, column='Time (s)', value=times)

# Save numerical data
writer = pd.ExcelWriter(filename + '.xlsx', engine='openpyxl')
df.to_excel(writer, sheet_name='Data', index=False)

# Save parameters
now = datetime.datetime.now()
# Print parameters to a sheet
param_df = pd.DataFrame({'Kernel, erode operation': kernel_erode,
                         'Kernel, dilate operation': kernel_dilate,
                         'Threshold for graphing (a.u.)': graph_threshold,
                         'Minimum region area (pix)': noise_size,
                         'Color channel analyzed': channel,
                         'Frame with max. intensity': max_index,
                         'Analysis date': now.strftime("%D"),
                         'Analysis time': now.strftime("%H:%M:%S")})
param_df.to_excel(writer, sheet_name='Parameters used', index=False)

writer.save()
writer.close()

writer.save()
writer.close()

# Label original frames with indices
img_shape = frames[0].shape
# Save images
for l in range(len(frames)):
    # Concatenate layer to recreate grayscale
    green = np.dstack((frames[l], frames[l], frames[l])).astype(np.uint8)
    PILimg = Image.fromarray(green)  # Set up image to label
    drawimg = ImageDraw.Draw(PILimg)  # " "
    for m in range(len(prop_table_df)):
        drawimg.text((prop_table_df['centroid-1'].iloc[m], prop_table_df['centroid-0'].iloc[m]),
                     str(prop_table_df['ID'].iloc[m]), fill="#ff0000")  # Label

    image_name = filename + '_frame_' + str(l).zfill(5)  # Set up name for image
    PILimg.save(image_name + "_labeled.png")  # Save image

opening = np.dstack((img_map_dilate, img_map_dilate, img_map_dilate))
PILimg = Image.fromarray(opening)  # Set up image to label
drawimg = ImageDraw.Draw(PILimg)  # " "
for o in range(len(prop_table_df)):
    drawimg.text((prop_table_df['centroid-1'].iloc[o], prop_table_df['centroid-0'].iloc[o]),
                 str(prop_table_df['ID'].iloc[o]), fill="#ff0000")  # Label

image_name = filename + '_map'  # Set up name for image
PILimg.save(image_name + "_labeled.png")  # Save image

# Graph
fig = plt.figure()
for p in np.arange(2, len(df.columns)):
    plt.plot(df['Time (s)'], df.iloc[:, [p]], color='palegreen')
plt.plot(df['Time (s)'], df['Mean int. (a.u.)'], color='seagreen')


n_patch = mpatches.Patch(color='palegreen', label='Individual regions')
mean_patch = mpatches.Patch(color='seagreen', label='Mean value')
plt.legend(handles=[n_patch, mean_patch])

plt.savefig(filename + '_all_graph.png', dpi=300)

# Graph
fig = plt.figure()
plt.plot(df['Time (s)'], df['Values > %d' % graph_threshold], color='seagreen')

plt.title(filename)
plt.xlabel('Time (s)')
plt.ylabel('Regions with fluoresence intensity value > %d (n)' % graph_threshold)

plt.tight_layout()

plt.savefig(filename + '_threshold_graph.png', dpi=300)