# Load libraries
from plantcv import plantcv as pcv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def read_ms_image(filename):
    # check if a .hdr file name was provided and set img_file to the binary location
    if os.path.splitext(filename)[1] == ".hdr":
        filename = os.path.splitext(filename)[0]

    ms_image = pcv.readimage(filename=filename, mode="envi")

    # convert data type from uint8 to float32 for downstream calculations
    ms_image.array_data = ms_image.array_data.astype("float32")

    return ms_image


def print_histogram(ms_image, out_folder=None):
    # if a dark value (wavelength == 0) is included, exclude it from the visualization
    if 0.0 in ms_image.wavelength_dict.keys():
        wavelengths = list(ms_image.wavelength_dict)[1:]
    else:
        wavelengths = list(ms_image.wavelength_dict)

    # create histogram
    hist = pcv.visualize.hyper_histogram(hsi=ms_image,
                                         wvlengths=wavelengths,
                                         bins=50)

    filename = os.path.basename(ms_image.filename) + "_hist.png"

    if out_folder:
        out_path = os.path.join(out_folder, filename)
    else:
        out_path = filename

    pcv.print_image(hist, out_path)
    print("Created histogram: " + out_path)


def reshape_ms_data(ms_image):
    # reshape the 3D-array into 2D
    ms_data = np.reshape(ms_image.array_data, (-1, ms_image.array_data.shape[2]))

    # convert the array to a pandas dataframe for downstream processing
    ms_data_df = pd.DataFrame(ms_data)
    ms_data_df.columns = list(ms_image.wavelength_dict)

    # changing the dataframe shape to long (required for easy plotting)
    ms_data_melted = ms_data_df.melt()
    ms_data_melted.columns = ["wavelength", "value"]

    return ms_data_melted


def export_data_summary(ms_image, out_folder=None):
    reshaped_ms_data = reshape_ms_data(ms_image)

    summary = reshaped_ms_data.groupby("wavelength")["value"].describe(percentiles=[.25, .5, .75, .95], include="all")

    filename = os.path.basename(ms_image.filename) + "_summary.csv"

    if out_folder:
        out_path = os.path.join(out_folder, filename)
    else:
        out_path = filename

    summary.to_csv(out_path, index=False)
    print("Created data summary: " + out_path)


def print_violin_plot(ms_image, out_folder=None):
    reshaped_ms_data = reshape_ms_data(ms_image)

    sns.set(style="darkgrid")
    sns.violinplot(x=reshaped_ms_data["wavelength"],
                   y=reshaped_ms_data["value"],
                   scale="count",
                   inner=None,
                   gridsize=20,
                   linewidth=1)

    filename = os.path.basename(ms_image.filename) + "_violin_plot.png"

    if out_folder:
        out_path = os.path.join(out_folder, filename)
    else:
        out_path = filename

    plt.savefig(out_path)
    print("Created violin plot: " + out_path)


if __name__ == '__main__':
    ms_image_path = input("Please insert path to multispectral binary file here: ")
    print("analyzing image" + ms_image_path)

    img = read_ms_image(ms_image_path)
    print_histogram(img)
    print_violin_plot(img)
    export_data_summary(img)

    print("Done!")
