# Load libaries
from plantcv import plantcv as pcv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# libraries & dataset


def read_ms_image(filename):
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


def reshape_ms_data(ms_image):
    # reshape the 3D-array into 2D
    ms_data = np.reshape(ms_image.array_data, (-1, ms_image.array_data.shape[2]))

    # convert the array to a pandas dataframe for downstream processing
    ms_data_df = pd.DataFrame(ms_data)
    ms_data_df.columns = list(ms_image.wavelength_dict)

    # changing the dataframe shape to long (required for easy plotting)
    ms_data_melted = ms_data_df.melt()
    ms_data_melted.columns = ["wavelength", "value"]

    return ms_data_melted, ms_image.filename


def export_data_summary(reshaped_ms_data, ms_img_filename, out_folder=None):
    summary = reshaped_ms_data.groupby("wavelength")["value"].describe(percentiles=[.25, .5, .75, .95], include="all")

    filename = os.path.basename(ms_img_filename) + "_summary.csv"

    if out_folder:
        out_path = os.path.join(out_folder, filename)
    else:
        out_path = filename

    summary.to_csv(out_path, index=False)


def print_violin_plot(reshaped_ms_data, ms_img_filename, out_folder=None):
    sns.set(style="darkgrid")

    sns.violinplot(x=reshaped_ms_data["wavelength"], y=reshaped_ms_data["value"],
                   scale="count", inner=None, gridsize=20, linewidth=1)

    filename = os.path.basename(ms_img_filename) + "_violin_plot.png"

    if out_folder:
        out_path = os.path.join(out_folder, filename)
    else:
        out_path = filename

    plt.savefig(out_path)


if __name__ == '__main__':
    ms_image_path = "/home/alex/Nextcloud/Documents/ETC/img_processing/notebooks/2023-04-11_intensity_compairison/RVstr-2218E0_000000_20230407_220000"

    img = read_ms_image(ms_image_path)
    data, name = reshape_ms_data(img)
    print_histogram(img)
    print_violin_plot(data, name)
    export_data_summary(data, name)


