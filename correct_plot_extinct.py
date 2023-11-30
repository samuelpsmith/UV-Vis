import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
import glob
from collections import Counter
import os
import json 
from tkinter import filedialog
from tkinter import *

# Load the JSON file
#selected_dir = "FeDppz"
#config_filename = f"{selected_dir}/config.json"

def select_directory():
    root = Tk()
    root.withdraw()  # Hide the main window

    # Show a directory selection dialog
    selected_directory = filedialog.askdirectory(title="Select a Directory")

    return selected_directory

def validate_csv_files_exist(selected_dir):
    # Validate if the CSV files exist
    csv_files = glob.glob(f"{selected_dir}/*.csv") + glob.glob(f"{selected_dir}/*.CSV")
    if not csv_files:
        return "No CSV files found in the selected directory. Validation failed."
    else:
        return "Validation passed."


def load_config(filename):
    try:
        with open(filename, "r") as json_file:
            config = json.load(json_file)
        return config
    except FileNotFoundError:
        print(f"JSON file '{filename}' not found.")
        return None


def baseline_als(y, lam=1e6, p=0.01, niter=10): # should remove magic numbers. 
    """
    Asymmetric Least Squares baseline correction algorithm
    Parameters:
    y (array) - input data (absorption spectra)
    lam (float) - smoothness parameter
    p (float) - asymmetry parameter
    niter (int) - number of iterations
    """
    L = len(y)
    D = np.diff(np.eye(L), 2)
    w = np.ones(L)
    for i in range(niter):
        W = np.diag(w)
        Z = np.linalg.inv(W + lam * D.dot(D.T))
        z = Z.dot(w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def find_molarity(mass_g, molar_mass, volume_ml):
    # Calculate molarity
    volume_l = volume_ml / 1000.0
    molarity = round(((mass_g / molar_mass) / volume_l), 6)
    return molarity


def load_and_plot_data(filename, molarity, volume_ml, baseline_correct, low_bound):
    # Load the CSV file and average the data
    df = pd.read_csv(filename, skiprows=1)
    avg_data = df.mean()

    # Truncate data to remove wavelengths below low_bound nm (e.g. 300nm)
    truncated_data = avg_data[avg_data.index.astype(float) >= low_bound]

    # Apply ALS baseline correction if baseline_correct is True
    if baseline_correct:
        corrected_data = truncated_data - baseline_als(truncated_data.values)
    else:
        corrected_data = truncated_data

    # Find tick positions for x-axis
    tick_values = [
        int(tick) for tick in corrected_data.index.astype(float) if tick % 50 == 0
    ]
    index_as_floats = corrected_data.index.astype(float).tolist()
    tick_indices = [
        index_as_floats.index(min(index_as_floats, key=lambda x: abs(x - tick_value)))
        for tick_value in tick_values
    ]

    # Plot the corrected spectra
    plt.figure(figsize=(12, 6))
    plt.plot(corrected_data.index, corrected_data.values)
    plt.title(f"Volume: {volume_ml}ml, Molarity: {molarity} M")
    plt.xlabel("Wavelength")
    plt.xticks(ticks=tick_indices, labels=tick_values, rotation=45)
    plt.ylabel("Absorbance")

    return corrected_data


def find_maxima(avg_data, percent_of_max):
    # Find local maxima
    max_value = np.amax(avg_data.values)
    height_threshold = percent_of_max * max_value
    peaks, _ = find_peaks(avg_data.values, height=height_threshold)
    return peaks


def main():
    try:

        # Ask the user to select a directory
        selected_dir = select_directory()
        print(f"Selected directory: {selected_dir}")
        
        if not selected_dir:
            print("No directory selected. Exiting.")
        else:
            validation_result = validate_csv_files_exist(selected_dir)
            print(validation_result)

        config_filename = f"{selected_dir}/config.json"    
        config = load_config(config_filename)

        if config:
            number_of_peaks = config.get("number_of_peaks")
            x_default = config.get("x_default")
            percent_of_max = config.get("percent_of_max")
            constant_C = config.get("constant_C")
            molar_mass = config.get("molar_mass")
            baseline_correct = config.get("baseline_correct")
            low_bound = config.get("low_bound")

        maxima_data = []

        csv_files = glob.glob(f"{selected_dir}/*.csv") + glob.glob(f"{selected_dir}/*.CSV")


        if not csv_files:
            print("No CSV files found. Exiting.")
            exit(1)
        else:
            print(f"Found {len(csv_files)} CSV files.")

        for filename in csv_files:
            print(f"Processing {filename}...")
            volume_match = re.search(r"-?(\d+)ml", filename)
            mass_match = re.search(r'(-?\d+\.\d+)g', filename)


            if volume_match:
                volume_ml = int(volume_match.group(1))
            else:
                print(f"Volume not found in filename {filename}. Skipping...")
                continue
            
            if mass_match:
                mass_g = float(mass_match.group(1))
            else:
                print(f"Mass not found in filename {filename}. Skipping...")
                continue

            molarity = find_molarity(mass_g, molar_mass, volume_ml)
            corrected_data = load_and_plot_data(
                filename, molarity, volume_ml, baseline_correct, low_bound
            )
            index_as_floats = corrected_data.index.astype(float).tolist()

            peaks = find_maxima(corrected_data, percent_of_max)

            for peak in peaks:
                print(
                    f"Peak at {corrected_data.index[peak]} with absorbance {corrected_data.values[peak]}"
                )
                maxima_data.append(
                    [molarity, corrected_data.index[peak], corrected_data.values[peak]]
                )

                peak_plot_index = index_as_floats.index(
                    min(
                        index_as_floats,
                        key=lambda x: abs(x - float(corrected_data.index[peak])),
                    )
                )

                peak_x_value = float(corrected_data.index[peak])
                peak_y_value = corrected_data.values[peak]
                plt.scatter(peak_plot_index, peak_y_value, color="red")
                plt.annotate(
                    f"({peak_x_value:.0f}, {peak_y_value:.2f})",
                    xy=(peak_plot_index, peak_y_value),
                    xytext=(peak_plot_index + 0.0005, peak_y_value + 0.0005),
                    # arrowprops=dict(arrowstyle="->")
                )
            plt.show()

        if not maxima_data:
            print("No maxima_data found. Exiting.")
            exit(1)

        maxima_df = pd.DataFrame(
            maxima_data, columns=["Molarity", "Wavelength", "Max_Absorbance"]
        )
        maxima_df.to_csv("maxima_data.csv", index=False)

        count_wavelength = Counter(maxima_df["Wavelength"])
        most_common_wavelengths = [
            item[0] for item in count_wavelength.most_common(number_of_peaks)
        ]
        print(f"Most common wavelengths: {most_common_wavelengths}")  # debug

        for wavelength in most_common_wavelengths:
            selected_data_frames = []
            selected_data = pd.DataFrame()
            subset = maxima_df[
                np.abs(maxima_df["Wavelength"].astype(float) - float(wavelength))
                <= x_default
            ]
            selected_data_frames.append(subset)

            selected_data = pd.concat(selected_data_frames, ignore_index=True)

            plt.figure()
            plt.scatter(
                selected_data["Molarity"] * constant_C, selected_data["Max_Absorbance"]
            )

            X = (selected_data["Molarity"] * constant_C).values.reshape(-1, 1)
            y = selected_data["Max_Absorbance"].values

            model = LinearRegression(fit_intercept=False)
            model.fit(X, y)
            slope = model.coef_[0]
            r_squared = model.score(X, y)

            plt.scatter(X, y)
            line_x = np.linspace(min(X), max(X), 100)
            line_y = slope * line_x

            plt.plot(line_x, line_y, "r")
            plt.title(
                f"Slope: {str(round(slope,2))}, R-squared: {str(round(r_squared, 2))}, Wavelength: {wavelength}"
            )
            plt.xlabel("Molarity * Pathlength")
            plt.ylabel("Absorbance")
            plt.show()

    except FileNotFoundError:
        print("Data folder or CSV files not found.")
        import traceback

        print(traceback.format_exc())

    except ValueError:
        print("Value error encountered, possibly during data conversion.")
        import traceback

        print(traceback.format_exc())

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    main()
