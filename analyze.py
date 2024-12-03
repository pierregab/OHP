#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze Stacked Spectra and Plot Peaks with Gaussian Fits

Author: Bibal Sobeaux Pierre Gabriel
Date: 2024-12-04

This script scans the 'Reduced/stacked' directory for stacked FITS spectra,
identifies peaks in each spectrum based on a dynamic threshold relative to
the mean intensity, fits Gaussian profiles to each peak, and
generates high-quality plots with wavelength calibration and Gaussian overlays.

Additionally, it matches detected peaks to known spectral lines using air wavelengths
when they provide a better fit, computes the associated redshift for each match,
and includes this information in both the summary file and the plots.

Usage:
    python analyze_stacked_spectra.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]
                                     [--calibration_file CALIBRATION_FILE]
                                     [--height_sigma HEIGHT_SIGMA] [--distance DISTANCE]
                                     [--prominence PROMINENCE] [--fitting_window FITTING_WINDOW]
                                     [--save_format SAVE_FORMAT]

Dependencies:
    - numpy
    - matplotlib
    - astropy
    - scipy
    - argparse
    - os
    - glob

Ensure all required dependencies are installed. You can install missing packages
using pip:
    pip install numpy matplotlib astropy scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import os
import argparse
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ======================================================================
# Spectral Lines Data
# ======================================================================

SPECTRAL_LINES = [
    {"Line": "Ca II K", "Wavelength_air": 3933.663, "Wavelength_vacuum": 3934.777, "Notes": "IS absorb line"},
    {"Line": "Ca II H", "Wavelength_air": 3968.468, "Wavelength_vacuum": 3969.591, "Notes": "IS absorb line"},
    {"Line": "[OII]3726", "Wavelength_air": 3726.03, "Wavelength_vacuum": 3727.09, "Notes": ""},
    {"Line": "[OII]3729", "Wavelength_air": 3728.82, "Wavelength_vacuum": 3729.88, "Notes": ""},
    {"Line": "[NeIII]", "Wavelength_air": 3868.71, "Wavelength_vacuum": 3869.81, "Notes": ""},
    {"Line": "[NeIII]", "Wavelength_air": 3967.41, "Wavelength_vacuum": 3968.53, "Notes": ""},
    {"Line": "He I", "Wavelength_air": 3888.65, "Wavelength_vacuum": 3889.75, "Notes": ""},
    {"Line": "H12", "Wavelength_air": 3750.15, "Wavelength_vacuum": 3751.22, "Notes": ""},
    {"Line": "H11", "Wavelength_air": 3770.63, "Wavelength_vacuum": 3771.70, "Notes": ""},
    {"Line": "H10", "Wavelength_air": 3797.90, "Wavelength_vacuum": 3798.98, "Notes": ""},
    {"Line": "H9", "Wavelength_air": 3835.39, "Wavelength_vacuum": 3836.48, "Notes": ""},
    {"Line": "H8", "Wavelength_air": 3889.05, "Wavelength_vacuum": 3890.15, "Notes": ""},
    {"Line": "Hε", "Wavelength_air": 3970.07, "Wavelength_vacuum": 3971.19, "Notes": ""},
    {"Line": "Hδ", "Wavelength_air": 4101.76, "Wavelength_vacuum": 4102.92, "Notes": ""},
    {"Line": "Hγ", "Wavelength_air": 4340.47, "Wavelength_vacuum": 4341.69, "Notes": ""},
    {"Line": "Hβ", "Wavelength_air": 4861.33, "Wavelength_vacuum": 4862.69, "Notes": ""},
    {"Line": "Hα", "Wavelength_air": 6562.79, "Wavelength_vacuum": 6564.61, "Notes": "NIST"},
    {"Line": "[OIII]4363", "Wavelength_air": 4363.21, "Wavelength_vacuum": 4364.44, "Notes": ""},
    {"Line": "[OIII]4959", "Wavelength_air": 4958.92, "Wavelength_vacuum": 4960.30, "Notes": ""},
    {"Line": "[OIII]5007", "Wavelength_air": 5006.84, "Wavelength_vacuum": 5008.24, "Notes": ""},
    {"Line": "Mg b", "Wavelength_air": 5167.321, "Wavelength_vacuum": 5168.761, "Notes": ""},
    {"Line": "Mg b", "Wavelength_air": 5172.684, "Wavelength_vacuum": 5174.125, "Notes": ""},
    {"Line": "Mg b", "Wavelength_air": 5183.604, "Wavelength_vacuum": 5185.048, "Notes": ""},
    {"Line": "[O I]5577", "Wavelength_air": 5577.3387, "Wavelength_vacuum": 5578.8874, "Notes": "Strong sky line"},
    {"Line": "[NII]5755", "Wavelength_air": 5754.64, "Wavelength_vacuum": 5756.24, "Notes": ""},
    {"Line": "He I", "Wavelength_air": 5875.67, "Wavelength_vacuum": 5877.30, "Notes": ""},
    {"Line": "Na I", "Wavelength_air": 5889.951, "Wavelength_vacuum": 5891.583, "Notes": "IS absorb line"},
    {"Line": "Na I", "Wavelength_air": 5895.924, "Wavelength_vacuum": 5897.558, "Notes": "IS absorb line"},
    {"Line": "[O I]6300", "Wavelength_air": 6300.30, "Wavelength_vacuum": 6302.04, "Notes": ""},
    {"Line": "[O I]6363", "Wavelength_air": 6363.776, "Wavelength_vacuum": 6364.60, "Notes": "NIST"},
    {"Line": "[NII]6549", "Wavelength_air": 6548.03, "Wavelength_vacuum": 6549.84, "Notes": ""},
    {"Line": "[NII]6583", "Wavelength_air": 6583.41, "Wavelength_vacuum": 6585.23, "Notes": ""},
    {"Line": "He I", "Wavelength_air": 6678.152, "Wavelength_vacuum": 6679.996, "Notes": ""},
    {"Line": "[SII]6717", "Wavelength_air": 6716.47, "Wavelength_vacuum": 6718.32, "Notes": ""},
    {"Line": "[SII]6731", "Wavelength_air": 6730.85, "Wavelength_vacuum": 6732.71, "Notes": ""},
    {"Line": "[S III]", "Wavelength_air": 6312.06, "Wavelength_vacuum": 6313.81, "Notes": "Use with [SIII]9068 as T diagnostic"},
    {"Line": "[O I]5577", "Wavelength_air": 5577.3387, "Wavelength_vacuum": 5578.8874, "Notes": "Strong sky line"},
    {"Line": "[ArIII]", "Wavelength_air": 7135.8, "Wavelength_vacuum": 7137.8, "Notes": ""},
    {"Line": "[ArIII]", "Wavelength_air": 7751.1, "Wavelength_vacuum": 7753.2, "Notes": ""},
    {"Line": "Ca II", "Wavelength_air": 8498.03, "Wavelength_vacuum": 8500.36, "Notes": "Ca II triplet"},
    {"Line": "Ca II", "Wavelength_air": 8542.09, "Wavelength_vacuum": 8544.44, "Notes": "Ca II triplet"},
    {"Line": "Ca II", "Wavelength_air": 8662.14, "Wavelength_vacuum": 8664.52, "Notes": "Ca II triplet"},
    {"Line": "[SIII]9068", "Wavelength_air": 9068.6, "Wavelength_vacuum": 9071.1, "Notes": "Use with [SIII]6312 as T diagnostic"},
    {"Line": "[SIII]9530", "Wavelength_air": 9530.6, "Wavelength_vacuum": 9533.2, "Notes": ""},
    {"Line": "P11", "Wavelength_air": 8862.783, "Wavelength_vacuum": 8865.217, "Notes": ""},
    {"Line": "P10", "Wavelength_air": 9014.910, "Wavelength_vacuum": 9017.385, "Notes": ""},
    {"Line": "P9", "Wavelength_air": 9229.014, "Wavelength_vacuum": 9231.547, "Notes": ""},
    {"Line": "P8", "Wavelength_air": 9545.972, "Wavelength_vacuum": 9548.590, "Notes": ""},
    {"Line": "P7", "Wavelength_air": 10049.373, "Wavelength_vacuum": 10052.128, "Notes": ""},
    {"Line": "Pγ", "Wavelength_air": 10938.095, "Wavelength_vacuum": 10941.091, "Notes": ""},
    {"Line": "Pβ", "Wavelength_air": 12818.08, "Wavelength_vacuum": 12821.59, "Notes": "(1.281808 µm)"},
    {"Line": "Pα", "Wavelength_air": 18751.01, "Wavelength_vacuum": 18756.13, "Notes": "(1.875101 µm)"},
    {"Line": "[Fe II]", "Wavelength_air": 16400.0, "Wavelength_vacuum": 16440.0, "Notes": "(1.64 µm)"},  # Approximate values
    {"Line": "Brγ", "Wavelength_air": 21655.29, "Wavelength_vacuum": 21661.20, "Notes": "(2.165529 µm)"},
    {"Line": "H₂ S(1) 1-0", "Wavelength_air": 21220.0, "Wavelength_vacuum": 21220.0, "Notes": "(2.122 µm)"},  # Approximate values
    {"Line": "H₂ S(0) 1-0", "Wavelength_air": 22230.0, "Wavelength_vacuum": 22230.0, "Notes": "(2.223 µm)"},  # Approximate values
]

# ======================================================================
# Helper Functions
# ======================================================================

def read_spectrum(filename, get_header=False):
    """
    Reads a 1D spectrum from a FITS file.

    Parameters:
        filename (str): Path to the FITS file.
        get_header (bool): Whether to return the FITS header.

    Returns:
        tuple: (xaxis, data) if get_header=False,
               (xaxis, data, header) if get_header=True
    """
    with fits.open(filename) as hdul:
        data = hdul[0].data.astype(float)
        # If data is 2D, squeeze to 1D
        if data.ndim > 1:
            data = data.squeeze()
        header = hdul[0].header if get_header else None
    xaxis = np.arange(len(data))
    return (xaxis, data, header) if get_header else (xaxis, data)

def read_wavelength_calibration(calibration_file):
    """
    Reads the wavelength calibration from a .dat file.

    Parameters:
        calibration_file (str): Path to the wavelength calibration file.

    Returns:
        np.ndarray: Wavelength array.
    """
    try:
        wavelength = np.loadtxt(calibration_file)
        return wavelength
    except Exception as e:
        print(f"Error reading calibration file {calibration_file}: {e}")
        return None

def get_wavelength_axis(calibration_file, spectrum_length):
    """
    Generates the wavelength axis using the calibration file.

    Parameters:
        calibration_file (str): Path to the wavelength calibration file.
        spectrum_length (int): Number of data points in the spectrum.

    Returns:
        np.ndarray: Wavelength array.
    """
    wavelength = read_wavelength_calibration(calibration_file)
    if wavelength is None:
        print("Using pixel indices as wavelength axis due to calibration file read error.")
        return np.arange(spectrum_length)
    if len(wavelength) != spectrum_length:
        print(f"Calibration file length ({len(wavelength)}) does not match spectrum length ({spectrum_length}). Using pixel indices.")
        return np.arange(spectrum_length)
    return wavelength

def detect_peaks(wavelength, intensity, height, distance, prominence):
    """
    Detects peaks in the spectrum using specified parameters.

    Parameters:
        wavelength (np.ndarray): Wavelength array.
        intensity (np.ndarray): Intensity array.
        height (float): Minimum height of peaks.
        distance (float): Minimum distance between peaks in pixels.
        prominence (float): Minimum prominence of peaks.

    Returns:
        np.ndarray: Indices of detected peaks.
    """
    peaks, properties = find_peaks(intensity, height=height, distance=distance, prominence=prominence)
    return peaks

def gaussian(x, amplitude, mean, sigma):
    """
    Gaussian function.

    Parameters:
        x (np.ndarray): Independent variable.
        amplitude (float): Amplitude of the Gaussian.
        mean (float): Mean of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: Gaussian function evaluated at x.
    """
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def fit_gaussian(wavelength, intensity, peak_idx, window):
    """
    Fits a Gaussian to the data around a peak.

    Parameters:
        wavelength (np.ndarray): Wavelength array.
        intensity (np.ndarray): Intensity array.
        peak_idx (int): Index of the peak.
        window (int): Number of pixels on each side of the peak to include in the fit.

    Returns:
        dict: Fit parameters (amplitude, mean, sigma) and fit success status.
    """
    # Define the fitting window
    left = max(peak_idx - window, 0)
    right = min(peak_idx + window + 1, len(intensity))
    x_data = wavelength[left:right]
    y_data = intensity[left:right]

    # Initial guesses
    amplitude_guess = intensity[peak_idx] - np.median(intensity)
    mean_guess = wavelength[peak_idx]
    if len(x_data) > 1:
        delta_wavelength = wavelength[1] - wavelength[0]
    else:
        delta_wavelength = 1  # Prevent division by zero
    sigma_guess = (delta_wavelength * window) / 2.355  # Approximate

    try:
        popt, _ = curve_fit(gaussian, x_data, y_data, p0=[amplitude_guess, mean_guess, sigma_guess])
        fit_success = True
    except RuntimeError:
        popt = [np.nan, np.nan, np.nan]
        fit_success = False

    return {
        'amplitude': popt[0],
        'mean': popt[1],
        'sigma': popt[2],
        'success': fit_success,
        'x_fit': x_data,
        'y_fit': gaussian(x_data, *popt) if fit_success else None
    }

def match_spectral_lines(peak_wavelength, spectral_lines, tolerance=7.0):
    """
    Matches a peak wavelength to possible spectral lines within a given tolerance.
    Only uses air wavelengths for matching.

    Parameters:
        peak_wavelength (float): Wavelength of the detected peak.
        spectral_lines (list of dict): List of spectral lines with their properties.
        tolerance (float): Maximum difference in Angstroms to consider a match.

    Returns:
        list of dict: List of matching spectral lines with air wavelength used.
    """
    matches = []
    for line in spectral_lines:
        # Calculate difference for air wavelength only
        diff_air = abs(peak_wavelength - line["Wavelength_air"])

        # Check if within tolerance
        if diff_air <= tolerance:
            matches.append({
                "Line": line["Line"],
                "Wavelength": line["Wavelength_air"],
                "Delta": peak_wavelength - line["Wavelength_air"],
                "Notes": line["Notes"]
            })
    return matches

def compute_redshift(observed_wavelength, rest_wavelength):
    """
    Computes the redshift based on observed and rest wavelengths.

    Parameters:
        observed_wavelength (float): Observed wavelength in Angstroms.
        rest_wavelength (float): Rest (air) wavelength in Angstroms.

    Returns:
        float: Calculated redshift z.
    """
    if rest_wavelength <= 0:
        return np.nan
    return (observed_wavelength / rest_wavelength) - 1

from adjustText import adjust_text

def plot_spectrum(wavelength, intensity, peaks, gaussian_fits, output_file, title=None, peak_matches=None):
    """
    Plots the spectrum with identified peaks and their Gaussian fits.

    Parameters:
        wavelength (np.ndarray): Wavelength array.
        intensity (np.ndarray): Intensity array.
        peaks (np.ndarray): Indices of detected peaks.
        gaussian_fits (list of dict): List containing Gaussian fit parameters for each peak.
        output_file (str): Path to save the plot.
        title (str, optional): Title of the plot.
        peak_matches (list of list of dict): List containing matched spectral lines for each peak.
    """
    plt.figure(figsize=(16, 10))  # Increased figure size for more space

    # Plot the spectrum and detected peaks
    plt.plot(wavelength, intensity, label='Spectrum', color='black', linewidth=1.5)
    plt.plot(wavelength[peaks], intensity[peaks], 'ro', label='Detected Peaks', markersize=6)

    # Plot Gaussian fits
    for fit in gaussian_fits:
        if fit['success']:
            plt.plot(fit['x_fit'], fit['y_fit'], 'b--',
                     label='Gaussian Fit' if 'Gaussian Fit' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Annotate peaks using adjust_text
    texts = []
    if peak_matches is not None:
        for idx, peak_idx in enumerate(peaks):
            matches = peak_matches[idx]
            if matches:
                # Choose the best match
                best_match = min(matches, key=lambda m: abs(m['Delta']))
                z = compute_redshift(gaussian_fits[idx]['mean'], best_match['Wavelength'])
                delta = best_match['Delta']
                annotation_text = f"{best_match['Line']} (Air)\nΔ={delta:.2f} Å\nz={z:.4f}"
                x = gaussian_fits[idx]['mean']
                y = intensity[peak_idx]
                text = plt.text(x, y, annotation_text, fontsize=10, color='blue', ha='center', va='bottom')
                texts.append(text)

    # Dynamically adjust text placements using adjust_text
    adjust_text(
        texts,
        arrowprops=dict(
            arrowstyle="-|>",  # Arrow style with a clean arrowhead
            color='darkblue',  # Arrow color
            lw=1.2,            # Line width of the arrow
            shrinkA=35,        # Adjusted to position arrow start appropriately
            shrinkB=5          # Slight shrink to avoid touching the point directly
        ),
        expand_points=(1.5, 1.7),  # Expand space around points
        expand_text=(4.0, 4.5),    # Further expand space around text for farther placement
        force_text=(2, 2.4),       # Increase force to push texts farther
        force_points=(1.5, 1.8),   # Increase force to push points away from texts
        lim=1000,                  # Increase the number of iterations for better adjustments
        only_move={'texts': 'y', 'points': 'xy'},  # Allow movement in both directions
        add_objects=[plt.gca().lines, plt.gca().collections],  # Avoid overlaps with plot elements
        expand=(1.5, 1),           # General expansion to provide more space
        arrow_length_ratio=0.1     # Adjust arrow length relative to the plot
    )

    # Adjust plot margins to provide more space for annotations
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Add labels, title, and grid
    plt.xlabel('Wavelength (Å)', fontsize=14)
    plt.ylabel('Intensity', fontsize=14)
    if title:
        plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()

    # Save and close the plot
    plt.savefig(output_file, dpi=300)
    plt.close()

def ensure_directory(directory):
    """
    Ensures that a directory exists. Creates it if it does not.

    Parameters:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def format_peak_detail(peak, spectral_lines, tolerance=7.0, width=13):
    """
    Formats the peak detail for summary, including possible spectral line matches and redshift.
    Only air wavelengths are considered.

    Parameters:
        peak (dict): Dictionary containing peak details.
        spectral_lines (list of dict): List of spectral lines for matching.
        tolerance (float): Tolerance for matching wavelengths.
        width (int): Width for formatting.

    Returns:
        str: Formatted peak detail string.
    """
    # Define fixed widths for each column
    peak_index_width = 10
    wavelength_width = 14
    amplitude_width = 9
    sigma_width = 9
    fit_success_width = 11
    spectral_line_width = 16
    line_wavelength_width = 18
    delta_width = 9
    z_width = 8

    # Format Peak Index
    peak_index_str = f"{peak['peak_index']}".rjust(peak_index_width)

    # Format Wavelength
    if not np.isnan(peak['wavelength']):
        wavelength_str = f"{peak['wavelength']:.2f}".rjust(wavelength_width)
    else:
        wavelength_str = "N/A".rjust(wavelength_width)

    # Format Amplitude
    if not np.isnan(peak['amplitude']):
        amplitude_str = f"{peak['amplitude']:.2f}".rjust(amplitude_width)
    else:
        amplitude_str = "N/A".rjust(amplitude_width)

    # Format Sigma
    if not np.isnan(peak['sigma']):
        sigma_str = f"{peak['sigma']:.2f}".rjust(sigma_width)
    else:
        sigma_str = "N/A".rjust(sigma_width)

    # Format Fit Success
    fit_success_str = "Yes".rjust(fit_success_width) if peak['fit_success'] else "No".rjust(fit_success_width)

    # Match spectral lines using only air wavelengths
    if not np.isnan(peak['wavelength']):
        matches = match_spectral_lines(peak['wavelength'], spectral_lines, tolerance)
    else:
        matches = []

    if matches:
        # Each match will have: Spectral Line | Line Wavelength | Delta | z
        matched_lines = "; ".join([
            f"{m['Line']:<{spectral_line_width}} | {m['Wavelength']:<{line_wavelength_width}.2f} Å | Δ={m['Delta']:<{delta_width - 2}.2f} Å | z={compute_redshift(peak['wavelength'], m['Wavelength']):<{z_width}.4f}"
            for m in matches
        ])
    else:
        # If no matches, fill the fields with "None" appropriately
        matched_lines = "None".ljust(spectral_line_width + line_wavelength_width + delta_width + z_width + 9)  # 9 accounts for separators and labels

    # Combine all fields into a single formatted string
    formatted_str = (
        f"  {peak_index_str} | {wavelength_str} | {amplitude_str} | {sigma_str} | {fit_success_str} | {matched_lines}\n"
    )

    return formatted_str

# ======================================================================
# Main Function
# ======================================================================

def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description="Analyze Stacked Spectra and Plot Peaks with Gaussian Fits")
    parser.add_argument('--input_dir', type=str, default='reducing/Reduced/stacked',
                        help='Directory containing stacked FITS spectra (default: reducing/Reduced/stacked)')
    parser.add_argument('--output_dir', type=str, default='Reduced/stacked/plots',
                        help='Directory to save the plots (default: Reduced/stacked/plots)')
    parser.add_argument('--calibration_file', type=str, default='reducing/reduced/wavelength_calibration.dat',
                        help='Path to the wavelength calibration file (default: reducing/reduced/wavelength_calibration.dat)')
    parser.add_argument('--height_sigma', type=float, default=0.5,
                        help='Number of standard deviations above the mean for peak detection (default: 1.0)')
    parser.add_argument('--distance', type=float, default=10,
                        help='Minimum distance between peaks in pixels (default: 10)')
    parser.add_argument('--prominence', type=float, default=300,
                        help='Minimum prominence of peaks (default: 300)')
    parser.add_argument('--fitting_window', type=int, default=3,
                        help='Number of pixels on each side of a peak to include in Gaussian fit (default: 5)')
    parser.add_argument('--save_format', type=str, choices=['png', 'pdf'], default='png',
                        help='Format to save plots (default: png)')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    calibration_file = args.calibration_file
    height_sigma = args.height_sigma
    distance = args.distance
    prominence = args.prominence
    fitting_window = args.fitting_window
    save_format = args.save_format

    # Ensure output directory exists
    ensure_directory(output_dir)

    # Read wavelength calibration once
    calibration_wavelength = read_wavelength_calibration(calibration_file)
    if calibration_wavelength is None:
        print("Exiting due to calibration file read error.")
        return

    # Find all FITS files in the input directory
    fits_files = glob.glob(os.path.join(input_dir, '*.fits'))
    if not fits_files:
        print(f"No FITS files found in directory: {input_dir}")
        return

    # Summary of processed files
    summary = []

    # Process each FITS file
    for fits_file in fits_files:
        print(f"Processing file: {fits_file}")
        try:
            x, data, header = read_spectrum(fits_file, get_header=True)
            spectrum_length = len(data)
            wavelength = get_wavelength_axis(calibration_file, spectrum_length)

            # Calculate mean and standard deviation
            mean_intensity = np.mean(data)
            std_intensity = np.std(data)

            # Set dynamic height based on mean and std
            dynamic_height = mean_intensity + height_sigma * std_intensity

            # Detect peaks with dynamic height
            peaks = detect_peaks(wavelength, data, height=dynamic_height, distance=distance, prominence=prominence)
            num_peaks = len(peaks)
            print(f"Detected {num_peaks} peaks with height > mean + {height_sigma}*std.")

            # Fit Gaussians to each peak
            gaussian_fits = []
            for peak_idx in peaks:
                fit = fit_gaussian(wavelength, data, peak_idx, fitting_window)
                gaussian_fits.append(fit)
                if fit['success']:
                    print(f"  Peak at index {peak_idx} fitted successfully: amplitude={fit['amplitude']:.2f}, mean={fit['mean']:.2f} Å, sigma={fit['sigma']:.2f} Å")
                else:
                    print(f"  Peak at index {peak_idx} fit failed.")

            # Match peaks to spectral lines and compute redshift
            peak_matches = []
            for idx, peak_idx in enumerate(peaks):
                fit = gaussian_fits[idx]
                if fit['success']:
                    peak_wavelength = fit['mean']
                    matches = match_spectral_lines(peak_wavelength, SPECTRAL_LINES, tolerance=7.0)
                    peak_matches.append(matches)
                else:
                    peak_matches.append([])

            # Plotting
            base_name = os.path.splitext(os.path.basename(fits_file))[0]
            plot_filename = os.path.join(output_dir, f"{base_name}_peaks.{save_format}")
            title = f"Spectrum with Detected Peaks: {base_name}"
            plot_spectrum(wavelength, data, peaks, gaussian_fits, plot_filename, title=title, peak_matches=peak_matches)
            print(f"Plot saved to {plot_filename}")

            # Append to summary
            peak_details = []
            for idx, peak_idx in enumerate(peaks):
                fit = gaussian_fits[idx]
                peak_info = {
                    'peak_index': int(peak_idx),
                    'wavelength': fit['mean'] if fit['success'] else np.nan,
                    'amplitude': fit['amplitude'] if fit['success'] else np.nan,
                    'sigma': fit['sigma'] if fit['success'] else np.nan,
                    'fit_success': fit['success']
                }
                peak_details.append(peak_info)

            summary.append({
                'file': os.path.basename(fits_file),
                'num_peaks': num_peaks,
                'peaks': peak_details
            })

        except Exception as e:
            print(f"Error processing {fits_file}: {e}")
            continue

    # Save summary to a text file
    summary_file = os.path.join(output_dir, 'peak_summary.txt')
    try:
        with open(summary_file, 'w') as f:
            f.write("Peak Detection and Gaussian Fit Summary\n")
            f.write("=======================================\n\n")
            for entry in summary:
                f.write(f"File: {entry['file']}\n")
                f.write(f"Number of Peaks Detected: {entry['num_peaks']}\n")
                f.write("Peak Details:\n")
                f.write("  Peak Index | Wavelength (Å) | Amplitude | Sigma (Å) | Fit Success | Spectral Line    | Line Wavelength (Å)  | Delta (Å)   | z      \n")
                f.write("-----------------------------------------------------------------------------------------------------------------------------------------\n")
                for peak in entry['peaks']:
                    # Use the updated helper function to format each peak detail
                    peak_detail_str = format_peak_detail(peak, SPECTRAL_LINES, tolerance=7.0)
                    f.write(peak_detail_str)
                f.write("\n")
        print(f"Summary of peak detections and Gaussian fits saved to {summary_file}")
    except Exception as e:
        print(f"Error writing summary file: {e}")

    print("All files processed.")

# ======================================================================
# Entry Point
# ======================================================================

if __name__ == '__main__':
    main()