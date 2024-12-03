#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze Stacked Spectra and Plot Peaks with Gaussian Fits

Author: Bibal Sobeaux Pierre Gabriel
Date: 2024-12-04

This script scans the 'Reduced/stacked' directory for stacked FITS spectra,
identifies peaks in each spectrum, fits Gaussian profiles to each peak, and
generates high-quality plots with wavelength calibration and Gaussian overlays.

Usage:
    python analyze_stacked_spectra.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]
                                     [--calibration_file CALIBRATION_FILE]
                                     [--height HEIGHT] [--distance DISTANCE]
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

def plot_spectrum(wavelength, intensity, peaks, gaussian_fits, output_file, title=None):
    """
    Plots the spectrum with identified peaks and their Gaussian fits.

    Parameters:
        wavelength (np.ndarray): Wavelength array.
        intensity (np.ndarray): Intensity array.
        peaks (np.ndarray): Indices of detected peaks.
        gaussian_fits (list of dict): List containing Gaussian fit parameters for each peak.
        output_file (str): Path to save the plot.
        title (str, optional): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(wavelength, intensity, label='Spectrum', color='black')
    plt.plot(wavelength[peaks], intensity[peaks], 'ro', label='Detected Peaks')

    # Plot Gaussian fits
    for fit in gaussian_fits:
        if fit['success']:
            plt.plot(fit['x_fit'], fit['y_fit'], 'b--', label='Gaussian Fit' if 'Gaussian Fit' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.xlabel('Wavelength (Å)', fontsize=14)
    plt.ylabel('Intensity (ADU)', fontsize=14)
    if title:
        plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
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

def format_peak_detail(peak, width=13):
    """
    Formats the peak detail for summary.

    Parameters:
        peak (dict): Dictionary containing peak details.
        width (int): Width for formatting.

    Returns:
        str: Formatted peak detail string.
    """
    if not np.isnan(peak['wavelength']):
        wavelength_str = f"{peak['wavelength']:.2f}".rjust(13)
    else:
        wavelength_str = "N/A".rjust(13)

    if not np.isnan(peak['amplitude']):
        amplitude_str = f"{peak['amplitude']:.2f}".rjust(9)
    else:
        amplitude_str = "N/A".rjust(9)

    if not np.isnan(peak['sigma']):
        sigma_str = f"{peak['sigma']:.2f}".rjust(9)
    else:
        sigma_str = "N/A".rjust(9)

    fit_success_str = "Yes" if peak['fit_success'] else "No"

    return f"  {peak['peak_index']:10d} | {wavelength_str} | {amplitude_str} | {sigma_str} | {fit_success_str}\n"

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
    parser.add_argument('--height', type=float, default=4000,
                        help='Minimum height of peaks for detection (default: 4000)')
    parser.add_argument('--distance', type=float, default=10,
                        help='Minimum distance between peaks in pixels (default: 10)')
    parser.add_argument('--prominence', type=float, default=1000,
                        help='Minimum prominence of peaks (default: 1000)')
    parser.add_argument('--fitting_window', type=int, default=5,
                        help='Number of pixels on each side of a peak to include in Gaussian fit (default: 5)')
    parser.add_argument('--save_format', type=str, choices=['png', 'pdf'], default='png',
                        help='Format to save plots (default: png)')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    calibration_file = args.calibration_file
    height = args.height
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

            # Detect peaks
            peaks = detect_peaks(wavelength, data, height=height, distance=distance, prominence=prominence)
            num_peaks = len(peaks)
            print(f"Detected {num_peaks} peaks.")

            # Fit Gaussians to each peak
            gaussian_fits = []
            for peak_idx in peaks:
                fit = fit_gaussian(wavelength, data, peak_idx, fitting_window)
                gaussian_fits.append(fit)
                if fit['success']:
                    print(f"  Peak at index {peak_idx} fitted successfully: amplitude={fit['amplitude']:.2f}, mean={fit['mean']:.2f} Å, sigma={fit['sigma']:.2f} Å")
                else:
                    print(f"  Peak at index {peak_idx} fit failed.")

            # Plotting
            base_name = os.path.splitext(os.path.basename(fits_file))[0]
            plot_filename = os.path.join(output_dir, f"{base_name}_peaks.{save_format}")
            title = f"Spectrum with Detected Peaks: {base_name}"
            plot_spectrum(wavelength, data, peaks, gaussian_fits, plot_filename, title=title)
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
                f.write("  Peak Index | Wavelength (Å) | Amplitude | Sigma (Å) | Fit Success\n")
                for peak in entry['peaks']:
                    # Use the helper function to format each peak detail
                    peak_detail_str = format_peak_detail(peak)
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