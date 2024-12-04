#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze Stacked Spectra and Plot Peaks with Gaussian Fits

Author: Bibal Sobeaux Pierre Gabriel
Date: 2024-12-04

This script scans the 'Reduced/stacked' directory for stacked FITS spectra,
identifies peaks in each spectrum based on a dynamic threshold relative to
the mean intensity, fits Gaussian profiles with a constant background to each peak,
and generates high-quality plots with wavelength calibration and Gaussian overlays.

Additionally, it matches detected peaks to known spectral lines using air wavelengths
when they provide a better fit, computes the associated redshift and velocity for each match,
and includes this information in both the summary file and the plots.

Furthermore, if both [SII] λ6717 and [SII] λ6731 lines are detected and fitted,
the script computes the electron density based on their flux ratio and includes
this information in the summary.

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
    - pyneb
    - adjustText

Ensure all required dependencies are installed. You can install missing packages
using pip:
    pip install numpy matplotlib astropy scipy pyneb adjustText
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import os
import argparse
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from astropy.constants import c  # Import speed of light
import sys

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

def gaussian_plus_bg(x, *params):
    """
    Gaussian function with a constant background.

    Parameters:
        x (np.ndarray): Independent variable.
        *params: Parameters for the model. The first parameter is the background (bg),
                 followed by sets of three parameters for each Gaussian (amplitude, mean, sigma).

    Returns:
        np.ndarray: Model evaluated at x.
    """
    result = np.full(len(x), 0.0)
    bg = params[0]
    result += bg
    for i in range(1, len(params), 3):
        amp, cen, wid = params[i:i+3]
        result += amp * np.exp(-0.5 * ((x - cen) / wid) ** 2)
    return result

def fit_gaussian_plus_bg(wavelength, intensity, peak_idx, window):
    """
    Fits Gaussian(s) with a constant background to the data around a peak.

    Parameters:
        wavelength (np.ndarray): Wavelength array.
        intensity (np.ndarray): Intensity array.
        peak_idx (int): Index of the peak.
        window (int): Number of pixels on each side of the peak to include in the fit.

    Returns:
        dict: Fit parameters (background, amplitude, mean, sigma) and fit success status.
    """
    # Define the fitting window
    left = max(peak_idx - window, 0)
    right = min(peak_idx + window + 1, len(intensity))
    x_data = wavelength[left:right]
    y_data = intensity[left:right]

    # Initial guesses
    bg_guess = np.median(intensity)
    amplitude_guess = intensity[peak_idx] - bg_guess
    mean_guess = wavelength[peak_idx]
    if len(x_data) > 1:
        delta_wavelength = wavelength[1] - wavelength[0]
    else:
        delta_wavelength = 1  # Prevent division by zero
    sigma_guess = (delta_wavelength * window) / 2.355  # Approximate

    # Initial parameter list: [bg, amp, mean, sigma]
    initial_params = [bg_guess, amplitude_guess, mean_guess, sigma_guess]

    try:
        popt, _ = curve_fit(gaussian_plus_bg, x_data, y_data, p0=initial_params)
        fit_success = True
    except RuntimeError:
        popt = [np.nan, np.nan, np.nan, np.nan]
        fit_success = False

    return {
        'background': popt[0],
        'amplitude': popt[1],
        'mean': popt[2],
        'sigma': popt[3],
        'success': fit_success,
        'x_fit': x_data,
        'y_fit': gaussian_plus_bg(x_data, *popt) if fit_success else None
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

def compute_velocity(z):
    """
    Computes the velocity based on redshift.

    Parameters:
        z (float): Redshift.

    Returns:
        float: Velocity in km/s.
    """
    return z * c.to('km/s').value  # c is already imported from astropy.constants

from adjustText import adjust_text

def plot_spectrum(wavelength, intensity, peaks, gaussian_fits, output_file, title=None, peak_matches=None):
    """
    Plots the spectrum with identified peaks and their Gaussian fits, including vertically oriented
    labels for matched spectral lines positioned directly above each peak. The y-axis is dynamically
    adjusted to ensure all labels are fully visible within the plot.

    Parameters:
        wavelength (np.ndarray): Wavelength array.
        intensity (np.ndarray): Intensity array.
        peaks (np.ndarray): Indices of detected peaks.
        gaussian_fits (list of dict): List containing Gaussian fit parameters for each peak.
        output_file (str): Path to save the plot.
        title (str, optional): Title of the plot.
        peak_matches (list of list of dict): List containing matched spectral lines for each peak.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 10))
    ax = plt.gca()

    # Plot the spectrum
    ax.plot(wavelength, intensity, label='Spectrum', color='black', linewidth=1.5)

    # Plot detected peaks
    ax.plot(wavelength[peaks], intensity[peaks], 'ro', label='Detected Peaks', markersize=6)

    # Plot Gaussian fits
    for fit in gaussian_fits:
        if fit['success']:
            ax.plot(fit['x_fit'], fit['y_fit'], 'b--', label='Gaussian Fit' if 'Gaussian Fit' not in ax.get_legend_handles_labels()[1] else "")

    # Get current y-axis limits
    current_ylim = ax.get_ylim()
    y_min, y_max = current_ylim
    y_range = y_max 

    # Initialize a list to store label positions for y-axis adjustment
    label_positions = []

    # Initialize lists to store label texts and positions
    labels = []  # To store label information for the second pass
    if peak_matches is not None:
        for idx, peak_idx in enumerate(peaks):
            matches = peak_matches[idx]
            if matches:
                # Choose the best match based on the smallest delta
                best_match = min(matches, key=lambda m: abs(m['Delta']))
                line_wavelength = best_match['Wavelength']
                line_name = best_match['Line']

                # Get the corresponding Gaussian fit
                fit = gaussian_fits[idx]
                if fit['success']:
                    peak_wavelength = fit['mean']
                    # Calculate the peak's intensity assuming amplitude is the peak height above the background
                    peak_intensity = fit['background'] + fit['amplitude']

                    # Define an offset for the label to appear above the peak (5% of y-axis range)
                    offset = 0.05 * y_range
                    label_y = peak_intensity + offset

                    # Compute redshift and velocity
                    z = compute_redshift(peak_wavelength, line_wavelength)
                    v = compute_velocity(z)

                    labels.append({
                        'x': peak_wavelength,
                        'y': label_y,
                        'text': f"{line_name}"
                    })

                    label_positions.append(label_y)

    # If no labels, ensure some margin above the data
    ax.set_ylim(top=y_max + 0.1 * y_range)

    # Second pass: Plot the labels
    for label in labels:
        peak_wavelength = label['x']
        label_y = label['y']
        line_info = label['text']

        # Add vertically rotated label
        ax.text(
            peak_wavelength, label_y, line_info,
            rotation=90,
            verticalalignment='bottom',
            horizontalalignment='center',
            color='black',
            fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0, alpha=0.6)
        )

    # Adjust plot margins to provide space for labels
    plt.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.1)

    # Add labels, title, and grid
    ax.set_xlabel('Wavelength (Å)', fontsize=14)
    ax.set_ylabel('Intensity', fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, which='both', ls='--', lw=0.5)
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
    Formats the peak detail for summary, including possible spectral line matches, redshift, and velocity.
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
    v_width = 11  # Added width for velocity

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
        # Each match will have: Spectral Line | Line Wavelength | Delta | z | v
        matched_lines = "; ".join([
            f"{m['Line']:<{spectral_line_width}} | {m['Wavelength']:<{line_wavelength_width}.2f} Å | Δ={m['Delta']:<{delta_width-4}.2f} Å | z={compute_redshift(peak['wavelength'], m['Wavelength']):<{z_width}.4f} | v={compute_velocity(compute_redshift(peak['wavelength'], m['Wavelength'])):<{v_width}.2f} km/s"
            for m in matches
        ])
    else:
        # If no matches, fill the fields with "None" appropriately
        matched_lines = "None".ljust(spectral_line_width + line_wavelength_width + delta_width + z_width + v_width + 9)  # 9 accounts for separators and labels

    # Combine all fields into a single formatted string
    formatted_str = (
        f"  {peak_index_str} | {wavelength_str} | {amplitude_str} | {sigma_str} | {fit_success_str} | {matched_lines}\n"
    )

    return formatted_str

# ======================================================================
# Electron Density Calculation Function
# ======================================================================

def calculate_electron_density(fit_results, spectral_lines, ratio_tolerance=1.0):
    """
    Calculates the electron density using the flux ratio of [SII] 6717 and [SII] 6731 lines.

    Parameters:
        fit_results (list of dict): List of Gaussian fit results for all peaks.
        spectral_lines (list of dict): List of spectral lines for matching.
        ratio_tolerance (float): Maximum allowed difference in Angstroms to match the [SII] lines.

    Returns:
        float: Electron density in cm^-3, or np.nan if calculation is not possible.
    """
    try:
        import pyneb as pn
    except ImportError:
        print("DEBUG: PyNeb is not installed. Please install it using 'pip install pyneb'.")
        return np.nan

    # Initialize variables to store fluxes
    flux_6717 = None
    flux_6731 = None

    # Debug: Initialize list to track matched lines
    matched_lines = []

    # Loop through all fitted peaks to find [SII] lines
    for idx, fit in enumerate(fit_results):
        if not fit['success']:
            print(f"DEBUG: Fit for peak index {idx} failed. Skipping.")
            continue
        wavelength = fit['mean']
        amplitude = fit['amplitude']
        print(f"DEBUG: Processing peak {idx} with fitted wavelength {wavelength:.2f} Å and amplitude {amplitude:.2f}.")

        # Find matching spectral lines within the tolerance
        matches = match_spectral_lines(wavelength, spectral_lines, tolerance=ratio_tolerance)
        if not matches:
            print(f"DEBUG: No spectral line matches found for peak {idx} at {wavelength:.2f} Å within tolerance {ratio_tolerance} Å.")
            continue

        for match in matches:
            line_name = match['Line']
            line_wavelength = match['Wavelength']
            delta = match['Delta']
            notes = match['Notes']
            print(f"DEBUG: Peak {idx} matches line '{line_name}' at {line_wavelength:.2f} Å with delta {delta:.2f} Å. Notes: {notes}")

            if line_name == "[SII]6717":
                if flux_6717 is not None:
                    print(f"WARNING: Multiple matches found for [SII]6717. Overwriting previous flux.")
                flux_6717 = amplitude
                matched_lines.append(line_name)
            elif line_name == "[SII]6731":
                if flux_6731 is not None:
                    print(f"WARNING: Multiple matches found for [SII]6731. Overwriting previous flux.")
                flux_6731 = amplitude
                matched_lines.append(line_name)

    # Debug: Check if both [SII] lines were found
    print(f"DEBUG: Flux [SII]6717 = {flux_6717}, Flux [SII]6731 = {flux_6731}")
    if flux_6717 is not None:
        print(f"DEBUG: Detected [SII]6717 with flux {flux_6717:.2f}.")
    else:
        print("DEBUG: [SII]6717 not detected.")

    if flux_6731 is not None:
        print(f"DEBUG: Detected [SII]6731 with flux {flux_6731:.2f}.")
    else:
        print("DEBUG: [SII]6731 not detected.")

    # Check if both fluxes are found and flux_6731 is not zero
    if flux_6717 is not None and flux_6731 is not None:
        if flux_6731 == 0:
            print("DEBUG: [SII]6731 flux is zero. Cannot compute flux ratio.")
            return np.nan

        ratio = flux_6717 / flux_6731
        print(f"DEBUG: Flux ratio [SII]6717/[SII]6731 = {ratio:.4f}")

        # Use PyNeb to calculate electron density
        # Assuming a typical temperature, e.g., 10,000 K
        # This can be adjusted or made dynamic based on other diagnostics
        T_e = 10000  # Electron temperature in K
        print(f"DEBUG: Assuming electron temperature T_e = {T_e} K for electron density calculation.")

        try:
            # Initialize PyNeb's Atom for [SII]
            SII = pn.Atom('S', 2)  # 'S', ionization stage 2 ([SII])

            # Calculate electron density using the ratio
            n_e = SII.getTemDen(ratio, tem=T_e, wave1=6716.44, wave2=6730.82)

            print(f"DEBUG: Calculated electron density n_e = {n_e:.2e} cm^-3 using PyNeb.")
            return n_e
        except Exception as e:
            print(f"ERROR: PyNeb encountered an error during electron density calculation: {e}")
            return np.nan
    else:
        # Required lines not detected
        print("DEBUG: Required [SII] lines not detected or fit failed. Electron density cannot be calculated.")
        return np.nan

# ======================================================================
# New Function to Plot Zoomed-In Peaks
# ======================================================================

def plot_zoomed_peaks(wavelength, intensity, peaks, gaussian_fits, output_file, fitting_window, title=None, peak_matches=None):
    """
    Creates a subplot for each fitted peak, zoomed in around the peak region,
    and overlays the observed data with the Gaussian fit.

    Parameters:
        wavelength (np.ndarray): Wavelength array.
        intensity (np.ndarray): Intensity array.
        peaks (np.ndarray): Indices of detected peaks.
        gaussian_fits (list of dict): List containing Gaussian fit parameters for each peak.
        output_file (str): Path to save the plot.
        fitting_window (int): Number of pixels on each side of a peak to include in the plot.
        title (str, optional): Title of the plot.
        peak_matches (list of list of dict): List containing matched spectral lines for each peak.
    """
    import matplotlib.pyplot as plt

    num_peaks = len(peaks)
    # Determine grid size for subplots
    ncols = 3  # You can adjust this based on preference
    nrows = (num_peaks + ncols - 1) // ncols  # Ceiling division

    plt.figure(figsize=(5 * ncols, 4 * nrows))
    for idx, peak_idx in enumerate(peaks):
        ax = plt.subplot(nrows, ncols, idx + 1)
        fit = gaussian_fits[idx]
        if fit['success']:
            # Define the plotting window
            left = max(peak_idx - fitting_window, 0)
            right = min(peak_idx + fitting_window + 1, len(intensity))
            x_data = wavelength[left:right]
            y_data = intensity[left:right]
            x_fit = fit['x_fit']
            y_fit = fit['y_fit']

            ax.plot(x_data, y_data, 'k-', label='Data')
            ax.plot(x_fit, y_fit, 'r--', label='Gaussian Fit')

            # Match spectral lines
            matches = peak_matches[idx] if peak_matches is not None else []
            if matches:
                # Choose the best match based on the smallest delta
                best_match = min(matches, key=lambda m: abs(m['Delta']))
                line_name = best_match['Line']
                ax.set_title(f"Peak {idx+1}: {line_name}")
            else:
                ax.set_title(f"Peak {idx+1}")

            ax.set_xlabel('Wavelength (Å)')
            ax.set_ylabel('Intensity')
            ax.legend()
            ax.grid(True)
        else:
            ax.set_title(f"Peak {idx+1}: Fit Failed")
            ax.plot([], [])  # Empty plot
            ax.set_xlabel('Wavelength (Å)')
            ax.set_ylabel('Intensity')
            ax.grid(True)

    if title:
        plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    plt.savefig(output_file, dpi=300)
    plt.close()

# ======================================================================
# Main Function
# ======================================================================

def main():
    # [As before, no changes to argument parsing]
    parser = argparse.ArgumentParser(description="Analyze Stacked Spectra and Plot Peaks with Gaussian Fits")
    parser.add_argument('--input_dir', type=str, default='spectrum_reduction/Reduced/stacked',
                        help='Directory containing stacked FITS spectra (default: spectrum_reduction/Reduced/stacked)')
    parser.add_argument('--output_dir', type=str, default='Reduced/stacked/plots',
                        help='Directory to save the plots (default: Reduced/stacked/plots)')
    parser.add_argument('--calibration_file', type=str, default='spectrum_reduction/reduced/wavelength_calibration.dat',
                        help='Path to the wavelength calibration file (default: spectrum_reduction/reduced/wavelength_calibration.dat)')
    parser.add_argument('--height_sigma', type=float, default=0.1,
                        help='Number of standard deviations above the mean for peak detection (default: 0.1)')
    parser.add_argument('--distance', type=float, default=10,
                        help='Minimum distance between peaks in pixels (default: 10)')
    parser.add_argument('--prominence', type=float, default=180,
                        help='Minimum prominence of peaks (default: 180)')
    parser.add_argument('--fitting_window', type=int, default=20,
                        help='Number of pixels on each side of a peak to include in Gaussian fit (default: 20)')
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
        sys.exit(1)

    # Find all FITS files in the input directory
    fits_files = glob.glob(os.path.join(input_dir, '*.fits'))
    if not fits_files:
        print(f"No FITS files found in directory: {input_dir}")
        sys.exit(1)

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
                fit = fit_gaussian_plus_bg(wavelength, data, peak_idx, fitting_window)
                gaussian_fits.append(fit)
                if fit['success']:
                    print(f"  Peak at index {peak_idx} fitted successfully: background={fit['background']:.2f}, amplitude={fit['amplitude']:.2f}, mean={fit['mean']:.2f} Å, sigma={fit['sigma']:.2f} Å")
                else:
                    print(f"  Peak at index {peak_idx} fit failed.")

            # Match peaks to spectral lines and compute redshift and velocity
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

            # Plot zoomed-in peaks
            zoomed_plot_filename = os.path.join(output_dir, f"{base_name}_zoomed_peaks.{save_format}")
            title_zoomed = f"Zoomed-in Fits for: {base_name}"
            plot_zoomed_peaks(wavelength, data, peaks, gaussian_fits, zoomed_plot_filename, fitting_window, title=title_zoomed, peak_matches=peak_matches)
            print(f"Zoomed-in plot saved to {zoomed_plot_filename}")

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

            # Calculate electron density if possible
            electron_density = calculate_electron_density(gaussian_fits, SPECTRAL_LINES, ratio_tolerance=1.0)

            summary.append({
                'file': os.path.basename(fits_file),
                'num_peaks': num_peaks,
                'peaks': peak_details,
                'electron_density': electron_density
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
                f.write("  Peak Index | Wavelength (Å) | Amplitude | Sigma (Å) | Fit Success | Spectral Line    | Line Wavelength (Å)  | Delta (Å) | z          | v (km/s)         \n")
                f.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------\n")
                for peak in entry['peaks']:
                    # Use the updated helper function to format each peak detail
                    peak_detail_str = format_peak_detail(peak, SPECTRAL_LINES, tolerance=7.0)
                    f.write(peak_detail_str)
                # Include electron density if available
                if not np.isnan(entry['electron_density']):
                    f.write(f"\nElectron Density (n_e): {entry['electron_density']:.2e} cm^-3\n")
                else:
                    f.write(f"\nElectron Density (n_e): N/A (Required [SII] lines not detected or fit failed)\n")
                f.write("\n")
        print(f"Summary of peak detections, Gaussian fits, and electron densities saved to {summary_file}")
    except Exception as e:
        print(f"Error writing summary file: {e}")

    print("All files processed.")

# ======================================================================
# Entry Point
# ======================================================================

if __name__ == '__main__':
    main()