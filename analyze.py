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

For each plot, the script also generates an additional file containing subplots
zoomed in on each fitted line to visually inspect the quality of the fit.

Usage:
    python analyze_stacked_spectra.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]
                                     [--calibration_file CALIBRATION_FILE]
                                     [--height_sigma HEIGHT_SIGMA] [--distance DISTANCE]
                                     [--prominence PROMINENCE] [--fitting_window_factor FITTING_WINDOW_FACTOR]
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
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from astropy.constants import c  # Import speed of light
import sys
import csv  # Added for CSV writing

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
    Detects peaks in the spectrum using specified parameters and estimates their widths.

    Parameters:
        wavelength (np.ndarray): Wavelength array.
        intensity (np.ndarray): Intensity array.
        height (float): Minimum height of peaks.
        distance (float): Minimum distance between peaks in pixels.
        prominence (float): Minimum prominence of peaks.

    Returns:
        tuple:
            np.ndarray: Indices of detected peaks.
            np.ndarray: Widths of the detected peaks in pixels.
    """
    peaks, properties = find_peaks(intensity, height=height, distance=distance, prominence=prominence)
    results_half = peak_widths(intensity, peaks, rel_height=0.5)
    widths = results_half[0]  # Widths in pixels at half prominence
    return peaks, widths

def gaussian_plus_bg(x, bg, amp, cen, wid):
    """
    Gaussian function with a constant background.

    Parameters:
        x (np.ndarray): Independent variable.
        bg (float): Background level.
        amp (float): Amplitude of the Gaussian.
        cen (float): Center of the Gaussian.
        wid (float): Width (sigma) of the Gaussian.

    Returns:
        np.ndarray: Model evaluated at x.
    """
    return bg + amp * np.exp(-0.5 * ((x - cen) / wid) ** 2)

def fit_gaussian_plus_bg(wavelength, intensity, peak_idx, window):
    """
    Fits a Gaussian with a constant background to the data around a peak.

    Parameters:
        wavelength (np.ndarray): Wavelength array.
        intensity (np.ndarray): Intensity array.
        peak_idx (int): Index of the peak.
        window (int): Number of pixels on each side of the peak to include in the fit.

    Returns:
        dict: Fit parameters (background, amplitude, mean, sigma, flux) and fit success status.
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
        popt, pcov = curve_fit(gaussian_plus_bg, x_data, y_data, p0=initial_params)
        fit_success = True

        # Compute uncertainties from covariance matrix
        perr = np.sqrt(np.diag(pcov))
    except (RuntimeError, ValueError):
        popt = [np.nan, np.nan, np.nan, np.nan]
        perr = [np.nan, np.nan, np.nan, np.nan]
        pcov = None
        fit_success = False

    # Calculate flux and its uncertainty if fit is successful
    if fit_success:
        bg, amp, mean, sigma = popt
        bg_err, amp_err, mean_err, sigma_err = perr

        flux = amp * sigma * np.sqrt(2 * np.pi)

        # Compute uncertainty in flux, including covariance
        if pcov is not None and not np.isnan(pcov[1,3]):
            cov_amp_sigma = pcov[1,3]
            flux_var = (sigma * np.sqrt(2 * np.pi))**2 * amp_err**2 + \
                       (amp * np.sqrt(2 * np.pi))**2 * sigma_err**2 + \
                       2 * (sigma * np.sqrt(2 * np.pi)) * (amp * np.sqrt(2 * np.pi)) * cov_amp_sigma
            flux_err = np.sqrt(abs(flux_var))  # Ensure flux_var is positive
        else:
            flux_err = np.nan
    else:
        flux = np.nan
        flux_err = np.nan
        bg_err, amp_err, mean_err, sigma_err = [np.nan]*4

    return {
        'background': popt[0],
        'amplitude': popt[1],
        'mean': popt[2],
        'sigma': popt[3],
        'flux': flux,
        'background_err': bg_err,
        'amplitude_err': amp_err,
        'mean_err': mean_err,
        'sigma_err': sigma_err,
        'flux_err': flux_err,
        'success': fit_success,
        'x_fit': x_data,
        'y_fit': gaussian_plus_bg(x_data, *popt) if fit_success else None
    }

def multi_gaussian_plus_bg(x, bg, *params):
    """
    Sum of multiple Gaussians with a constant background.
    Parameters:
        x (np.ndarray): Independent variable.
        bg (float): Background level.
        params: List of parameters for each Gaussian [amp1, cen1, wid1, amp2, cen2, wid2, ...]
    Returns:
        np.ndarray: Model evaluated at x.
    """
    y = bg
    num_gaussians = len(params) // 3
    for i in range(num_gaussians):
        amp = params[3 * i]
        cen = params[3 * i + 1]
        wid = params[3 * i + 2]
        y += amp * np.exp(-0.5 * ((x - cen) / wid) ** 2)
    return y

def fit_gaussian_plus_bg(wavelength, intensity, peak_indices, window):
    """
    Fits one or multiple Gaussians with a constant background to the data around one or more peaks.
    Parameters:
        wavelength (np.ndarray): Wavelength array.
        intensity (np.ndarray): Intensity array.
        peak_indices (list): Indices of the peaks in the cluster.
        window (int): Number of pixels on each side of the cluster to include in the fit.
    Returns:
        list of dict: Fit parameters for each peak and fit success status.
    """
    # Define the fitting window
    left = max(min(peak_indices) - window, 0)
    right = min(max(peak_indices) + window + 1, len(intensity))
    x_data = wavelength[left:right]
    y_data = intensity[left:right]

    num_peaks = len(peak_indices)

    # Initial guesses
    bg_guess = np.median(intensity)
    initial_params = [bg_guess]

    for peak_idx in peak_indices:
        amplitude_guess = intensity[peak_idx] - bg_guess
        mean_guess = wavelength[peak_idx]
        if len(x_data) > 1:
            delta_wavelength = wavelength[1] - wavelength[0]
        else:
            delta_wavelength = 1  # Prevent division by zero
        sigma_guess = delta_wavelength * window / (2.355 * num_peaks)  # Adjust sigma_guess based on window and number of peaks

        initial_params.extend([amplitude_guess, mean_guess, sigma_guess])

    try:
        popt, pcov = curve_fit(
            multi_gaussian_plus_bg,
            x_data,
            y_data,
            p0=initial_params,
            maxfev=10000
        )
        fit_success = True

        # Compute uncertainties from covariance matrix
        perr = np.sqrt(np.diag(pcov))
    except (RuntimeError, ValueError) as e:
        print(f"Fit failed for peaks at indices {peak_indices}: {e}")
        popt = [np.nan] * len(initial_params)
        perr = [np.nan] * len(initial_params)
        pcov = None
        fit_success = False

    # Extract parameters for each Gaussian
    fits = []
    if fit_success:
        bg = popt[0]
        bg_err = perr[0]
        num_gaussians = (len(popt) - 1) // 3
        for i in range(num_gaussians):
            amp = popt[1 + 3 * i]
            cen = popt[1 + 3 * i + 1]
            wid = popt[1 + 3 * i + 2]
            amp_err = perr[1 + 3 * i]
            cen_err = perr[1 + 3 * i + 1]
            wid_err = perr[1 + 3 * i + 2]
            flux = amp * wid * np.sqrt(2 * np.pi)
            # Compute uncertainty in flux, including covariance
            # Note: Extract covariances from pcov
            if pcov is not None:
                idx_amp = 1 + 3 * i
                idx_wid = idx_amp + 2
                cov_amp_wid = pcov[idx_amp, idx_wid]
                flux_var = (wid * np.sqrt(2 * np.pi))**2 * amp_err**2 + \
                           (amp * np.sqrt(2 * np.pi))**2 * wid_err**2 + \
                           2 * (wid * np.sqrt(2 * np.pi)) * (amp * np.sqrt(2 * np.pi)) * cov_amp_wid
                flux_err = np.sqrt(abs(flux_var))
            else:
                flux_err = np.nan

            fits.append({
                'background': bg,
                'background_err': bg_err,
                'amplitude': amp,
                'amplitude_err': amp_err,
                'mean': cen,
                'mean_err': cen_err,
                'sigma': wid,
                'sigma_err': wid_err,
                'flux': flux,
                'flux_err': flux_err,
                'success': True,
                'x_fit': x_data,
                'y_fit': None  # We'll compute individual y_fit below
            })
    else:
        # Create placeholder fits with failure status
        for _ in peak_indices:
            fits.append({
                'background': np.nan,
                'background_err': np.nan,
                'amplitude': np.nan,
                'amplitude_err': np.nan,
                'mean': np.nan,
                'mean_err': np.nan,
                'sigma': np.nan,
                'sigma_err': np.nan,
                'flux': np.nan,
                'flux_err': np.nan,
                'success': False,
                'x_fit': None,
                'y_fit': None
            })

    # Compute y_fit for each individual Gaussian
    if fit_success:
        for i, fit in enumerate(fits):
            amp = fit['amplitude']
            cen = fit['mean']
            wid = fit['sigma']
            fit['x_fit'] = x_data
            fit['y_fit'] = fit['background'] + amp * np.exp(-0.5 * ((x_data - cen) / wid) ** 2)

    return fits


def group_peaks(peaks, cluster_distance):
    """
    Groups peaks that are close to each other into clusters.
    Parameters:
        peaks (np.ndarray): Array of peak indices.
        cluster_distance (float): Maximum distance between peaks to be considered in the same cluster.
    Returns:
        list of list: List of clusters, each cluster is a list of peak indices.
    """
    clusters = []
    current_cluster = [peaks[0]]

    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i - 1] <= cluster_distance:
            current_cluster.append(peaks[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [peaks[i]]
    clusters.append(current_cluster)
    return clusters


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

def compute_redshift(observed_wavelength, rest_wavelength, observed_wavelength_err=None):
    """
    Computes the redshift based on observed and rest wavelengths.
    If observed_wavelength_err is provided, computes the uncertainty in redshift.

    Parameters:
        observed_wavelength (float): Observed wavelength in Angstroms.
        rest_wavelength (float): Rest (air) wavelength in Angstroms.
        observed_wavelength_err (float, optional): Uncertainty in observed wavelength.

    Returns:
        tuple: (z, z_err) if observed_wavelength_err is provided, else z
    """
    if rest_wavelength <= 0:
        z = np.nan
        z_err = np.nan
    else:
        z = (observed_wavelength / rest_wavelength) - 1
        if observed_wavelength_err is not None:
            z_err = observed_wavelength_err / rest_wavelength
        else:
            z_err = None
    if observed_wavelength_err is not None:
        return z, z_err
    else:
        return z

def compute_velocity(z, z_err=None):
    """
    Computes the velocity based on redshift.
    If z_err is provided, computes the uncertainty in velocity.

    Parameters:
        z (float): Redshift.
        z_err (float, optional): Uncertainty in redshift.

    Returns:
        tuple: (v, v_err) if z_err is provided, else v
    """
    v = z * c.to('km/s').value
    if z_err is not None:
        v_err = z_err * c.to('km/s').value
        return v, v_err
    else:
        return v


def plot_spectrum(wavelength, intensity, peaks, gaussian_fits, output_file, title=None, peak_matches=None):
    """
    Plots the spectrum with identified peaks and their Gaussian fits, including individually plotted
    Gaussians for overlapping peaks, and vertically oriented labels for matched spectral lines.

    Parameters:
        wavelength (np.ndarray): Wavelength array.
        intensity (np.ndarray): Intensity array.
        peaks (list): Indices of detected peaks.
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

    # Plot individual Gaussian fits
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
                    z, z_err = compute_redshift(peak_wavelength, line_wavelength, observed_wavelength_err=fit['mean_err'])
                    v, v_err = compute_velocity(z, z_err=z_err)

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

# ======================================================================
# Electron Density Calculation Function
# ======================================================================

def calculate_electron_density(fit_results, peak_matches):
    """
    Calculates the electron density using the flux ratio of [SII] 6717 and [SII] 6731 lines.

    Parameters:
        fit_results (list of dict): List of Gaussian fit results for all peaks.
        peak_matches (list of list of dict): List containing matched spectral lines for each peak.

    Returns:
        float: Electron density in cm^-3, or np.nan if calculation is not possible.
    """
    try:
        import pyneb as pn
    except ImportError:
        print("PyNeb is not installed. Please install it using 'pip install pyneb'.")
        return np.nan

    flux_6717 = None
    flux_6717_err = None
    flux_6731 = None
    flux_6731_err = None

    for idx, fit in enumerate(fit_results):
        if not fit['success']:
            continue

        matches = peak_matches[idx]
        for match in matches:
            line_name = match['Line']
            if line_name == "[SII]6717":
                flux_6717 = fit['flux']
                flux_6717_err = fit['flux_err']
            elif line_name == "[SII]6731":
                flux_6731 = fit['flux']
                flux_6731_err = fit['flux_err']

    # Debug: Check if both [SII] lines were found
    print(f"DEBUG: Flux [SII]6717 = {flux_6717}, Flux [SII]6731 = {flux_6731}")
    if flux_6717 is not None:
        print(f"DEBUG: Detected [SII]6717 with flux {flux_6717:.2e}.")
    else:
        print("DEBUG: [SII]6717 not detected.")

    if flux_6731 is not None:
        print(f"DEBUG: Detected [SII]6731 with flux {flux_6731:.2e}.")
    else:
        print("DEBUG: [SII]6731 not detected.")

    # Check if both fluxes are found and flux_6731 is not zero
    if flux_6717 is not None and flux_6731 is not None:
        if flux_6731 == 0:
            print("DEBUG: [SII]6731 flux is zero. Cannot compute flux ratio.")
            return np.nan

        ratio = flux_6717 / flux_6731
        # Propagate uncertainty in ratio
        ratio_err = ratio * np.sqrt((flux_6717_err / flux_6717) ** 2 + (flux_6731_err / flux_6731) ** 2)
        print(f"DEBUG: Flux ratio [SII]6717/[SII]6731 = {ratio:.4f} ± {ratio_err:.4f}")

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
            # Calculate error in electron density (approximate method)
            n_e_plus = SII.getTemDen(ratio + ratio_err, tem=T_e, wave1=6716.44, wave2=6730.82)
            n_e_minus = SII.getTemDen(ratio - ratio_err, tem=T_e, wave1=6716.44, wave2=6730.82)
            n_e_err = (n_e_plus - n_e_minus) / 2

            print(f"DEBUG: Calculated electron density n_e = {n_e:.2e} ± {n_e_err:.2e} cm^-3 using PyNeb.")
            return n_e, n_e_err
        except Exception as e:
            print(f"ERROR: PyNeb encountered an error during electron density calculation: {e}")
            return np.nan, np.nan
    else:
        # Required lines not detected
        print("DEBUG: Required [SII] lines not detected or fit failed. Electron density cannot be calculated.")
        return np.nan, np.nan

# ======================================================================
# Function to Plot Zoomed-In Peaks
# ======================================================================

def plot_zoomed_peaks(wavelength, intensity, peaks, gaussian_fits, output_file, fitting_windows, title=None, peak_matches=None):
    """
    Creates a subplot for each fitted peak, zoomed in around the peak region,
    and overlays the observed data with the Gaussian fit.

    Parameters:
        wavelength (np.ndarray): Wavelength array.
        intensity (np.ndarray): Intensity array.
        peaks (list): Indices of detected peaks.
        gaussian_fits (list of dict): List containing Gaussian fit parameters for each peak.
        output_file (str): Path to save the plot.
        fitting_windows (list of int): List of window sizes for each peak.
        title (str, optional): Title of the plot.
        peak_matches (list of list of dict): List containing matched spectral lines for each peak.
    """
    import matplotlib.pyplot as plt

    num_peaks = len(peaks)
    if num_peaks == 0:
        print("No peaks to plot zoomed-in fits.")
        return

    # Determine grid size for subplots
    ncols = 3  # You can adjust this based on preference
    nrows = (num_peaks + ncols - 1) // ncols  # Ceiling division

    plt.figure(figsize=(5 * ncols, 4 * nrows))
    for idx in range(num_peaks):
        peak_idx = peaks[idx]
        fit = gaussian_fits[idx]
        window = fitting_windows[idx]
        ax = plt.subplot(nrows, ncols, idx + 1)
        if fit['success']:
            # Define the plotting window
            left = max(peak_idx - window, 0)
            right = min(peak_idx + window + 1, len(intensity))
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
    # Argument Parser
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
    parser.add_argument('--cluster_distance', type=float, default=50,
                        help='Maximum distance between peaks to be considered in the same cluster (default: 5)')
    parser.add_argument('--save_format', type=str, choices=['png', 'pdf'], default='png',
                        help='Format to save plots (default: png)')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    calibration_file = args.calibration_file
    height_sigma = args.height_sigma
    distance = args.distance
    prominence = args.prominence
    cluster_distance = args.cluster_distance
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

            # Detect peaks with dynamic height and estimate their widths
            peaks, widths = detect_peaks(wavelength, data, height=dynamic_height, distance=distance, prominence=prominence)
            num_peaks = len(peaks)
            print(f"Detected {num_peaks} peaks with height > mean + {height_sigma}*std.")

            # Group peaks into clusters
            clusters = group_peaks(peaks, cluster_distance)

            # Prepare lists to collect all individual peak fits and indices
            all_gaussian_fits = []
            all_peak_indices = []
            all_fitting_windows = []
            peak_matches = []

            for cluster in clusters:
                # For clusters, use the average width to define the fitting window
                cluster_widths = [widths[np.where(peaks == idx)[0][0]] for idx in cluster]
                avg_width = np.mean(cluster_widths)
                window = int(np.ceil(avg_width * 2.5))
                # Fit the cluster with multiple Gaussians
                fits = fit_gaussian_plus_bg(wavelength, data, cluster, window)
                all_gaussian_fits.extend(fits)
                all_peak_indices.extend(cluster)
                all_fitting_windows.extend([window] * len(cluster))
                for idx, fit in zip(cluster, fits):
                    if fit['success']:
                        print(f"  Peak at index {idx} in cluster fitted successfully:")
                        print(f"    Background={fit['background']:.2f} ± {fit['background_err']:.2f}")
                        print(f"    Amplitude={fit['amplitude']:.2f} ± {fit['amplitude_err']:.2f}")
                        print(f"    Mean={fit['mean']:.2f} Å ± {fit['mean_err']:.2f} Å")
                        print(f"    Sigma={fit['sigma']:.2f} Å ± {fit['sigma_err']:.2f} Å")
                        print(f"    Flux={fit['flux']:.2e} cm^-1 ± {fit['flux_err']:.2e} cm^-1")
                        print(f"    Window={window} pixels")
                    else:
                        print(f"  Peak at index {idx} in cluster fit failed.")
                # Match spectral lines for each peak in the cluster
                for fit in fits:
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
            plot_spectrum(wavelength, data, all_peak_indices, all_gaussian_fits, plot_filename, title=title, peak_matches=peak_matches)
            print(f"Plot saved to {plot_filename}")

            # Plot zoomed-in peaks
            zoomed_plot_filename = os.path.join(output_dir, f"{base_name}_zoomed_peaks.{save_format}")
            title_zoomed = f"Zoomed-in Fits for: {base_name}"
            plot_zoomed_peaks(wavelength, data, all_peak_indices, all_gaussian_fits, zoomed_plot_filename, fitting_windows=all_fitting_windows, title=title_zoomed, peak_matches=peak_matches)
            print(f"Zoomed-in plot saved to {zoomed_plot_filename}")

            # Append to summary
            peak_details = []
            for idx, fit in zip(all_peak_indices, all_gaussian_fits):
                peak_info = {
                    'peak_index': int(idx),
                    'background': fit['background'] if fit['success'] else np.nan,
                    'background_err': fit['background_err'] if fit['success'] else np.nan,
                    'amplitude': fit['amplitude'] if fit['success'] else np.nan,
                    'amplitude_err': fit['amplitude_err'] if fit['success'] else np.nan,
                    'mean': fit['mean'] if fit['success'] else np.nan,
                    'mean_err': fit['mean_err'] if fit['success'] else np.nan,
                    'sigma': fit['sigma'] if fit['success'] else np.nan,
                    'sigma_err': fit['sigma_err'] if fit['success'] else np.nan,
                    'flux': fit['flux'] if fit['success'] else np.nan,
                    'flux_err': fit['flux_err'] if fit['success'] else np.nan,
                    'window': window,  # All peaks in the cluster have the same window
                    'fit_success': fit['success'],
                    'spectral_lines': peak_matches[all_peak_indices.index(idx)]  # Add matched spectral lines
                }
                peak_details.append(peak_info)

            # Calculate electron density if possible
            electron_density, electron_density_err = calculate_electron_density(all_gaussian_fits, peak_matches)

            summary.append({
                'file': os.path.basename(fits_file),
                'num_peaks': num_peaks,
                'peaks': peak_details,
                'electron_density': electron_density,
                'electron_density_err': electron_density_err
            })

        except Exception as e:
            print(f"Error processing {fits_file}: {e}")
            continue

    # Save summary to a CSV file
    summary_file = os.path.join(output_dir, 'peak_summary.csv')
    try:
        with open(summary_file, 'w', newline='') as csvfile:
            fieldnames = [
                'File',
                'Number of Peaks',
                'Electron Density (cm^-3)',
                'Electron Density Error',
                'Peak Index',
                'Background',
                'Background Error',
                'Amplitude',
                'Amplitude Error',
                'Mean (Å)',
                'Mean Error (Å)',
                'Sigma (Å)',
                'Sigma Error (Å)',
                'Flux (cm^-1)',
                'Flux Error (cm^-1)',
                'Window (pixels)',
                'Fit Success',
                'Spectral Line',
                'Line Wavelength (Å)',
                'Delta (Å)',
                'z',
                'z Error',
                'v (km/s)',
                'v Error (km/s)'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for entry in summary:
                file_name = entry['file']
                num_peaks = entry['num_peaks']
                electron_density = entry['electron_density']
                electron_density_err = entry['electron_density_err']
                for peak in entry['peaks']:
                    row = {
                        'File': file_name,
                        'Number of Peaks': num_peaks,
                        'Electron Density (cm^-3)': f"{electron_density:.2e}" if not np.isnan(electron_density) else "N/A",
                        'Electron Density Error': f"{electron_density_err:.2e}" if not np.isnan(electron_density_err) else "N/A",
                        'Peak Index': peak['peak_index'],
                        'Background': f"{peak['background']:.2f}" if not np.isnan(peak['background']) else "N/A",
                        'Background Error': f"{peak['background_err']:.2f}" if not np.isnan(peak['background_err']) else "N/A",
                        'Amplitude': f"{peak['amplitude']:.2f}" if not np.isnan(peak['amplitude']) else "N/A",
                        'Amplitude Error': f"{peak['amplitude_err']:.2f}" if not np.isnan(peak['amplitude_err']) else "N/A",
                        'Mean (Å)': f"{peak['mean']:.2f}" if not np.isnan(peak['mean']) else "N/A",
                        'Mean Error (Å)': f"{peak['mean_err']:.2f}" if not np.isnan(peak['mean_err']) else "N/A",
                        'Sigma (Å)': f"{peak['sigma']:.2f}" if not np.isnan(peak['sigma']) else "N/A",
                        'Sigma Error (Å)': f"{peak['sigma_err']:.2f}" if not np.isnan(peak['sigma_err']) else "N/A",
                        'Flux (cm^-1)': f"{peak['flux']:.2e}" if not np.isnan(peak['flux']) else "N/A",
                        'Flux Error (cm^-1)': f"{peak['flux_err']:.2e}" if not np.isnan(peak['flux_err']) else "N/A",
                        'Window (pixels)': peak['window'],
                        'Fit Success': "Yes" if peak['fit_success'] else "No",
                        'Spectral Line': "",
                        'Line Wavelength (Å)': "",
                        'Delta (Å)': "",
                        'z': "",
                        'z Error': "",
                        'v (km/s)': "",
                        'v Error (km/s)': ""
                    }

                    if peak['spectral_lines']:
                        # Assuming only one best match per peak
                        best_match = min(peak['spectral_lines'], key=lambda m: abs(m['Delta']))
                        line_name = best_match['Line']
                        line_wavelength = best_match['Wavelength']
                        delta = best_match['Delta']
                        z, z_err = compute_redshift(peak['mean'], line_wavelength, observed_wavelength_err=peak['mean_err'])
                        v, v_err = compute_velocity(z, z_err=z_err)

                        row['Spectral Line'] = line_name
                        row['Line Wavelength (Å)'] = f"{line_wavelength:.2f}"
                        row['Delta (Å)'] = f"{delta:.2f}"
                        row['z'] = f"{z:.6f}"
                        row['z Error'] = f"{z_err:.6f}" if z_err is not None else "N/A"
                        row['v (km/s)'] = f"{v:.2f}"
                        row['v Error (km/s)'] = f"{v_err:.2f}" if v_err is not None else "N/A"
                    else:
                        row['Spectral Line'] = "None"
                        row['Line Wavelength (Å)'] = "None"
                        row['Delta (Å)'] = "None"
                        row['z'] = "None"
                        row['z Error'] = "None"
                        row['v (km/s)'] = "None"
                        row['v Error (km/s)'] = "None"

                    writer.writerow(row)
        print(f"Summary of peak detections, Gaussian fits, fluxes, and electron densities saved to {summary_file}")
    except Exception as e:
        print(f"Error writing summary file: {e}")

    print("All files processed.")

# ======================================================================
# Entry Point
# ======================================================================

if __name__ == '__main__':
    main()