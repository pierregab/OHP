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

For [OI] and [NII] spectral lines, the script calculates and reports the flux ratios
([O I]6300/[O I]6363 and [NII]6548/[NII]6583) when applicable.

For each plot, the script also generates an additional file containing subplots
zoomed in on each fitted line group to visually inspect the quality of the fit.

Usage:
    python analyze_stacked_spectra.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]
                                     [--calibration_file CALIBRATION_FILE]
                                     [--height_sigma HEIGHT_SIGMA] [--distance DISTANCE]
                                     [--prominence PROMINENCE] [--cluster_distance CLUSTER_DISTANCE]
                                     [--save_format SAVE_FORMAT]
                                     [--target_name TARGET_NAME]

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
    - astroplan

Ensure all required dependencies are installed. You can install missing packages
using pip:
    pip install numpy matplotlib astropy scipy pyneb adjustText astroplan
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
import astropy.units as u  # For units
from astropy.time import Time  # For observation time
from astropy.coordinates import SkyCoord, EarthLocation  # For coordinates
from astroplan import FixedTarget  # For target information
import pyneb as pn  # For electron density calculations
import logging

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
    {"Line": "[NII]6548", "Wavelength_air": 6548.03, "Wavelength_vacuum": 6549.84, "Notes": ""},
    {"Line": "[NII]6583", "Wavelength_air": 6583.41, "Wavelength_vacuum": 6585.23, "Notes": ""},
    {"Line": "He I", "Wavelength_air": 6678.152, "Wavelength_vacuum": 6679.996, "Notes": ""},
    {"Line": "[SII]6717", "Wavelength_air": 6716.47, "Wavelength_vacuum": 6718.32, "Notes": ""},
    {"Line": "[SII]6731", "Wavelength_air": 6730.85, "Wavelength_vacuum": 6732.71, "Notes": ""},
    {"Line": "[SIII]", "Wavelength_air": 6312.06, "Wavelength_vacuum": 6313.81, "Notes": "Use with [SIII]9068 as T diagnostic"},
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
    def multi_gaussian_plus_bg(x, bg, *params):
        y = bg
        num_gaussians = len(params) // 3
        for i in range(num_gaussians):
            amp = params[3 * i]
            cen = params[3 * i + 1]
            wid = params[3 * i + 2]
            y += amp * np.exp(-0.5 * ((x - cen) / wid) ** 2)
        return y

    # Define the fitting window
    left = max(np.searchsorted(wavelength, wavelength[min(peak_indices)] - window * (wavelength[1] - wavelength[0])), 0)
    right = min(np.searchsorted(wavelength, wavelength[max(peak_indices)] + window * (wavelength[1] - wavelength[0])), len(intensity))
    x_data = wavelength[left:right]
    y_data = intensity[left:right]

    num_peaks = len(peak_indices)

    # Initial guesses
    bg_guess = np.median(intensity)
    initial_params = [bg_guess]

    # Define bounds
    lower_bounds = [0]  # Background must be >= 0
    upper_bounds = [np.inf]

    for peak_idx in peak_indices:
        amplitude_guess = intensity[peak_idx] - bg_guess
        amplitude_guess = max(amplitude_guess, 0)  # Amplitude should be positive
        mean_guess = wavelength[peak_idx]
        # Set a small initial guess for sigma
        sigma_guess = 1.0  # Adjust this value as needed

        initial_params.extend([amplitude_guess, mean_guess, sigma_guess])

        # Define bounds for amplitude, center, and sigma
        lower_bounds.extend([0, mean_guess - window * (wavelength[1] - wavelength[0]), 0.5])
        upper_bounds.extend([np.inf, mean_guess + window * (wavelength[1] - wavelength[0]), 10.0])

    try:
        popt, pcov = curve_fit(
            multi_gaussian_plus_bg,
            x_data,
            y_data,
            p0=initial_params,
            bounds=(lower_bounds, upper_bounds),
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
            if pcov is not None:
                idx_amp = 1 + 3 * i
                idx_wid = idx_amp + 2
                cov_amp_wid = pcov[idx_amp, idx_wid]
                flux_var = (wid * np.sqrt(2 * np.pi))**2 * amp_err**2 + \
                           (amp * np.sqrt(2 * np.pi))**2 * wid_err**2 + \
                           2 * (wid * np.sqrt(2 * np.pi)) * (amp * np.sqrt(2 * np.pi)) * cov_amp_wid
                flux_err = np.sqrt(abs(flux_var)) if flux_var > 0 else np.nan
            else:
                flux_err = np.nan

            # Calculate redshift and velocity if spectral lines are matched
            matches = match_spectral_lines(cen, SPECTRAL_LINES, tolerance=7.0)
            if matches:
                # Choose the best match based on smallest delta
                best_match = min(matches, key=lambda m: abs(m['Delta']))
                rest_wavelength = best_match['Wavelength']
                z, z_err = compute_redshift(cen, rest_wavelength, observed_wavelength_err=cen_err)
                v, v_err = compute_velocity(z, z_err=z_err)
            else:
                z, z_err, v, v_err = (np.nan, np.nan, np.nan, np.nan)

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
                'z': z,
                'z_err': z_err,
                'v': v,
                'v_err': v_err,
                'success': True,
                'x_fit': x_data,
                'y_fit': multi_gaussian_plus_bg(x_data, bg, *popt[1:])
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
                'z': np.nan,
                'z_err': np.nan,
                'v': np.nan,
                'v_err': np.nan,
                'success': False,
                'x_fit': None,
                'y_fit': None
            })

    return fits

def group_peaks(peaks, cluster_distance):
    """
    Groups peaks that are close to each other into clusters.
    Only clusters with three or more peaks within 'cluster_distance' are grouped.
    Clusters with one or two peaks are treated as separate individual clusters.

    Parameters:
        peaks (np.ndarray): Array of peak indices sorted in ascending order.
        cluster_distance (float): Maximum distance between consecutive peaks to be considered in the same cluster.

    Returns:
        list of list: List of clusters, each cluster is a list of peak indices.
    """
    if len(peaks) == 0:
        return []
        
    clusters = []
    current_cluster = [peaks[0]]

    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i - 1] <= cluster_distance:
            current_cluster.append(peaks[i])
        else:
            # Check the size of the current cluster
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
            else:
                # If the cluster has one or two peaks, add each peak as a separate cluster
                for peak in current_cluster:
                    clusters.append([peak])
            # Start a new cluster
            current_cluster = [peaks[i]]

    # Handle the last cluster
    if len(current_cluster) >= 3:
        clusters.append(current_cluster)
    else:
        for peak in current_cluster:
            clusters.append([peak])

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

def compute_barycentric_correction(header, target_name=None, default_obs_date=None):
    """
    Computes the barycentric correction velocity using observation time and target coordinates.

    Parameters:
        header (astropy.io.fits.Header): FITS header containing observation metadata.
        target_name (str, optional): Name of the target object. If not provided, attempts to get coordinates from header.
        default_obs_date (str, optional): Hardcoded observation date in ISO format. Used if header date is missing.

    Returns:
        float: Barycentric correction velocity in km/s.
    """
    try:
        # Get observation time
        date_obs = header.get('DATE-OBS') or header.get('DATE')
        if date_obs is None:
            if default_obs_date is not None:
                date_obs = default_obs_date
                logging.warning(f"Observation date not found in header. Using hardcoded date: {date_obs}")
            else:
                logging.error("Observation date not found in header ('DATE-OBS' or 'DATE'). Returning 0.0 km/s.")
                return 0.0
        try:
            t = Time(date_obs, format='isot', scale='utc')
            logging.debug(f"Parsed observation time: {t.iso}")
        except Exception as e:
            logging.error(f"Error parsing observation time '{date_obs}': {e}. Returning 0.0 km/s.")
            return 0.0

        # Get observer location (OHP coordinates)
        try:
            loc = EarthLocation(lat=43.9346*u.deg, lon=5.7107*u.deg, height=650.*u.m)
            logging.debug(f"Observer location: {loc}")
        except Exception as e:
            logging.error(f"Error defining observer location: {e}. Returning 0.0 km/s.")
            return 0.0

        # Get target coordinates
        if target_name:
            try:
                target = FixedTarget.from_name(target_name)
                sc = SkyCoord(ra=target.ra, dec=target.dec, frame='icrs')
                logging.debug(f"Resolved target '{target_name}': RA={sc.ra.deg} deg, DEC={sc.dec.deg} deg")
            except Exception as e:
                logging.error(f"Error resolving target name '{target_name}': {e}. Returning 0.0 km/s.")
                return 0.0
        else:
            ra = header.get('RA')
            dec = header.get('DEC')
            if ra is None or dec is None:
                logging.error("Target coordinates ('RA' and 'DEC') not found in header. Returning 0.0 km/s.")
                return 0.0
            try:
                # Attempt to parse RA and DEC using SkyCoord, accommodating different formats
                sc = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame='icrs', equinox='J2000')
                logging.debug(f"Parsed target coordinates from header: RA={sc.ra.deg} deg, DEC={sc.dec.deg} deg")
            except ValueError:
                try:
                    sc = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs', equinox='J2000')
                    logging.debug(f"Parsed target coordinates from header: RA={sc.ra.deg} deg, DEC={sc.dec.deg} deg")
                except Exception as e:
                    logging.error(f"Error parsing 'RA' and 'DEC' from header: {e}. Returning 0.0 km/s.")
                    return 0.0

        # Compute barycentric correction
        try:
            vcorr = sc.radial_velocity_correction(kind='barycentric', obstime=t, location=loc)
            vcorr_kms = vcorr.to('km/s').value
            logging.debug(f"Barycentric correction velocity: {vcorr_kms:.2f} km/s")
            return vcorr_kms
        except Exception as e:
            logging.error(f"Error computing barycentric correction: {e}. Returning 0.0 km/s.")
            return 0.0
    except Exception as e:
        logging.error(f"Unexpected error in barycentric correction: {e}. Returning 0.0 km/s.")
        return 0.0

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

def compute_fwhm(sigma, sigma_err=None):
    """
    Computes the Full Width at Half Maximum (FWHM) from the Gaussian sigma.

    Parameters:
        sigma (float): Sigma of the Gaussian.
        sigma_err (float, optional): Uncertainty in sigma.

    Returns:
        tuple: FWHM and its uncertainty.
    """
    fwhm = 2.3548 * sigma  # 2 * sqrt(2 * ln(2)) * sigma
    if sigma_err is not None:
        fwhm_err = 2.3548 * sigma_err
        return fwhm, fwhm_err
    else:
        return fwhm, None

def compute_flux_ratio(flux1, flux1_err, flux2, flux2_err):
    """
    Computes the flux ratio of two lines and propagates the uncertainty.

    Parameters:
        flux1 (float): Flux of the first line.
        flux1_err (float): Uncertainty in flux of the first line.
        flux2 (float): Flux of the second line.
        flux2_err (float): Uncertainty in flux of the second line.

    Returns:
        tuple: Flux ratio and its uncertainty.
    """
    if flux2 == 0:
        return np.nan, np.nan
    ratio = flux1 / flux2
    if flux1 > 0 and flux2 > 0:
        ratio_err = ratio * np.sqrt((flux1_err / flux1) ** 2 + (flux2_err / flux2) ** 2)
    else:
        ratio_err = np.nan
    return ratio, ratio_err

def calculate_electron_density_sii(fit_results, peak_matches):
    """
    Calculates the electron density using the flux ratio of [SII] 6717 and [SII] 6731 lines.

    Parameters:
        fit_results (list of dict): List of Gaussian fit results for all peaks.
        peak_matches (list of list of dict): List containing matched spectral lines for each peak.

    Returns:
        tuple: (float, float) Electron density in cm^-3 and its uncertainty, or (np.nan, np.nan) if calculation is not possible.
    """
    flux_6717 = None
    flux_6717_err = None
    flux_6731 = None
    flux_6731_err = None

    for fit, matches in zip(fit_results, peak_matches):
        if not fit['success']:
            continue

        for match in matches:
            line_name = match['Line']
            if line_name == "[SII]6717":
                flux_6717 = fit['flux']
                flux_6717_err = fit['flux_err']
            elif line_name == "[SII]6731":
                flux_6731 = fit['flux']
                flux_6731_err = fit['flux_err']

    if flux_6717 is not None and flux_6731 is not None:
        if flux_6731 == 0:
            print("DEBUG: [SII]6731 flux is zero. Cannot compute flux ratio.")
            return np.nan, np.nan

        ratio = flux_6717 / flux_6731
        # Propagate uncertainty in ratio
        if flux_6717_err is not None and flux_6731_err is not None:
            ratio_err = ratio * np.sqrt((flux_6717_err / flux_6717) ** 2 + (flux_6731_err / flux_6731) ** 2)
        else:
            ratio_err = np.nan
        print(f"DEBUG: Flux ratio [SII]6717/[SII]6731 = {ratio:.4f} ± {ratio_err:.4f}")

        # Use PyNeb to calculate electron density
        T_e = 10000  # Electron temperature in K
        print(f"DEBUG: Assuming electron temperature T_e = {T_e} K for electron density calculation.")

        try:
            # Initialize PyNeb's Atom for [SII]
            SII = pn.Atom('S', 2)  # 'S', ionization stage 2 ([SII])

            # Calculate electron density using the ratio
            n_e = SII.getTemDen(ratio, tem=T_e, to_eval='L(6716)/L(6731)')
            # Calculate error in electron density (approximate method)
            if not np.isnan(ratio_err):
                ratio_plus = ratio + ratio_err
                ratio_minus = ratio - ratio_err
                n_e_plus = SII.getTemDen(ratio_plus, tem=T_e, to_eval='L(6716)/L(6731)')
                n_e_minus = SII.getTemDen(ratio_minus, tem=T_e, to_eval='L(6716)/L(6731)')
                n_e_err = (n_e_plus - n_e_minus) / 2
            else:
                n_e_err = np.nan

            print(f"DEBUG: Calculated electron density n_e = {n_e:.2e} ± {n_e_err:.2e} cm^-3 using PyNeb.")
            return n_e, n_e_err
        except Exception as e:
            print(f"ERROR: PyNeb encountered an error during electron density calculation: {e}")
            return np.nan, np.nan
    else:
        # Required lines not detected
        print("DEBUG: Required [SII] lines not detected or fit failed. Electron density cannot be calculated.")
        return np.nan, np.nan

def compute_flux_ratio_oi(fit_results, peak_matches, tolerance=7.0):
    """
    Computes the flux ratio [O I]6300/[O I]6363 and its uncertainty.

    Parameters:
        fit_results (list of dict): List of Gaussian fit results for all peaks.
        peak_matches (list of list of dict): List containing matched spectral lines for each peak.
        tolerance (float): Tolerance in Angstroms for matching spectral lines.

    Returns:
        tuple: (flux_ratio, flux_ratio_err) or (np.nan, np.nan) if calculation is not possible.
    """
    flux_6300 = None
    flux_6300_err = None
    flux_6363 = None
    flux_6363_err = None

    for fit, matches in zip(fit_results, peak_matches):
        if not fit['success']:
            continue

        for match in matches:
            line_name = match['Line']
            if line_name == "[O I]6300":
                flux_6300 = fit['flux']
                flux_6300_err = fit['flux_err']
            elif line_name == "[O I]6363":
                flux_6363 = fit['flux']
                flux_6363_err = fit['flux_err']

    if flux_6300 is not None and flux_6363 is not None:
        if flux_6363 == 0:
            print("DEBUG: [O I]6363 flux is zero. Cannot compute flux ratio.")
            return np.nan, np.nan

        ratio, ratio_err = compute_flux_ratio(flux_6300, flux_6300_err, flux_6363, flux_6363_err)
        print(f"DEBUG: Flux ratio [O I]6300/[O I]6363 = {ratio:.4f} ± {ratio_err:.4f}")
        return ratio, ratio_err
    else:
        print("DEBUG: Required [O I] lines not detected or fit failed. Flux ratio cannot be calculated.")
        return np.nan, np.nan

def compute_flux_ratio_nii(fit_results, peak_matches, tolerance=7.0):
    """
    Computes the flux ratio [NII]6548/[NII]6583 and its uncertainty.

    Parameters:
        fit_results (list of dict): List of Gaussian fit results for all peaks.
        peak_matches (list of list of dict): List containing matched spectral lines for each peak.
        tolerance (float): Tolerance in Angstroms for matching spectral lines.

    Returns:
        tuple: (flux_ratio, flux_ratio_err) or (np.nan, np.nan) if calculation is not possible.
    """
    flux_6548 = None
    flux_6548_err = None
    flux_6583 = None
    flux_6583_err = None

    for fit, matches in zip(fit_results, peak_matches):
        if not fit['success']:
            continue

        for match in matches:
            line_name = match['Line']
            if line_name == "[NII]6548":
                flux_6548 = fit['flux']
                flux_6548_err = fit['flux_err']
            elif line_name == "[NII]6583":
                flux_6583 = fit['flux']
                flux_6583_err = fit['flux_err']

    if flux_6548 is not None and flux_6583 is not None:
        if flux_6583 == 0:
            print("DEBUG: [NII]6583 flux is zero. Cannot compute flux ratio.")
            return np.nan, np.nan

        ratio, ratio_err = compute_flux_ratio(flux_6548, flux_6548_err, flux_6583, flux_6583_err)
        print(f"DEBUG: Flux ratio [NII]6548/[NII]6583 = {ratio:.4f} ± {ratio_err:.4f}")
        return ratio, ratio_err
    else:
        print("DEBUG: Required [NII] lines not detected or fit failed. Flux ratio cannot be calculated.")
        return np.nan, np.nan

def plot_zoomed_peaks(wavelength, intensity, cluster_data_list, output_file, title=None):
    """
    Creates a subplot for each cluster of peaks, zoomed in around the cluster region,
    and overlays the observed data with the Gaussian fits.

    Parameters:
        wavelength (np.ndarray): Wavelength array.
        intensity (np.ndarray): Intensity array.
        cluster_data_list (list of dict): List containing data for each cluster.
        output_file (str): Path to save the plot.
        title (str, optional): Title of the plot.
    """
    num_clusters = len(cluster_data_list)
    if num_clusters == 0:
        print("No clusters to plot zoomed-in fits.")
        return

    # Determine grid size for subplots
    ncols = 3  # Adjust as needed
    nrows = (num_clusters + ncols - 1) // ncols  # Ceiling division

    plt.figure(figsize=(5 * ncols, 4 * nrows))
    for idx, cluster_data in enumerate(cluster_data_list):
        cluster_indices = cluster_data['cluster_indices']
        fits = cluster_data['fits']
        window = cluster_data['window']
        cluster_peak_matches = cluster_data['peak_matches']

        # For each cluster, plot one subplot
        ax = plt.subplot(nrows, ncols, idx + 1)

        # Determine plotting window for the cluster
        min_peak_idx = min(cluster_indices)
        max_peak_idx = max(cluster_indices)

        # Convert indices to wavelengths for the plotting window
        min_peak_wavelength = wavelength[min_peak_idx]
        max_peak_wavelength = wavelength[max_peak_idx]

        # Define the plotting window based on wavelengths
        left = max(min_peak_wavelength - window * (wavelength[1] - wavelength[0]), wavelength[0])
        right = min(max_peak_wavelength + window * (wavelength[1] - wavelength[0]), wavelength[-1])

        # Select data within the plotting window
        mask = (wavelength >= left) & (wavelength <= right)
        x_data = wavelength[mask]
        y_data = intensity[mask]

        # Plot the data
        ax.plot(x_data, y_data, 'k-', label='Data')

        # Plot the Gaussian fits
        for fit in fits:
            if fit['success']:
                x_fit = fit['x_fit']
                y_fit = fit['y_fit']
                ax.plot(x_fit, y_fit, 'r--', label='Gaussian Fit' if 'Gaussian Fit' not in ax.get_legend_handles_labels()[1] else "")
            else:
                pass  # Skip failed fits

        # Title can be based on matched spectral lines
        matches_in_cluster = []
        for matches in cluster_peak_matches:
            if matches:
                best_match = min(matches, key=lambda m: abs(m['Delta']))
                line_name = best_match['Line']
                matches_in_cluster.append(line_name)
        if matches_in_cluster:
            # Remove duplicates and join line names
            matches_in_cluster = list(dict.fromkeys(matches_in_cluster))
            cluster_title = f"Cluster {idx+1}: {', '.join(matches_in_cluster)}"
        else:
            cluster_title = f"Cluster {idx+1}"

        ax.set_title(cluster_title)
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel('Intensity')
        ax.legend()
        ax.grid(True)


    if title:
        plt.suptitle(title, fontsize=16)
        
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
                        help='Maximum distance between peaks to be considered in the same cluster (default: 50)')
    parser.add_argument('--save_format', type=str, choices=['png', 'pdf'], default='png',
                        help='Format to save plots (default: png)')
    parser.add_argument('--target_name', type=str, default='M81',
                        help='Name of the target object (optional, if not in header)')
    parser.add_argument('--obs_date', type=str, default="2024-12-04T22:15:30",
                        help='Hardcoded observation date in ISO format (e.g., "2024-12-04T22:15:30"). Overrides header date if provided.')
    args = parser.parse_args()
    
    # Extract the obs_date argument
    obs_date = args.obs_date
    input_dir = args.input_dir
    output_dir = args.output_dir
    calibration_file = args.calibration_file
    height_sigma = args.height_sigma
    distance = args.distance
    prominence = args.prominence
    cluster_distance = args.cluster_distance
    save_format = args.save_format
    target_name = args.target_name

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
            cluster_data_list = []
            peak_matches = []

            for cluster in clusters:
                # For clusters, use the average width to define the fitting window
                cluster_widths = [widths[np.where(peaks == idx)[0][0]] for idx in cluster]
                avg_width = np.mean(cluster_widths)
                window = int(np.ceil(avg_width * 2.5))

                # Fit the cluster with multiple Gaussians
                fits = fit_gaussian_plus_bg(wavelength, data, cluster, window)

                # Collect peak matches for the cluster
                cluster_peak_matches = []
                for fit in fits:
                    if fit['success']:
                        peak_wavelength = fit['mean']
                        matches = match_spectral_lines(peak_wavelength, SPECTRAL_LINES, tolerance=7.0)
                        fit['spectral_lines'] = matches  # Add matches to fit dictionary
                        cluster_peak_matches.append(matches)
                    else:
                        cluster_peak_matches.append([])

                # Store cluster data
                cluster_data_list.append({
                    'cluster_indices': cluster,
                    'fits': fits,
                    'window': window,
                    'peak_matches': cluster_peak_matches
                })

                # Collect all peak indices and fits for further processing (if needed)
                for fit in fits:
                    peak_matches.append(fit['spectral_lines'] if 'spectral_lines' in fit else [])

            # Flatten all Gaussian fits and peak matches
            all_gaussian_fits = []
            all_peak_indices = []
            all_fitting_windows = []
            peak_matches_flat = []

            for cluster_data in cluster_data_list:
                cluster_indices = cluster_data['cluster_indices']
                fits = cluster_data['fits']
                cluster_peak_matches = cluster_data['peak_matches']

                all_gaussian_fits.extend(fits)
                all_peak_indices.extend(cluster_indices)
                all_fitting_windows.extend([cluster_data['window']] * len(cluster_indices))
                peak_matches_flat.extend(cluster_peak_matches)

            # Compute barycentric correction
            vcorr = compute_barycentric_correction(header, target_name=target_name, default_obs_date=obs_date)

            # Identify Halpha flux for flux ratios (if needed)
            flux_Halpha = None
            flux_Halpha_err = None
            for fit, matches in zip(all_gaussian_fits, peak_matches_flat):
                if not fit['success']:
                    continue
                for match in matches:
                    if match['Line'] == 'Hα':
                        flux_Halpha = fit['flux']
                        flux_Halpha_err = fit['flux_err']
                        break
                if flux_Halpha is not None:
                    break  # Exit once Hα is found

            # Plotting
            base_name = os.path.splitext(os.path.basename(fits_file))[0]
            plot_filename = os.path.join(output_dir, f"{base_name}_peaks.{save_format}")
            title = f"Spectrum with Detected Peaks: {base_name}"
            plot_spectrum(wavelength, data, all_peak_indices, all_gaussian_fits, plot_filename, title=title, peak_matches=peak_matches_flat)
            print(f"Plot saved to {plot_filename}")

            # Plot zoomed-in peaks for clusters
            zoomed_plot_filename = os.path.join(output_dir, f"{base_name}_zoomed_peaks.{save_format}")
            title_zoomed = f"Zoomed-in Fits for: {base_name}"
            plot_zoomed_peaks(wavelength, data, cluster_data_list, zoomed_plot_filename, title=title_zoomed)
            print(f"Zoomed-in plot saved to {zoomed_plot_filename}")

            # Append to summary
            peak_details = []
            for cluster_data in cluster_data_list:
                cluster_indices = cluster_data['cluster_indices']
                fits = cluster_data['fits']
                cluster_peak_matches = cluster_data['peak_matches']

                for idx_in_cluster, (idx, fit) in enumerate(zip(cluster_indices, fits)):
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
                        'window': cluster_data['window'],  # All peaks in the cluster have the same window
                        'fit_success': fit['success'],
                        'spectral_lines': cluster_peak_matches[idx_in_cluster]  # Add matched spectral lines
                    }
                    peak_details.append(peak_info)

            # Calculate electron densities and flux ratios
            electron_density_sii, electron_density_sii_err = calculate_electron_density_sii(all_gaussian_fits, peak_matches_flat)
            flux_ratio_oi, flux_ratio_oi_err = compute_flux_ratio_oi(all_gaussian_fits, peak_matches_flat)
            flux_ratio_nii, flux_ratio_nii_err = compute_flux_ratio_nii(all_gaussian_fits, peak_matches_flat)

            summary.append({
                'file': os.path.basename(fits_file),
                'num_peaks': num_peaks,
                'peaks': peak_details,
                'electron_density_sii': electron_density_sii,
                'electron_density_sii_err': electron_density_sii_err,
                'flux_ratio_oi': flux_ratio_oi,
                'flux_ratio_oi_err': flux_ratio_oi_err,
                'flux_ratio_nii': flux_ratio_nii,
                'flux_ratio_nii_err': flux_ratio_nii_err,
                'vcorr': vcorr
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
                'Electron Density [SII] (cm^-3)',
                'Electron Density Error [SII]',
                'Flux Ratio [O I]6300/[O I]6363',
                'Flux Ratio Error [O I]6300/[O I]6363',
                'Flux Ratio [NII]6548/[NII]6583',
                'Flux Ratio Error [NII]6548/[NII]6583',
                'Barycentric Correction (km/s)',
                'Peak Index',
                'Rest Wavelength (λ₀) (Å)',
                'Observed Wavelength (λ_obs) (Å)',
                'Observed Wavelength Error (Å)',
                'Observed Velocity (v_obs) (km/s)',
                'Velocity Error (δV_obs) (km/s)',
                'Corrected Velocity (v_corr) (km/s)',
                'FWHM (Å)',
                'FWHM Error (Å)',
                'Flux',
                'Flux Error',
                'Spectral Line',
                'Fit Success'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for entry in summary:
                file_name = entry['file']
                num_peaks = entry['num_peaks']
                electron_density_sii = entry['electron_density_sii']
                electron_density_sii_err = entry['electron_density_sii_err']
                flux_ratio_oi = entry['flux_ratio_oi']
                flux_ratio_oi_err = entry['flux_ratio_oi_err']
                flux_ratio_nii = entry['flux_ratio_nii']
                flux_ratio_nii_err = entry['flux_ratio_nii_err']
                vcorr = entry['vcorr']
                for peak in entry['peaks']:
                    row = {
                        'File': file_name,
                        'Number of Peaks': num_peaks,
                        'Electron Density [SII] (cm^-3)': f"{electron_density_sii:.2e}" if not np.isnan(electron_density_sii) else "N/A",
                        'Electron Density Error [SII]': f"{electron_density_sii_err:.2e}" if not np.isnan(electron_density_sii_err) else "N/A",
                        'Flux Ratio [O I]6300/[O I]6363': f"{flux_ratio_oi:.4f}" if not np.isnan(flux_ratio_oi) else "N/A",
                        'Flux Ratio Error [O I]6300/[O I]6363': f"{flux_ratio_oi_err:.4f}" if not np.isnan(flux_ratio_oi_err) else "N/A",
                        'Flux Ratio [NII]6548/[NII]6583': f"{flux_ratio_nii:.4f}" if not np.isnan(flux_ratio_nii) else "N/A",
                        'Flux Ratio Error [NII]6548/[NII]6583': f"{flux_ratio_nii_err:.4f}" if not np.isnan(flux_ratio_nii_err) else "N/A",
                        'Barycentric Correction (km/s)': f"{vcorr:.2f}",
                        'Peak Index': peak['peak_index'],
                        'Rest Wavelength (λ₀) (Å)': "",
                        'Observed Wavelength (λ_obs) (Å)': f"{peak['mean']:.2f}" if not np.isnan(peak['mean']) else "N/A",
                        'Observed Wavelength Error (Å)': f"{peak['mean_err']:.2f}" if not np.isnan(peak['mean_err']) else "N/A",
                        'Observed Velocity (v_obs) (km/s)': "",
                        'Velocity Error (δV_obs) (km/s)': "",
                        'Corrected Velocity (v_corr) (km/s)': "",
                        'FWHM (Å)': "",
                        'FWHM Error (Å)': "",
                        'Flux': f"{peak['flux']:.2e}" if not np.isnan(peak['flux']) else "N/A",
                        'Flux Error': f"{peak['flux_err']:.2e}" if not np.isnan(peak['flux_err']) else "N/A",
                        'Spectral Line': "",
                        'Fit Success': "Yes" if peak['fit_success'] else "No"
                    }

                    if peak['spectral_lines']:
                        # Assuming only one best match per peak
                        best_match = min(peak['spectral_lines'], key=lambda m: abs(m['Delta']))
                        line_name = best_match['Line']
                        line_wavelength = best_match['Wavelength']
                        delta = best_match['Delta']
                        z, z_err = compute_redshift(peak['mean'], line_wavelength, observed_wavelength_err=peak['mean_err'])
                        v_obs, v_err = compute_velocity(z, z_err=z_err)
                        v_corr_corrected = v_obs + vcorr

                        # Compute FWHM
                        fwhm, fwhm_err = compute_fwhm(peak['sigma'], peak['sigma_err'])

                        # Compute Flux Ratios for [O I] and [NII] are already computed outside

                        row['Spectral Line'] = line_name
                        row['Rest Wavelength (λ₀) (Å)'] = f"{line_wavelength:.2f}"
                        row['Observed Velocity (v_obs) (km/s)'] = f"{v_obs:.2f}"
                        row['Velocity Error (δV_obs) (km/s)'] = f"{v_err:.2f}" if v_err is not None else "N/A"
                        row['Corrected Velocity (v_corr) (km/s)'] = f"{v_corr_corrected:.2f}"
                        row['FWHM (Å)'] = f"{fwhm:.2f}"
                        row['FWHM Error (Å)'] = f"{fwhm_err:.2f}" if fwhm_err is not None else "N/A"

                    else:
                        row['Spectral Line'] = "None"

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