#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spectral Data Reduction and Calibration Script with Enhanced GUI
Author: Bibal Sobeaux Pierre Gabriel 
Date: 2024-09-25

This script performs spectral data reduction, including bias subtraction,
flat-field correction, wavelength calibration using ThAr lamp spectra, and saves
the reduced spectra into a new directory. It provides an enhanced GUI for monitoring
the process and inspecting data, allows interactive peak inspection and wavelength
assignment, and generates high-resolution debugging images.

Additionally, it allows stacking multiple reduced spectra using a selected stacking method.

Usage:
    python spectral_reduction.py

Dependencies:
    - numpy
    - matplotlib
    - astropy
    - scipy
    - Tkinter (standard with Python)

Ensure all required data files are in the specified directories.

Directory Structure:
    - spectral_reduction.py   # This script
    - Raw/                    # Directory containing raw observational spectra
    - bias/                   # Directory containing bias frames
    - Tung/                   # Directory containing flat frames (e.g., Tungsten lamp)
    - ThAr/                   # Directory containing calibration lamp spectra
    - Reduced/                # Directory where reduced spectra will be saved
    - linelists/              # Directory containing line lists for calibration
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import os
from numpy.polynomial import chebyshev
import re
import warnings
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.simpledialog import askstring
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

# Ensure Tkinter backend is used for matplotlib
import matplotlib
matplotlib.use('TkAgg')

# Apply a matplotlib style for all plots
plt.rcParams['figure.dpi'] = 150  # Increase the resolution of plots

# ======================================================================
# Configuration
# ======================================================================

# Define base directory (the directory where the script is located)
BASE_DIR = os.getcwd()

# Define data directories relative to BASE_DIR
RAW_DIR = os.path.join(BASE_DIR, 'Raw')
BIAS_DIR = os.path.join(BASE_DIR, 'bias')
FLAT_DIR = os.path.join(BASE_DIR, 'Tung')
THAR_DIR = os.path.join(BASE_DIR, 'ThAr')
REDUCED_DIR = os.path.join(BASE_DIR, 'Reduced')
LINELIST_DIR = os.path.join(BASE_DIR, 'linelists')

# Ensure the Reduced directory exists
os.makedirs(REDUCED_DIR, exist_ok=True)

# Define file patterns
BIAS_PATTERN = os.path.join(BIAS_DIR, '*.fits')
FLAT_PATTERN = os.path.join(FLAT_DIR, '*.fits')
RAW_PATTERN = os.path.join(RAW_DIR, '*.fits')
THAR_PATTERN = os.path.join(THAR_DIR, '*.fits')

# Output filenames
MASTER_BIAS_FILE = os.path.join(BIAS_DIR, 'master_bias.fits')
MASTER_FLAT_FILE = os.path.join(FLAT_DIR, 'master_flat.fits')
CALIB_FILE = os.path.join(REDUCED_DIR, 'wavelength_calibration.dat')

# ======================================================================
# Helper Functions
# ======================================================================

def read_spectrum(filename, get_header=False):
    """Reads a 1D spectrum from a FITS file."""
    with fits.open(filename) as hdul:
        data = hdul[0].data.astype(float)
        # If data is 2D, we squeeze it to make it 1D
        if data.ndim > 1:
            data = data.squeeze()
        header = hdul[0].header if get_header else None
    xaxis = np.arange(len(data))
    return (xaxis, data, header) if get_header else (xaxis, data)

def write_spectrum(filename, data, header=None):
    """Writes a 1D spectrum to a FITS file."""
    hdu = fits.PrimaryHDU(data)
    if header:
        hdu.header = header
    hdu.writeto(filename, overwrite=True)

def median_combine(files):
    """Median combines a list of spectra."""
    spectra = []
    for file in files:
        _, data = read_spectrum(file)
        spectra.append(data)
    return np.median(spectra, axis=0)

def normalize_spectrum(data):
    """Normalizes a spectrum by its median value."""
    return data / np.median(data)

def fit_chebyshev(x, y, degree=3, weights=None):
    """Fits a Chebyshev polynomial to the data."""
    coeffs = chebyshev.chebfit(x, y, degree, w=weights)
    y_fit = chebyshev.chebval(x, coeffs)
    return coeffs, y_fit

def sort_files_numerically(files):
    """Sorts file names numerically based on embedded digits."""
    files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    return files

def remove_bad_flats(files, bad_indices):
    """Removes bad flats based on provided indices."""
    return [f for i, f in enumerate(files) if i not in bad_indices]

def generate_first_guess(peaks):
    npeaks = len(peaks)
    initial_cen = peaks
    initial_amp = np.full(npeaks, 20000.)
    initial_wid = np.full(npeaks, 1.)
    first_guess = np.full((npeaks, 3), 0.)
    for i in range(npeaks):
        first_guess[i,0] = initial_amp[i]
        first_guess[i,1] = initial_cen[i]
        first_guess[i,2] = initial_wid[i]
    first_guess = np.reshape(first_guess, 3*npeaks)
    return first_guess

def gaussian(x, *params):
    """Sum of Gaussians function."""
    y = np.zeros_like(x)
    num_gaussians = len(params) // 3
    for i in range(num_gaussians):
        amplitude = params[3*i]
        mean = params[3*i + 1]
        stddev = params[3*i + 2]
        y += amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
    return y

# ======================================================================
# Main Reduction Steps
# ======================================================================

class SpectralReduction:
    def __init__(self, gui=None):
        self.master_bias = None
        self.master_flat = None
        self.wavelengths = None
        self.gui = gui
        self.bias_data_list = []   # Stores (x, data, label) tuples for bias frames
        self.flat_data_list = []   # Stores (x, data, label) tuples for flat frames
        self.thar_data = None      # Stores the ThAr lamp data
        self.calibration_coeffs = None
        self.combined_flat = None
        self.y_fit = None
        self.allcens = None        # Store Gaussian centroid positions
        self.atlas_wavelengths = None
        self.calibration_residuals = None
        self.calibration_peaks = None
        self.stacked_spectrum = None  # Stores the final stacked spectrum

    def create_master_bias(self):
        """Creates a master bias frame."""
        self.gui.update_status("Creating master bias...")
        bias_files = glob.glob(BIAS_PATTERN)
        bias_files = sort_files_numerically(bias_files)
        if not bias_files:
            self.gui.update_status("No bias files found.")
            return None
        spectra = []
        self.bias_data_list = []
        for file in bias_files:
            x, data = read_spectrum(file)
            spectra.append(data)
            self.bias_data_list.append((x, data, os.path.basename(file)))
        self.master_bias = np.median(spectra, axis=0)
        write_spectrum(MASTER_BIAS_FILE, self.master_bias)
        self.gui.update_status(f"Master Bias saved to {MASTER_BIAS_FILE}")
        return self.master_bias

    def create_master_flat(self):
        """Creates a master flat frame."""
        self.gui.update_status("Creating master flat...")
        flat_files = glob.glob(FLAT_PATTERN)
        flat_files = sort_files_numerically(flat_files)

        if not flat_files:
            self.gui.update_status("No flat files found.")
            return None

        # Optionally remove bad flats based on visual inspection
        bad_flat_indices = []  # Adjust based on your data
        flat_files = remove_bad_flats(flat_files, bad_flat_indices)

        normalized_flats = []
        self.flat_data_list = []
        for file in flat_files:
            x, data = read_spectrum(file)
            data -= self.master_bias
            data_norm = normalize_spectrum(data)
            normalized_flats.append(data_norm)
            self.flat_data_list.append((x, data_norm, os.path.basename(file)))

        # Median combine normalized flats
        self.combined_flat = np.median(normalized_flats, axis=0)

        # Fit and remove the slope using Chebyshev polynomial
        x = np.arange(len(self.combined_flat))
        coeffs, self.y_fit = fit_chebyshev(x, self.combined_flat)
        self.master_flat = self.combined_flat / self.y_fit
        write_spectrum(MASTER_FLAT_FILE, self.master_flat)
        self.gui.update_status(f"Master Flat saved to {MASTER_FLAT_FILE}")
        return self.master_flat

    def calibrate_wavelength(self):
        """Performs wavelength calibration using ThAr lamp spectra."""
        self.gui.update_status("Performing wavelength calibration...")

        # Load ThAr lamp spectrum file
        thar_files = glob.glob(THAR_PATTERN)
        thar_files = sort_files_numerically(thar_files)

        if not thar_files:
            self.gui.update_status("No ThAr lamp files found.")
            return None

        # Use the first ThAr file for calibration
        thar_file = thar_files[0]
        xaxis, data = read_spectrum(thar_file)

        # Save the ThAr data
        self.thar_data = data

        # Detect peaks
        peaks, _ = find_peaks(data, height=12000)
        self.calibration_peaks = peaks

        # Fit Gaussians to the peaks for better centroid determination
        first_guess = generate_first_guess(peaks)
        try:
            params, covariance = curve_fit(gaussian, xaxis, data, p0=first_guess)
        except RuntimeError:
            self.gui.update_status("Gaussian fitting failed.")
            return None

        # Reshape the fitted parameters
        num_gaussians = len(params) // 3
        print('Number of peaks:', num_gaussians)
        params = np.array(params).reshape((num_gaussians, 3))
        allamps = params[:, 0]  # Gaussian amplitudes
        allcens = params[:, 1]  # Gaussian centroids (pixel positions of the peaks)
        allwids = params[:, 2]  # Gaussian widths

        # Store the centroid positions for GUI access
        self.allcens = allcens

        # Define prefilled pixel_lambda based on specific allcens indices
        prefilled_indices = [5, 13, 14, 21, 22, 26]
        known_wavelengths = [6182.62, 6457.28, 6531.34, 6677.28, 6752.83, 6911.23]
        pixel_lambda_prefilled = []
        for idx, wl in zip(prefilled_indices, known_wavelengths):
            if idx < len(allcens):
                pixel = allcens[idx]
                pixel_lambda_prefilled.append([pixel, wl])
            else:
                print(f"Warning: Peak index {idx} is out of range for detected peaks.")

        # Launch Peak Assignment GUI with prefilled data
        self.gui.update_status("Launching peak assignment window for user input...")
        pixel_wavelength_pairs = self.gui.launch_peak_assignment(
            xaxis, data, allcens, prefilled_pairs=pixel_lambda_prefilled)

        if len(pixel_wavelength_pairs) < 2:
            self.gui.update_status("Insufficient peak assignments for calibration.")
            return None

        # Convert to numpy array for processing
        pixel_wavelength_pairs = np.array(pixel_wavelength_pairs)

        # Fit a Chebyshev polynomial to the user-provided pixel-wavelength pairs
        degree = 2  # You can adjust the degree based on requirements
        coeffs_initial, y_fit_initial = chebyshev.chebfit(
            pixel_wavelength_pairs[:, 0], pixel_wavelength_pairs[:, 1], degree, full=True)[:2]

        # Evaluate the Chebyshev polynomial to predict wavelengths across the x-axis
        y_fit = chebyshev.chebval(xaxis, coeffs_initial)

        # Calculate residuals
        predicted_wl_initial = chebyshev.chebval(pixel_wavelength_pairs[:, 0], coeffs_initial)
        atlas_wl_initial = pixel_wavelength_pairs[:, 1]
        residuals_initial = predicted_wl_initial - atlas_wl_initial

        # Display residual statistics
        print('==============================')
        print('Initial Wavelength Calibration Residuals')
        print('NLINES=', len(pixel_wavelength_pairs))
        print('RESIDUALS AVG', np.average(residuals_initial), 'Angstrom')
        print('RESIDUALS RMS', np.std(residuals_initial), 'Angstrom')
        print('RESIDUALS RMS (in km/s)', np.std(residuals_initial) /
              np.average(atlas_wl_initial) * 3e5, 'km/s')
        print('==============================')

        # Load the NIST line list
        line_list_file = os.path.join(LINELIST_DIR, 'ThI.csv')
        try:
            line_data = np.loadtxt(line_list_file, delimiter=',', skiprows=1)
        except IOError:
            self.gui.update_status("Line list file not found.")
            return None

        NIST_wls = line_data[:, 0]
        NIST_rels = line_data[:, 1]
        NIST_rels /= np.max(NIST_rels)

        # Select the brightest lines from the NIST linelist (those with intensity >= 0.2)
        ind = np.where(NIST_rels >= 0.2)[0]
        NIST_wls = NIST_wls[ind]
        NIST_rels = NIST_rels[ind]
        print(f"Number of NIST lines kept: {len(ind)}")

        # Prepare arrays for matching the peaks with the NIST linelist
        match = np.zeros(num_gaussians)
        predicted_wl = np.zeros(num_gaussians)
        atlas_wl = np.zeros(num_gaussians)
        residuals = np.zeros(num_gaussians)
        maxdelta = 0.2  # Maximum allowable delta for matching (in Angstrom)

        # Predict wavelengths for all detected peaks using the initial calibration
        predicted_wl = chebyshev.chebval(allcens, coeffs_initial)

        # Match the detected peaks with the NIST linelist
        for i in range(num_gaussians):
            imin = np.argmin(np.abs(predicted_wl[i] - NIST_wls))
            residuals[i] = predicted_wl[i] - NIST_wls[imin]
            atlas_wl[i] = NIST_wls[imin]

            if np.abs(residuals[i]) < maxdelta:  # Keep match if within allowable delta
                match[i] = imin + 1  # Avoid zero index

        # Select only matched lines
        ind_matched = np.where(match != 0)[0]
        print(f"Number of matched lines: {len(ind_matched)}")

        if len(ind_matched) < 2:
            self.gui.update_status("Insufficient matched lines for calibration.")
            return None

        # Prepare new pixel-to-wavelength matches for second iteration
        new_pixel_lambda = np.zeros((len(ind_matched), 2))
        new_pixel_lambda[:, 0] = allcens[ind_matched]
        new_pixel_lambda[:, 1] = atlas_wl[ind_matched]

        # Combine user-assigned and automatically matched pixel-wavelength pairs
        combined_pixel_lambda = np.vstack((pixel_wavelength_pairs, new_pixel_lambda))

        # Perform second iteration of Chebyshev polynomial fitting
        degree = 2
        coeffs_final, y_fit_final = chebyshev.chebfit(
            combined_pixel_lambda[:, 0], combined_pixel_lambda[:, 1], degree, full=True)[:2]
        print(f"Chebyshev coefficients after second iteration: {coeffs_final}")

        # Final residuals for the second iteration
        predicted_final = chebyshev.chebval(combined_pixel_lambda[:, 0], coeffs_final)
        residuals_final = predicted_final - combined_pixel_lambda[:, 1]

        # Save the residuals and atlas wavelengths for plotting
        self.calibration_residuals = residuals_final
        self.atlas_wavelengths = combined_pixel_lambda[:, 1]

        # Display residual statistics
        print('==============================')
        print('Final Wavelength Calibration Residuals')
        print(f"NLINES = {len(combined_pixel_lambda)}")
        print(f"DEGREE = {degree}")
        print(f"RESIDUALS AVG = {np.average(residuals_final)}")
        print(f"RESIDUALS RMS = {np.std(residuals_final)}")
        print(f"RESIDUALS RMS (in km/s) = {np.std(residuals_final) / np.average(combined_pixel_lambda[:, 1]) * 3e5} km/s")
        print('==============================')

        # Save final wavelength calibration
        self.wavelengths = chebyshev.chebval(xaxis, coeffs_final)
        np.savetxt(CALIB_FILE, self.wavelengths)
        self.gui.update_status(f"Wavelength calibration saved to {CALIB_FILE}")

        # Store the calibration coefficients for later use
        self.calibration_coeffs = coeffs_final
        self.atlas_wavelengths = combined_pixel_lambda[:, 1]

        return self.wavelengths

    def reduce_observations(self):
        """Reduces observational spectra and saves wavelength calibration into the FITS headers."""
        self.gui.update_status("Reducing observations...")
        raw_files = glob.glob(RAW_PATTERN)
        raw_files = sort_files_numerically(raw_files)

        if not raw_files:
            self.gui.update_status("No raw observation files found.")
            return

        for raw_file in raw_files:
            x, data, header = read_spectrum(raw_file, get_header=True)
            data -= self.master_bias
            data /= self.master_flat

            # Apply wavelength calibration if available
            if self.wavelengths is not None:
                # Add wavelength calibration to the FITS header
                header['CRVAL1'] = self.wavelengths[0]  # Starting wavelength (first pixel)
                header['CDELT1'] = (self.wavelengths[-1] - self.wavelengths[0]) / (
                    len(self.wavelengths) - 1)  # Wavelength increment
                header['CTYPE1'] = 'Wavelength'  # Specify the type of axis

                # Add the coefficients of the polynomial fit used for the calibration
                if self.calibration_coeffs is not None:
                    for i, coeff in enumerate(self.calibration_coeffs):
                        header[f'CAL_COEF{i}'] = coeff  # Store each calibration coefficient in the header

            # Save the reduced spectrum into the "Reduced" directory
            base_filename = os.path.basename(raw_file).replace('.fits', '_reduced.fits')
            reduced_file = os.path.join(REDUCED_DIR, base_filename)
            write_spectrum(reduced_file, data, header)

            self.gui.update_status(f"Reduced spectrum saved to {reduced_file}")

    def stack_reduced_spectra(self, files_to_stack, stacking_method):
        """
        Stacks selected reduced spectra using the specified stacking method.
        
        The stacked spectrum is saved in the 'Reduced/stacked' directory.
        The filename is based on the common prefix of the selected files.
        If no common prefix exists, a warning is displayed and a default name is used.
        
        Parameters:
            files_to_stack (list): List of file paths to stack.
            stacking_method (str): Method to stack spectra ('Median', 'Average', 'Weighted Average').
        
        Returns:
            np.ndarray: The stacked spectrum data, or None if stacking fails.
        """
        self.gui.update_status("Stacking reduced spectra...")
        
        if not files_to_stack:
            self.gui.update_status("No reduced spectra selected for stacking.")
            return None

        # Extract base filenames from the selected files
        base_names = [os.path.basename(f) for f in files_to_stack]
        
        # Find the common prefix among the filenames
        common_prefix = os.path.commonprefix(base_names)
        
        # Clean the common prefix by removing trailing underscores, hyphens, or dots
        common_prefix = common_prefix.rstrip('_-.')

        # Determine if the common prefix is significant (e.g., at least 3 characters)
        if len(common_prefix) >= 3:
            stacked_filename = f"{common_prefix}_stacked.fits"
        else:
            # Warn the user that the selected files may be different
            messagebox.showwarning(
                "No Common Prefix Detected",
                "The selected files do not share a common prefix. "
                "You may be stacking different spectra, which could lead to inconsistent results."
            )
            stacked_filename = "stacked_spectrum.fits"

        # Define the stacked directory path
        stacked_dir = os.path.join(REDUCED_DIR, 'stacked')
        
        # Create the 'stacked' directory if it doesn't exist
        os.makedirs(stacked_dir, exist_ok=True)
        
        # Full path for the stacked spectrum file
        stacked_file = os.path.join(stacked_dir, stacked_filename)
        
        # Initialize a list to hold the spectrum data
        stacked_spectra = []
        
        # Read and collect data from each selected file
        for file in files_to_stack:
            try:
                _, data = read_spectrum(file)
                stacked_spectra.append(data)
            except Exception as e:
                messagebox.showerror(
                    "Error Reading File",
                    f"An error occurred while reading {os.path.basename(file)}:\n{e}"
                )
                self.gui.update_status(f"Failed to read {os.path.basename(file)}. Skipping.")
                continue

        if not stacked_spectra:
            self.gui.update_status("No valid spectra were loaded for stacking.")
            return None

        # Convert the list of spectra to a NumPy array for stacking
        stacked_spectra = np.array(stacked_spectra)

        # Perform stacking based on the selected method
        if stacking_method == 'Median':
            self.stacked_spectrum = np.median(stacked_spectra, axis=0)
        elif stacking_method == 'Average':
            self.stacked_spectrum = np.mean(stacked_spectra, axis=0)
        elif stacking_method == 'Weighted Average':
            # Implement actual weighting logic as needed
            # Currently using equal weights for simplicity
            weights = np.ones(stacked_spectra.shape[0])
            self.stacked_spectrum = np.average(stacked_spectra, axis=0, weights=weights)
        else:
            self.gui.update_status("Invalid stacking method selected.")
            messagebox.showerror(
                "Invalid Stacking Method",
                f"The stacking method '{stacking_method}' is not recognized. "
                "Please choose 'Median', 'Average', or 'Weighted Average'."
            )
            return None

        # Save the stacked spectrum to the designated file
        try:
            write_spectrum(stacked_file, self.stacked_spectrum)
            self.gui.update_status(f"Stacked spectrum saved to {stacked_file}")
        except Exception as e:
            messagebox.showerror(
                "Error Saving Stacked Spectrum",
                f"An error occurred while saving the stacked spectrum:\n{e}"
            )
            self.gui.update_status("Failed to save the stacked spectrum.")
            return None

        return self.stacked_spectrum

# ======================================================================
# Peak Assignment GUI Window
# ======================================================================

class PeakAssignmentWindow(tk.Toplevel):
    def __init__(self, master, xaxis, data, allcens, prefilled_pairs=None):
        super().__init__(master)
        self.title("Peak Assignment")

        # Get screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        # Set window size to 80% of screen dimensions
        window_width = int(screen_width * 0.7)
        window_height = int(screen_height * 0.9)
        self.geometry(f"{window_width}x{window_height}")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.xaxis = xaxis
        self.data = data
        self.allcens = allcens

        # Initialize lists to store pixel-wavelength pairs
        self.pixel_wavelength_pairs = [] if prefilled_pairs is None else prefilled_pairs.copy()
        self.prefilled_pairs = prefilled_pairs.copy() if prefilled_pairs else []

        # Create matplotlib figure
        self.fig = plt.Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(self.xaxis, self.data, label='ThAr Spectrum')
        self.ax.plot(self.allcens, self.data[self.allcens.astype(int)], 'ro', label='Detected Peaks')
        self.ax.set_title('ThAr Spectrum with Detected Peaks')
        self.ax.set_xlabel('Pixel')
        self.ax.set_ylabel('Intensity')
        self.ax.legend()

        # Embed the plot in Tkinter
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Instruction Label
        instruction = ttk.Label(self, text="Select a peak by clicking on it, then enter the known wavelength.")
        instruction.pack(pady=5)

        # Bind click event
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Listbox to show assigned peaks
        self.assigned_listbox = tk.Listbox(self, width=50)
        self.assigned_listbox.pack(pady=10)

        # Prepopulate listbox with prefilled pairs
        if prefilled_pairs:
            for pair in prefilled_pairs:
                pixel = pair[0]
                wl = pair[1]
                self.assigned_listbox.insert(tk.END, f"Pixel: {int(pixel)}, Wavelength: {wl} Å")
                # Mark prefilled peaks with blue 'x'
                self.ax.plot(pixel, self.data[int(pixel)], 'bx', markersize=12, label='Prefilled Peak')
            self.canvas.draw()

        # Buttons Frame
        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(side=tk.RIGHT, fill='x')

        # Button to remove selected peak
        self.remove_button = ttk.Button(buttons_frame, text="Remove Selected Peak", command=self.remove_selected_peak, width=20)
        self.remove_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Button to finish assignment
        self.finish_button = ttk.Button(buttons_frame, text="Done", command=self.finish_assignment, width=10)
        self.finish_button.pack(side=tk.TOP, padx=5, pady=5)

        # Adding padding inside the buttons to make them appear bigger
        self.remove_button.config(padding=(10, 10))
        self.finish_button.config(padding=(10, 10))

    def on_click(self, event):
        """Handle click events on the plot to select peaks."""
        if event.inaxes != self.ax:
            return
        x_click = event.xdata
        # Find the nearest peak to the click
        idx = (np.abs(self.allcens - x_click)).argmin()
        selected_pixel = self.allcens[idx]
        if any(np.isclose(selected_pixel, pair[0], atol=1e-2) for pair in self.pixel_wavelength_pairs):
            messagebox.showinfo("Peak Already Assigned", f"Peak at pixel {int(selected_pixel)} is already assigned.")
            return
        # Prompt user to enter the known wavelength
        wavelength = askstring("Input Wavelength", f"Enter known wavelength for peak at pixel {int(selected_pixel)}:")
        if wavelength is not None:
            try:
                wavelength = float(wavelength)
                self.pixel_wavelength_pairs.append([selected_pixel, wavelength])
                # Update the listbox
                self.assigned_listbox.insert(tk.END, f"Pixel: {int(selected_pixel)}, Wavelength: {wavelength} Å")
                # Mark the assigned peak with green 'x'
                self.ax.plot(selected_pixel, self.data[int(selected_pixel)], 'gx', markersize=12, label='Assigned Peak')
                self.canvas.draw()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid numerical wavelength.")

    def remove_selected_peak(self):
        """Removes the selected peak from the listbox and internal list."""
        selected_indices = self.assigned_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("No Selection", "Please select a peak to remove.")
            return
        for index in reversed(selected_indices):
            # Remove from listbox
            peak_info = self.assigned_listbox.get(index)
            self.assigned_listbox.delete(index)
            # Remove from internal list
            try:
                pixel = float(re.findall(r'Pixel: (\d+)', peak_info)[0])
                wavelength = float(re.findall(r'Wavelength: ([\d.]+)', peak_info)[0])
                self.pixel_wavelength_pairs = [pair for pair in self.pixel_wavelength_pairs if not (np.isclose(pair[0], pixel, atol=1e-2) and np.isclose(pair[1], wavelength, atol=1e-2))]
            except (IndexError, ValueError):
                continue
        # Clear and replot all markers
        self.ax.lines = [line for line in self.ax.lines if line.get_label() not in ['Assigned Peak', 'Prefilled Peak']]
        # Replot all assigned peaks
        for pair in self.pixel_wavelength_pairs:
            pixel, wl = pair
            self.ax.plot(pixel, self.data[int(pixel)], 'gx', markersize=12, label='Assigned Peak')
        # Replot prefilled peaks
        for pair in self.prefilled_pairs:
            pixel, wl = pair
            self.ax.plot(pixel, self.data[int(pixel)], 'bx', markersize=12, label='Prefilled Peak')
        self.canvas.draw()

    def finish_assignment(self):
        """Close the window and indicate completion."""
        if not self.pixel_wavelength_pairs and not self.prefilled_pairs:
            if not messagebox.askyesno("No Assignments", "No peaks have been assigned. Do you want to exit without assigning?"):
                return
        self.destroy()

    def on_close(self):
        """Handle window close event."""
        if not self.pixel_wavelength_pairs and not self.prefilled_pairs:
            if not messagebox.askyesno("No Assignments", "No peaks have been assigned. Do you want to exit without assigning?"):
                return
        self.destroy()

# ======================================================================
# GUI Class
# ======================================================================

class ReductionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectral Data Reduction and Calibration")

        # Set window size to 80% of screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        self.root.geometry(f"{window_width}x{window_height}")

        self.apply_styles()
        self.create_widgets()
        self.reduction = SpectralReduction(gui=self)

    def apply_styles(self):
        # Apply a theme
        style = ttk.Style()
        style.theme_use('clam')  # You can choose 'clam', 'alt', 'default', 'classic'

        # Customize styles if needed
        style.configure('TButton', font=('Helvetica', 12))
        style.configure('TLabel', font=('Helvetica', 12))
        style.configure('TFrame', padding=10)

    def create_widgets(self):
        # Create frames
        self.frame = ttk.Frame(self.root)
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights for responsiveness
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)
        self.frame.columnconfigure(2, weight=1)  # Added weight for the third column
        self.frame.rowconfigure(3, weight=1)  # Allow treeview to expand

        # Status Label
        self.status_label = ttk.Label(self.frame, text="Ready", anchor='center')
        self.status_label.grid(row=0, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))  # Updated columnspan to 3

        # Buttons
        self.start_button = ttk.Button(self.frame, text="Start Reduction", command=self.start_reduction)
        self.start_button.grid(row=1, column=0, pady=5, padx=5, sticky=tk.E)

        self.inspect_button = ttk.Button(self.frame, text="Inspect Data", command=self.inspect_data)
        self.inspect_button.grid(row=1, column=1, pady=5, padx=5, sticky=tk.W)

        # New Button: Show Gaussian Peaks
        self.peaks_button = ttk.Button(self.frame, text="Show Gaussian Peaks", command=self.show_gaussian_peaks)
        self.peaks_button.grid(row=1, column=2, pady=5, padx=5, sticky=tk.W)

        # New Button: Stack Spectra
        self.stack_button = ttk.Button(self.frame, text="Stack Spectra", command=self.stack_spectra)
        self.stack_button.grid(row=2, column=0, pady=5, padx=5, sticky=tk.E)

        # Treeview for files
        self.tree = ttk.Treeview(self.frame, columns=('Type', 'File'), show='headings', height=10)
        self.tree.heading('Type', text='Type')
        self.tree.heading('File', text='File')
        self.tree.column('Type', width=100, anchor='center')
        self.tree.column('File', width=600, anchor='w')
        self.tree.grid(row=3, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))  # Updated columnspan to 3

        # Add scrollbar to the treeview
        scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=3, column=3, sticky=(tk.N, tk.S))

    def update_status(self, message):
        self.status_label.config(text=message)
        self.status_label.update()
        # Also update the treeview with the latest file
        if "saved to" in message:
            file_type = message.split()[0]
            file_path = message.split("saved to ")[1]
            self.tree.insert('', 'end', values=(file_type, file_path))

    def start_reduction(self):
        # Disable the start button to prevent multiple clicks
        self.start_button.config(state='disabled')
        # Run the reduction in a separate thread to keep the GUI responsive
        threading.Thread(target=self.run_reduction).start()

    def run_reduction(self):
        self.update_status("Starting data reduction...")
        self.reduction.create_master_bias()
        if self.reduction.master_bias is None:
            self.update_status("Master bias creation failed.")
            self.start_button.config(state='normal')
            return
        self.reduction.create_master_flat()
        if self.reduction.master_flat is None:
            self.update_status("Master flat creation failed.")
            self.start_button.config(state='normal')
            return
        self.reduction.calibrate_wavelength()
        if self.reduction.wavelengths is None:
            self.update_status("Wavelength calibration failed.")
            self.start_button.config(state='normal')
            return
        self.reduction.reduce_observations()
        self.update_status("Data reduction completed.")
        # Generate plots
        self.display_plots()
        # Re-enable the start button
        self.start_button.config(state='normal')

    def inspect_data(self):
        # Open a file dialog to select a FITS file
        filetypes = [('FITS files', '*.fits'), ('All files', '*')]
        filename = filedialog.askopenfilename(title='Open a FITS file', initialdir=BASE_DIR, filetypes=filetypes)
        if filename:
            # Read and plot the spectrum
            x, data = read_spectrum(filename)
            # Check if wavelength calibration is available
            if os.path.exists(CALIB_FILE):
                # Load wavelength calibration
                wavelengths = np.loadtxt(CALIB_FILE)
                if len(wavelengths) != len(data):
                    messagebox.showwarning("Calibration Mismatch", "Wavelength calibration data does not match the spectrum length.")
                    x_axis = x
                    x_label = 'Pixel'
                else:
                    x_axis = wavelengths
                    x_label = 'Wavelength (Angstrom)'
            else:
                x_axis = x
                x_label = 'Pixel'

            # Create a new window for the plot
            inspect_window = tk.Toplevel(self.root)
            inspect_window.title(f"Inspecting {os.path.basename(filename)}")

            # Get screen dimensions
            screen_width = inspect_window.winfo_screenwidth()
            screen_height = inspect_window.winfo_screenheight()
            # Set window size to 80% of screen dimensions
            window_width = int(screen_width * 0.8)
            window_height = int(screen_height * 0.8)
            inspect_window.geometry(f"{window_width}x{window_height}")

            # Create a matplotlib figure
            fig = plt.Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(x_axis, data)
            ax.set_title(f"Inspecting {os.path.basename(filename)}")
            ax.set_xlabel(x_label)
            ax.set_ylabel('Intensity')

            # Embed the plot in the Tkinter window
            canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(fig, master=inspect_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
        else:
            messagebox.showinfo("No file selected", "Please select a file to inspect.")

    def show_gaussian_peaks(self):
        """Opens a new window displaying the ThAr spectrum with detected Gaussian peaks annotated."""
        if self.reduction.thar_data is None or self.reduction.allcens is None:
            messagebox.showwarning("Data Not Available", "Please run wavelength calibration first.")
            return

        # Create a new window
        peaks_window = tk.Toplevel(self.root)
        peaks_window.title("Detected Gaussian Peaks")

        # Get screen dimensions
        screen_width = peaks_window.winfo_screenwidth()
        screen_height = peaks_window.winfo_screenheight()
        # Set window size to 80% of screen dimensions
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        peaks_window.geometry(f"{window_width}x{window_height}")

        # Create a matplotlib figure
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Plot the ThAr spectrum
        xaxis = np.arange(len(self.reduction.thar_data))
        data = self.reduction.thar_data
        ax.plot(xaxis, data, label='ThAr Spectrum')

        # Plot the detected peaks
        ax.plot(self.reduction.allcens, data[self.reduction.allcens.astype(int)], 'ro', label='Detected Peaks')

        # Annotate each peak with its number
        for idx, cen in enumerate(self.reduction.allcens):
            cen_int = int(cen)
            ax.annotate(str(idx+1), (cen, data[cen_int]), textcoords="offset points", xytext=(0,10), ha='center', color='blue')

        ax.set_title('Detected Gaussian Peaks')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Intensity')
        ax.legend()

        # Embed the plot in the Tkinter window
        canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(fig, master=peaks_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def stack_spectra(self):
        """Handles the stacking process: selecting files, method, and saving the result."""
        # Open file dialog to select reduced spectra
        filetypes = [('FITS files', '*.fits'), ('All files', '*')]
        files_to_stack = filedialog.askopenfilenames(
            title='Select Reduced Spectra to Stack', initialdir=REDUCED_DIR, filetypes=filetypes)
        if not files_to_stack:
            messagebox.showinfo("No Files Selected", "Please select at least one reduced spectrum to stack.")
            return

        # Ask user to select stacking method
        stacking_method = self.select_stacking_method()
        if not stacking_method:
            return

        # Run stacking in a separate thread
        threading.Thread(target=self.run_stacking, args=(files_to_stack, stacking_method)).start()

    def select_stacking_method(self):
        """Opens a dialog for the user to select the stacking method."""
        method_window = tk.Toplevel(self.root)
        method_window.title("Select Stacking Method")

        # Get screen dimensions
        screen_width = method_window.winfo_screenwidth()
        screen_height = method_window.winfo_screenheight()
        # Set window size to 40% of screen dimensions
        window_width = int(screen_width * 0.4)
        window_height = int(screen_height * 0.3)
        method_window.geometry(f"{window_width}x{window_height}")

        method_window.grab_set()  # Make this window modal

        method_var = tk.StringVar(value='Median')

        ttk.Label(method_window, text="Choose Stacking Method:").pack(pady=10)

        methods = ['Median', 'Average', 'Weighted Average']
        for method in methods:
            ttk.Radiobutton(method_window, text=method, variable=method_var, value=method).pack(anchor=tk.W)

        def confirm():
            method_window.destroy()

        ttk.Button(method_window, text="OK", command=confirm).pack(pady=10)

        self.root.wait_window(method_window)

        return method_var.get()

    def run_stacking(self, files_to_stack, stacking_method):
        self.update_status("Starting stacking process...")
        self.reduction.stack_reduced_spectra(files_to_stack, stacking_method)
        self.update_status("Stacking completed.")
        # Display the stacked spectrum
        self.display_stacked_spectrum()

    def display_stacked_spectrum(self):
        """Displays the stacked spectrum."""
        if self.reduction.stacked_spectrum is None:
            messagebox.showwarning("Stacking Failed", "No stacked spectrum to display.")
            return

        # Create a new window for the stacked spectrum plot
        stacked_window = tk.Toplevel(self.root)
        stacked_window.title("Stacked Spectrum")

        # Get screen dimensions
        screen_width = stacked_window.winfo_screenwidth()
        screen_height = stacked_window.winfo_screenheight()
        # Set window size to 80% of screen dimensions
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        stacked_window.geometry(f"{window_width}x{window_height}")

        # Create a matplotlib figure
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Determine if wavelength calibration is available
        if os.path.exists(CALIB_FILE):
            wavelengths = np.loadtxt(CALIB_FILE)
            if len(wavelengths) == len(self.reduction.stacked_spectrum):
                x_axis = wavelengths
                x_label = 'Wavelength (Angstrom)'
            else:
                x_axis = np.arange(len(self.reduction.stacked_spectrum))
                x_label = 'Pixel'
                messagebox.showwarning("Calibration Mismatch", "Wavelength calibration data does not match the stacked spectrum length.")
        else:
            x_axis = np.arange(len(self.reduction.stacked_spectrum))
            x_label = 'Pixel'

        ax.plot(x_axis, self.reduction.stacked_spectrum, label='Stacked Spectrum')
        ax.set_title('Final Stacked Spectrum')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Intensity')
        ax.legend()

        # Embed the plot in the Tkinter window
        canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(fig, master=stacked_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def display_plots(self):
        """Displays the plots of biases, master bias, flats, master flat, and calibration steps."""
        self.update_status("Generating plots...")

        # Create a new window for plots
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Reduction and Calibration Plots")

        # Get screen dimensions
        screen_width = plot_window.winfo_screenwidth()
        screen_height = plot_window.winfo_screenheight()
        # Set window size to 80% of screen dimensions
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        plot_window.geometry(f"{window_width}x{window_height}")

        notebook = ttk.Notebook(plot_window)
        notebook.pack(fill='both', expand=True)

        # Bias Plot Tab
        bias_frame = ttk.Frame(notebook)
        notebook.add(bias_frame, text='Bias Frames')

        # Flat Plot Tab
        flat_frame = ttk.Frame(notebook)
        notebook.add(flat_frame, text='Flat Frames')

        # Calibration Plot Tab
        calib_frame = ttk.Frame(notebook)
        notebook.add(calib_frame, text='Calibration')

        # Master Bias Plot
        self.plot_biases(bias_frame)

        # Master Flat Plot
        self.plot_flats(flat_frame)

        # Calibration Plots
        self.plot_calibration(calib_frame)

        self.update_status("Plots generated.")

    def plot_biases(self, parent):
        """Plots all bias frames and the master bias."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        for x, data, label in self.reduction.bias_data_list:
            ax.plot(x, data, label=label)
        ax.plot(x, self.reduction.master_bias, label='Master Bias', linewidth=2, color='black')
        ax.set_title('Bias Frames and Master Bias')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Intensity')
        ax.legend(fontsize='small', loc='upper right')

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_flats(self, parent):
        """Plots all flat frames, combined flat, Chebyshev fit, and master flat."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        # Create three subplots
        fig = Figure(figsize=(10, 12), dpi=100)
        axs = fig.subplots(3, 1)

        # Plot normalized flats
        ax = axs[0]
        for x, data, label in self.reduction.flat_data_list:
            ax.plot(x, data, label=label)
        ax.set_title('Normalized Flat Frames')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Normalized Intensity')
        ax.legend(fontsize='small', loc='upper right')

        # Plot combined flat and Chebyshev fit
        x = np.arange(len(self.reduction.combined_flat))
        ax = axs[1]
        ax.plot(x, self.reduction.combined_flat, label='Combined Flat')
        ax.plot(x, self.reduction.y_fit, label='Chebyshev Fit', linestyle='--')
        ax.set_title('Combined Flat and Chebyshev Fit')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Normalized Intensity')
        ax.legend()

        # Plot master flat
        ax = axs[2]
        ax.plot(x, self.reduction.master_flat, label='Master Flat', color='green')
        ax.set_title('Master Flat')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Normalized Intensity')
        ax.legend()

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_calibration(self, parent):
        """Plots the calibration steps including residuals plot."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(10, 16), dpi=100)  # Increase figure size to fit residuals plot
        axs = fig.subplots(3, 1)  # Using 3 subplots: ThAr spectrum, wavelength calibration, and residuals scatter

        # Plot ThAr spectrum with identified peaks
        ax = axs[0]
        xaxis = np.arange(len(self.reduction.thar_data))
        data = self.reduction.thar_data
        ax.plot(xaxis, data, label='ThAr Spectrum')
        
        # Mark the peaks
        peaks = self.reduction.calibration_peaks
        ax.plot(peaks, data[peaks], 'ro', label='Peaks')
        ax.set_title('ThAr Spectrum with Identified Peaks')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Intensity')
        ax.legend()

        # Plot wavelength calibration
        ax = axs[1]
        if self.reduction.wavelengths is not None:
            ax.plot(xaxis, self.reduction.wavelengths, label='Wavelength Calibration')
            ax.set_title('Wavelength Calibration')
            ax.set_xlabel('Pixel')
            ax.set_ylabel('Wavelength (Angstrom)')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No Wavelength Calibration Available', ha='center', va='center')

        # Plot residuals (use the ones calculated during calibration)
        ax = axs[2]
        if self.reduction.calibration_residuals is not None:
            # Use the saved residuals from the calibration process
            ax.scatter(self.reduction.atlas_wavelengths, self.reduction.calibration_residuals, label='Residuals (Final Fit)', color='purple')
            ax.set_title('Residuals of Wavelength Calibration')
            ax.set_xlabel('Wavelength (Angstrom)')
            ax.set_ylabel('Residual (Angstrom)')
            ax.grid()
            ax.legend()

            # Optionally print summary of residuals again if needed
            print('==============================')
            print('Final Wavelength Calibration Residuals')
            print(f"NLINES = {len(self.reduction.calibration_residuals)}")
            print(f"DEGREE = {len(self.reduction.calibration_coeffs)}")
            print(f"RESIDUALS AVG = {np.average(self.reduction.calibration_residuals)}")
            print(f"RESIDUALS RMS = {np.std(self.reduction.calibration_residuals)}")
            print(f"RESIDUALS RMS (in km/s) = {np.std(self.reduction.calibration_residuals) / np.average(self.reduction.atlas_wavelengths) * 3e5} km/s")
            print('==============================')
        else:
            ax.text(0.5, 0.5, 'No Residuals Available', ha='center', va='center')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def launch_peak_assignment(self, xaxis, data, allcens, prefilled_pairs=None):
        """Launches the Peak Assignment window and returns pixel-wavelength pairs."""
        # Create the PeakAssignmentWindow
        peak_window = PeakAssignmentWindow(self.root, xaxis, data, allcens, prefilled_pairs=prefilled_pairs)

        # Wait for the window to be closed
        self.root.wait_window(peak_window)

        # Retrieve the pixel-wavelength pairs
        return peak_window.pixel_wavelength_pairs

    def display_plots(self):
        """Displays the plots of biases, master bias, flats, master flat, and calibration steps."""
        self.update_status("Generating plots...")

        # Create a new window for plots
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Reduction and Calibration Plots")

        # Get screen dimensions
        screen_width = plot_window.winfo_screenwidth()
        screen_height = plot_window.winfo_screenheight()
        # Set window size to 80% of screen dimensions
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        plot_window.geometry(f"{window_width}x{window_height}")

        notebook = ttk.Notebook(plot_window)
        notebook.pack(fill='both', expand=True)

        # Bias Plot Tab
        bias_frame = ttk.Frame(notebook)
        notebook.add(bias_frame, text='Bias Frames')

        # Flat Plot Tab
        flat_frame = ttk.Frame(notebook)
        notebook.add(flat_frame, text='Flat Frames')

        # Calibration Plot Tab
        calib_frame = ttk.Frame(notebook)
        notebook.add(calib_frame, text='Calibration')

        # Master Bias Plot
        self.plot_biases(bias_frame)

        # Master Flat Plot
        self.plot_flats(flat_frame)

        # Calibration Plots
        self.plot_calibration(calib_frame)

        self.update_status("Plots generated.")

# ======================================================================
# Main Execution
# ======================================================================

def main():
    # Initialize the GUI
    root = tk.Tk()
    app = ReductionGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()