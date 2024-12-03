Spectral Data Reduction and Calibration Script with Enhanced GUI

Author: Bibal Sobeaux Pierre Gabriel
Date: September 25, 2024

This script is designed for astronomers and scientists who need to perform spectral data reduction, including bias subtraction, flat-field correction, and wavelength calibration using ThAr lamp spectra. It features an enhanced GUI for monitoring the process, inspecting data, and interactively assigning wavelengths to spectral lines. Additionally, it allows stacking multiple reduced spectra using various stacking methods.

Table of Contents

	•	Features
	•	Prerequisites
	•	Installation
	•	Directory Structure
	•	Data Preparation
	•	Usage
	•	Running the Script
	•	GUI Overview
	•	Data Reduction Steps
	•	Creating Master Bias
	•	Creating Master Flat
	•	Wavelength Calibration
	•	First Guess Wavelength Calibration
	•	Finalizing Calibration
	•	Reducing Observations
	•	Generating and Viewing Plots
	•	Inspecting Data
	•	Stacking Spectra
	•	Screenshots
	•	Troubleshooting
	•	Contributing
	•	License
	•	Contact

Features

	•	Automated Data Reduction: Performs bias subtraction, flat-field correction, and wavelength calibration.
	•	Enhanced GUI: User-friendly interface for monitoring the reduction process and inspecting data.
	•	Interactive Peak Assignment: Allows for manual assignment of wavelengths to detected peaks.
	•	Spectra Stacking: Stack multiple reduced spectra using Median, Average, or Weighted Average methods.
	•	High-Resolution Plots: Generates detailed plots for debugging and analysis.

Prerequisites

	•	Python 3.6 or higher
	•	Required Python Packages:
	•	numpy
	•	matplotlib
	•	astropy
	•	scipy
	•	tkinter (included with Python)
	•	Additional Modules:
	•	threading
	•	tkinter.ttk
	•	tkinter.filedialog
	•	tkinter.messagebox
	•	tkinter.simpledialog

Installation

	1.	Clone the Repository

git clone https://github.com/pierregab/OHP.git
cd spectral_reduction


	2.	Install Dependencies

pip install numpy matplotlib astropy scipy

Note: Ensure that you have Python 3.6 or higher installed.

Directory Structure

The script expects a specific directory structure:

spectral_reduction/
├── spectral_reduction.py   # This script
├── Raw/                    # Directory containing raw observational spectra
├── bias/                   # Directory containing bias frames
├── Tung/                   # Directory containing flat frames (e.g., Tungsten lamp)
├── ThAr/                   # Directory containing calibration lamp spectra
├── Reduced/                # Directory where reduced spectra will be saved
└── linelists/              # Directory containing line lists for calibration

Ensure all required data files are placed in the corresponding directories.

Data Preparation

	•	Bias Frames: Place your bias FITS files in the bias/ directory.
	•	Flat Frames: Place your flat FITS files in the Tung/ directory.
	•	ThAr Lamp Spectra: Place your ThAr lamp FITS files in the ThAr/ directory.
	•	Raw Observational Spectra: Place your raw FITS files in the Raw/ directory.
	•	Line Lists: Place your line list files (e.g., ThI.csv) in the linelists/ directory.

Example data is provided in the repository for testing purposes. With this data, you can proceed directly to running the script and performing wavelength calibration by simply pressing “Done” when prompted.

Usage

Running the Script

Navigate to the spectral_reduction directory and run:

python spectral_reduction.py

GUI Overview

Upon running the script, a GUI window will appear with the following components:
	•	Status Label: Displays the current status of the process.
	•	Start Reduction: Begins the data reduction workflow.
	•	Inspect Data: Opens a file dialog to inspect individual FITS files.
	•	Show Gaussian Peaks: Visualizes detected peaks in the ThAr spectrum.
	•	Stack Spectra: Opens a dialog to select and stack reduced spectra.
	•	File Treeview: Displays processed files and their statuses.

Data Reduction Steps

The data reduction process is divided into several key steps, each accessible through the GUI.

Creating Master Bias

	1.	Start Reduction: Click the Start Reduction button.
	2.	Master Bias Creation:
	•	The script searches the bias/ directory for all bias frames.
	•	It combines these frames by taking the median to create a Master Bias.
	•	The Master Bias is saved as master_bias.fits in the bias/ directory.
	•	The status label updates to indicate completion.

Creating Master Flat

	1.	Master Flat Creation:
	•	After the Master Bias is created, the script proceeds to the flat-field correction.
	•	It searches the Tung/ directory for all flat frames.
	•	Each flat frame is bias-subtracted using the Master Bias and then normalized.
	•	The normalized flats are combined by taking the median to create a Combined Flat.
	•	A Chebyshev polynomial is fitted to the Combined Flat to remove any residual slope.
	•	The result is a Master Flat, saved as master_flat.fits in the Tung/ directory.
	•	The status label updates to indicate completion.

Wavelength Calibration

	1.	Initiate Calibration:
	•	The script automatically detects peaks in the first ThAr lamp spectrum found in the ThAr/ directory.
	•	A window pops up displaying the ThAr spectrum with detected peaks marked.
	2.	First Guess Wavelength Calibration:
	•	Automatic Prefilled Peaks:
	•	The script provides a set of prefilled pixel-wavelength pairs based on known line positions from the example data.
	•	These prefilled peaks are marked with blue ‘X’s on the spectrum plot.
	•	Interactive Peak Assignment:
	•	Selecting Peaks:
	•	Click on a detected peak (marked with red circles) in the spectrum plot.
	•	A prompt appears asking for the known wavelength corresponding to the selected peak.
	•	Entering Wavelengths:
	•	Enter the wavelength value (in Angstroms) for the selected peak.
	•	Assigned peaks are marked with green ‘X’s and listed in the side panel for reference.
	•	Example Data:
	•	If you’re using the provided example data, most peaks may already be prefilled. You can review these assignments and press Done if no further adjustments are needed.
	•	Completing Assignment:
	•	Once all necessary peaks are assigned, click the Done button to finalize the calibration.
	•	Finalizing Calibration:
	•	The script fits a Chebyshev polynomial to the assigned pixel-wavelength pairs.
	•	It evaluates the fit across the entire spectrum to establish a wavelength calibration.
	•	Residuals (differences between fitted and actual wavelengths) are calculated and displayed for quality assessment.
	•	The calibration is saved as wavelength_calibration.dat in the Reduced/ directory.
	•	The status label updates to indicate completion.

Reducing Observations

	1.	Apply Corrections:
	•	With the Master Bias, Master Flat, and wavelength calibration in place, the script processes each raw observational spectrum in the Raw/ directory.
	•	Each spectrum undergoes bias subtraction and flat-field correction.
	2.	Wavelength Calibration Application:
	•	The wavelength calibration is applied to each reduced spectrum.
	•	The FITS headers are updated with wavelength calibration details, including calibration coefficients.
	3.	Saving Reduced Spectra:
	•	The reduced and calibrated spectra are saved in the Reduced/ directory with filenames appended by _reduced.fits.
	•	The status label updates with each saved file.

Generating and Viewing Plots

After the data reduction process completes, the script generates several high-resolution plots for analysis:
	•	Bias Frames and Master Bias: Displays all individual bias frames and the combined Master Bias.
	•	Flat Frames and Master Flat: Shows all normalized flat frames, the Combined Flat, the Chebyshev fit, and the final Master Flat.
	•	Calibration Plots: Includes the ThAr spectrum with detected peaks, the wavelength calibration curve, and residuals.

These plots are accessible through the GUI and open in separate windows for detailed inspection.

Inspecting Data

	•	Inspect Data:
	•	Click the Inspect Data button.
	•	A file dialog appears; navigate to and select a FITS file from the Reduced/ directory.
	•	The selected spectrum is displayed in a new window with appropriate axes (Pixel or Wavelength).
	•	This feature allows for quick visual verification of individual spectra.

Wavelength Calibration

First Guess Wavelength Calibration

The first guess wavelength calibration is a crucial step in accurately mapping pixel positions to physical wavelengths. Here’s a detailed guide on how to perform this step:
	1.	Peak Detection:
	•	The script automatically detects peaks in the first ThAr lamp spectrum.
	•	Detected peaks are marked with red circles on the spectrum plot.
	2.	Prefilled Peak Assignments:
	•	The script provides prefilled pixel-wavelength pairs based on known line positions from the example data.
	•	These prefilled peaks are marked with blue ‘X’s and listed in the side panel.
	•	Note: If you are using custom data, you may need to assign wavelengths manually.
	3.	Interactive Peak Assignment:
	•	Selecting a Peak:
	•	Click on a red-marked peak in the spectrum plot.
	•	A prompt will appear asking for the known wavelength corresponding to that peak.
	•	Entering Wavelength:
	•	Input the wavelength value (in Angstroms) for the selected peak.
	•	Upon confirmation, the peak is marked with a green ‘X’ and added to the assignment list.
	•	Reviewing Assignments:
	•	Assigned peaks are listed in the side panel, categorized as “Assigned” or “Prefilled”.
	•	Ensure that the assigned wavelengths are accurate for optimal calibration.
	4.	Finalizing Calibration:
	•	After assigning all necessary peaks, click the Done button.
	•	The script will compute a Chebyshev polynomial fit based on the assigned pixel-wavelength pairs.
	•	It evaluates the fit across the entire spectrum, calculates residuals, and saves the calibration data.
	•	Example Data: With the provided example data, most peaks may already be prefilled, allowing you to simply review and press Done.

Finalizing Calibration

After completing the peak assignments:
	•	Polynomial Fit:
	•	A Chebyshev polynomial is fitted to the assigned pixel-wavelength pairs.
	•	This fit models the relationship between pixel positions and wavelengths across the spectrum.
	•	Residual Analysis:
	•	The script calculates residuals (differences between fitted and actual wavelengths) to assess calibration accuracy.
	•	Residual statistics, including average and RMS values, are printed in the console for review.
	•	Saving Calibration:
	•	The final wavelength calibration is saved as wavelength_calibration.dat in the Reduced/ directory.
	•	This calibration is applied to all reduced spectra, ensuring consistent wavelength mapping.

Reducing Observations

	1.	Apply Corrections:
	•	With the Master Bias, Master Flat, and wavelength calibration in place, the script processes each raw observational spectrum in the Raw/ directory.
	•	Each spectrum undergoes bias subtraction and flat-field correction.
	2.	Wavelength Calibration Application:
	•	The wavelength calibration is applied to each reduced spectrum.
	•	The FITS headers are updated with wavelength calibration details, including calibration coefficients.
	3.	Saving Reduced Spectra:
	•	The reduced and calibrated spectra are saved in the Reduced/ directory with filenames appended by _reduced.fits.
	•	The status label updates with each saved file.

Stacking Spectra

	1.	Initiate Stacking:
	•	Click the Stack Spectra button.
	•	A file dialog appears; navigate to and select the reduced FITS files you wish to stack from the Reduced/ directory.
	2.	Select Stacking Method:
	•	After selecting the files, a dialog will prompt you to choose a stacking method:
	•	Median: Takes the median value at each wavelength across all spectra.
	•	Average: Computes the mean value at each wavelength.
	•	Weighted Average: Calculates a weighted mean, allowing for differing weights per spectrum (currently using equal weights by default).
	3.	Executing Stacking:
	•	The script performs the stacking operation based on your chosen method.
	•	The stacked spectrum is saved in the Reduced/stacked/ directory with a filename derived from the common prefix of the selected files. If no common prefix is found, a default name stacked_spectrum.fits is used.
	•	The status label updates to indicate completion.
	4.	Viewing Stacked Spectrum:
	•	After stacking, the stacked spectrum is automatically displayed in a new window for inspection.

Contributing

Contributions are welcome!
	1.	Fork the Repository

git clone https://github.com/pierregab/OHP.git


	2.	Create a Feature Branch

git checkout -b feature/your-feature-name


	3.	Commit Your Changes

git commit -am 'Add some feature'


	4.	Push to the Branch

git push origin feature/your-feature-name


	5.	Open a Pull Request

License

This project is licensed under the MIT License - see the LICENSE file for details.

Contact

For questions or suggestions:
	•	Author: Bibal Sobeaux Pierre Gabriel
	•	Email: pierre.bibal-sobeaux@etu.unistra.fr