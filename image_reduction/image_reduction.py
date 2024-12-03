import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import astroalign as aa
from astropy.io import fits
import argparse
from astropy.visualization import (ImageNormalize, PercentileInterval, AsinhStretch)

# -----------------------------------------------------------------------------
# Script for Image Reduction, Alignment, Stacking, and Multiple Color Combinations
#
# This script performs image reduction and alignment for astronomical images.
# It includes the creation of master bias, master dark, master flat frames,
# reduces the science images, aligns them, stacks them per filter, and
# creates final color-combined images based on multiple color mappings.
#
# Directory Structure:
# - data/photo/offsets/     : Contains bias frames (offsets)
# - data/photo/dark/        : Contains dark frames
# - data/photo/flats/       : Contains flat frames for various filters
#     - data/photo/flats/B/ : Flats for B filter
#     - data/photo/flats/R/ : Flats for R filter
#     - etc.
# - data/photo/M1/          : Contains science images
#
# The script allows the user to select which data to process, including filters
# and whether to include dark frame processing.
# -----------------------------------------------------------------------------

# Define functions for reading and writing FITS images
def read_raw_image(fits_file, get_header=0):
    """
    Reads a FITS file and returns the data and optionally the header.

    Parameters:
        fits_file (str): Path to the FITS file.
        get_header (int): If 1, returns a tuple (data, header). Otherwise, returns data.

    Returns:
        data (numpy.ndarray): Image data.
        header (astropy.io.fits.Header, optional): FITS header.
    """
    with fits.open(fits_file) as hdulist:
        header = hdulist[0].header
        data = hdulist[0].data
    if get_header == 0:
        return data
    else:
        return data, header

def write_fits_image(target_file, data, header=None):
    """
    Writes data to a FITS file with an optional header.

    Parameters:
        target_file (str): Path to the output FITS file.
        data (numpy.ndarray): Image data to write.
        header (astropy.io.fits.Header, optional): FITS header to include.
    """
    hdu = fits.PrimaryHDU(data, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(target_file, overwrite=True)
    print(f'  - Wrote {target_file}')

# Function to create directories
def create_directory(path):
    """
    Creates a directory if it does not exist.

    Parameters:
        path (str): Path of the directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"  - Directory created: {path}")
    else:
        print(f"  - Directory already exists: {path}")

# Function to create master bias from bias frames
def create_master_bias(bias_files, output_dir):
    """
    Creates a master bias frame by taking the median of all bias frames.

    Parameters:
        bias_files (list): List of bias FITS files.
        output_dir (str): Directory to save the master bias.

    Returns:
        target_file (str): Path to the saved master bias FITS file.
    """
    print("\n>>> Creating Master Bias")
    nfiles = len(bias_files)
    print(f"  - {nfiles} bias files found")

    # Read the first bias file to get the shape and header
    data, header_ori = read_raw_image(bias_files[0], get_header=1)
    nx, ny = np.shape(data)
    datastore = np.zeros((nx, ny, nfiles))

    for i in range(nfiles):
        print(f"  > Reading bias file {i+1}/{nfiles}: {bias_files[i]}")
        datastore[:, :, i] = read_raw_image(bias_files[i])

    # Compute median and save as master_bias
    result = np.median(datastore, axis=2)
    target_file = os.path.join(output_dir, 'master_bias.fits')
    header = header_ori
    header['OBJECT'] = 'MASTER_BIAS'
    write_fits_image(target_file, result, header)

    print(f"  - Master bias created and saved as: {target_file}\n")
    return target_file

# Function to create master dark from dark frames
def create_master_dark(dark_files, master_bias_file, output_dir):
    """
    Creates a master dark frame by subtracting the master bias from each dark frame
    and then taking the median.

    Parameters:
        dark_files (list): List of dark FITS files.
        master_bias_file (str): Path to the master bias FITS file.
        output_dir (str): Directory to save the master dark.

    Returns:
        target_file (str): Path to the saved master dark FITS file.
    """
    print("\n>>> Creating Master Dark")
    nfiles = len(dark_files)
    print(f"  - {nfiles} dark files found")

    # Read the first dark file to get the shape and header
    data, header_ori = read_raw_image(dark_files[0], get_header=1)
    nx, ny = np.shape(data)
    datastore = np.zeros((nx, ny, nfiles))

    master_bias = read_raw_image(master_bias_file)

    for i in range(nfiles):
        print(f"  > Reading dark file {i+1}/{nfiles}: {dark_files[i]}")
        dark_data = read_raw_image(dark_files[i]) - master_bias
        datastore[:, :, i] = dark_data

    # Compute median and save as master_dark
    result = np.median(datastore, axis=2)
    target_file = os.path.join(output_dir, 'master_dark.fits')
    header = header_ori
    header['OBJECT'] = 'MASTER_DARK'
    write_fits_image(target_file, result, header)

    print(f"  - Master dark created and saved as: {target_file}\n")
    return target_file

# Function to create a master flat by normalizing and subtracting bias and dark
def create_master_flat(flat_files, master_bias_file, master_dark_file, filter_name, output_dir):
    """
    Creates a master flat frame by subtracting bias and dark frames from each flat,
    normalizing them, and then taking the median.

    Parameters:
        flat_files (list): List of flat FITS files for a specific filter.
        master_bias_file (str): Path to the master bias FITS file.
        master_dark_file (str): Path to the master dark FITS file.
        filter_name (str): Name of the filter (e.g., 'B', 'R', 'V', 'Halpha', 'OIII').
        output_dir (str): Directory to save the master flat.

    Returns:
        target_file (str): Path to the saved master flat FITS file.
    """
    print(f"\n>>> Creating Master Flat for {filter_name} filter")
    nfiles = len(flat_files)
    print(f"  - {nfiles} flat files found")

    # Read the first flat file to get the shape and header
    data, header_ori = read_raw_image(flat_files[0], get_header=1)
    nx, ny = np.shape(data)
    datastore = np.zeros((nx, ny, nfiles))

    master_bias = read_raw_image(master_bias_file)
    master_dark = read_raw_image(master_dark_file)

    # Process flats: subtract bias and dark, then normalize
    for i in range(nfiles):
        print(f"  > Processing flat file {i+1}/{nfiles}: {flat_files[i]}")
        flat_data = read_raw_image(flat_files[i])
        flat_corrected = flat_data - master_bias - master_dark
        median_val = np.median(flat_corrected)
        if median_val == 0:
            print(f"    ! Warning: Median of flat_corrected is zero for {flat_files[i]}. Skipping normalization.")
            flat_normalized = flat_corrected
        else:
            flat_normalized = flat_corrected / median_val
        datastore[:, :, i] = flat_normalized

    # Compute median flat and save as master flat
    result = np.median(datastore, axis=2)
    target_file = os.path.join(output_dir, f'master_flat_{filter_name}.fits')
    header = header_ori
    header['OBJECT'] = f'MASTER_FLAT_{filter_name}'
    write_fits_image(target_file, result, header)

    print(f"  - Master flat created and saved as: {target_file}\n")
    return target_file

# Function to reduce science images using bias, dark, and flat frames
def reduce_science_images(science_files, master_bias_file, master_dark_file, master_flat_file, filter_name, output_dir):
    """
    Reduces science images by subtracting bias and dark frames and dividing by the master flat.

    Parameters:
        science_files (list): List of science FITS files to reduce.
        master_bias_file (str): Path to the master bias FITS file.
        master_dark_file (str): Path to the master dark FITS file.
        master_flat_file (str): Path to the master flat FITS file.
        filter_name (str): Name of the filter corresponding to these science images.
        output_dir (str): Directory to save the reduced science images.

    Returns:
        reduced_files (list): List of paths to the reduced science FITS files.
    """
    print(f"\n>>> Reducing Science Images with {filter_name.upper()} filter")

    master_bias = read_raw_image(master_bias_file)
    master_dark = read_raw_image(master_dark_file)
    master_flat = read_raw_image(master_flat_file)

    reduced_files = []

    for idx, file in enumerate(science_files, start=1):
        print(f"  > Reducing science image: {file}")
        data, header = read_raw_image(file, get_header=1)
        reduced_data = (data - master_bias - master_dark) / master_flat

        # Construct the reduced file name
        base_name = os.path.basename(file)
        # Insert the filter name before the file extension
        name_parts = os.path.splitext(base_name)
        reduced_file = os.path.join(output_dir, f"{name_parts[0]}_{filter_name}_reduced{''.join(name_parts[1:])}")

        write_fits_image(reduced_file, reduced_data, header)
        print(f"    - Reduced image saved to: {reduced_file}")
        reduced_files.append(reduced_file)

    return reduced_files

# Function to align images
def align_images(reference_file, target_files, output_dir):
    """
    Aligns target images to a reference image using astroalign.

    Parameters:
        reference_file (str): Path to the reference FITS file.
        target_files (list): List of target FITS files to align.
        output_dir (str): Directory to save the aligned images.
    """
    print("\n>>> Aligning Target Images")
    ref_data = read_raw_image(reference_file)
    print(f"  - Reference image: {reference_file}")

    # Save a copy of the reference image in the align directory with _align suffix
    base_ref, ext_ref = os.path.splitext(os.path.basename(reference_file))
    reference_align_file = os.path.join(output_dir, f'{base_ref}_align{ext_ref}')
    write_fits_image(reference_align_file, ref_data)
    print(f"  - Reference image saved as: {reference_align_file}")

    # Align target images
    for target_file in target_files:
        print(f"  > Aligning target image: {target_file}")
        target_data = read_raw_image(target_file)

        try:
            # Align the target image to the reference image
            aligned_data, footprint = aa.register(np.float32(target_data), np.float32(ref_data))
        except Exception as e:
            print(f"    ! Alignment failed for {target_file}: {e}")
            continue

        # Generate the filename for the aligned image
        base, ext = os.path.splitext(os.path.basename(target_file))
        aligned_file = os.path.join(output_dir, f'{base}_align{ext}')

        # Save the aligned image
        write_fits_image(aligned_file, aligned_data)
        print(f"    - Aligned image saved to: {aligned_file}")

# Function to rename aligned files to simpler names
def rename_aligned_files(aligned_dir, filters_to_process):
    """
    Renames aligned reduced FITS files to a simpler naming convention.

    Parameters:
        aligned_dir (str): Directory containing aligned reduced FITS files.
        filters_to_process (list): List of filter names processed.

    Returns:
        renamed_files_dict (dict): Dictionary with filter names as keys and lists of renamed file paths as values.
    """
    print("\n>>> Renaming Aligned Reduced Files to Simpler Names")
    renamed_files_dict = {filter_name: [] for filter_name in filters_to_process}

    # Iterate over each filter
    for filter_name in filters_to_process:
        # Find aligned files for the filter
        pattern = f"*_{filter_name}_reduced_align.fits"
        search_pattern = os.path.join(aligned_dir, pattern)
        aligned_files = glob.glob(search_pattern)

        if not aligned_files:
            print(f"  ! No aligned files found for filter {filter_name}.")
            continue

        # Sort files to ensure consistent ordering
        aligned_files_sorted = sorted(aligned_files)

        # Rename each file with a sequential number
        for idx, file in enumerate(aligned_files_sorted, start=1):
            new_name = f"{filter_name}_{idx:03d}_aligned.fits"
            new_path = os.path.join(aligned_dir, new_name)
            os.rename(file, new_path)
            print(f"    - Renamed {file} to {new_path}")
            renamed_files_dict[filter_name].append(new_path)

    return renamed_files_dict

# Function to stack images per filter
def stack_images(renamed_files_dict, stacked_dir):
    """
    Stacks aligned reduced images for each filter and saves the stacked images.

    Parameters:
        renamed_files_dict (dict): Dictionary with filter names as keys and lists of renamed file paths as values.
        stacked_dir (str): Directory to save the stacked images.

    Returns:
        stacked_files (dict): Dictionary with filter names as keys and paths to stacked FITS files as values.
    """
    print("\n>>> Stacking Aligned Reduced Images per Filter")
    stacked_files = {}

    for filter_name, files in renamed_files_dict.items():
        if not files:
            print(f"  ! No files to stack for filter {filter_name}. Skipping...")
            continue

        print(f"\n  > Stacking {len(files)} images for filter {filter_name}")
        stack_data = []

        for file in files:
            data = read_raw_image(file)
            stack_data.append(data)

        # Convert list to 3D numpy array for stacking
        stack_array = np.array(stack_data)

        # Perform median stacking
        stacked_image = np.median(stack_array, axis=0)

        # Save the stacked image
        stacked_file = os.path.join(stacked_dir, f'master_stacked_{filter_name}.fits')
        header = fits.getheader(files[0])
        header['OBJECT'] = f'MASTER_STACKED_{filter_name}'
        write_fits_image(stacked_file, stacked_image, header)
        print(f"    - Stacked image saved to: {stacked_file}")

        stacked_files[filter_name] = stacked_file

    return stacked_files

# Function to create final color images
def create_color_images(stacked_files, color_combinations, stacked_dir):
    """
    Creates final color images by combining stacked images from different filters based on multiple color combinations.

    Parameters:
        stacked_files (dict): Dictionary with filter names as keys and paths to stacked FITS files as values.
        color_combinations (list): List of dictionaries mapping RGB channels to filter names.
        stacked_dir (str): Directory where stacked images are saved.
    """
    print("\n>>> Creating Final Color Images")

    for combo_idx, color_combination in enumerate(color_combinations, start=1):
        print(f"\n  > Creating Color Combination {combo_idx}: {color_combination}")
        # Initialize RGB channels
        rgb_channels = {'R': None, 'G': None, 'B': None}

        for color, filter_name in color_combination.items():
            if filter_name in stacked_files:
                data = read_raw_image(stacked_files[filter_name])
                # Normalize the data using AsinhStretch for better visualization
                norm = ImageNormalize(data, interval=PercentileInterval(99.5), stretch=AsinhStretch())
                norm_data = norm(data)
                # If norm_data is a masked array, fill masked regions with 0
                if isinstance(norm_data, np.ma.MaskedArray):
                    norm_data = norm_data.filled(0)
                # Clip values to [0,1] for image display
                norm_data = np.clip(norm_data, 0, 1)
                rgb_channels[color] = norm_data
                print(f"    - Loaded and normalized {filter_name} for {color} channel")
            else:
                print(f"    ! No stacked image found for filter {filter_name} to assign to {color} channel.")
                rgb_channels[color] = np.zeros_like(next(iter(stacked_files.values()), None))

        # Check if at least one channel is available
        if all(v is None for v in rgb_channels.values()):
            print("    ! No channels available to create a color image.")
            continue

        # Replace None channels with zeros
        for color in rgb_channels:
            if rgb_channels[color] is None:
                rgb_channels[color] = np.zeros_like(next(iter(stacked_files.values()), None))

        # Stack channels into an RGB image
        rgb_image = np.dstack((rgb_channels['R'], rgb_channels['G'], rgb_channels['B']))

        # Ensure rgb_image is a regular NumPy array (not a MaskedArray)
        if isinstance(rgb_image, np.ma.MaskedArray):
            rgb_image = rgb_image.filled(0)

        # Convert to float32 for FITS saving
        rgb_image = rgb_image.astype(np.float32)

        # Display the RGB image
        plt.figure(figsize=(8, 8))
        plt.imshow(rgb_image, origin='lower')
        plt.axis('off')
        plt.title(f'Final RGB Image Combination {combo_idx}')

        # Save the RGB image as PNG with enhanced contrast
        rgb_png = os.path.join(stacked_dir, f'final_rgb_image_combo_{combo_idx}.png')
        plt.savefig(rgb_png, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"    - Final RGB PNG image saved to: {rgb_png}")

        # Save the RGB image as a FITS file correctly
        rgb_fits = os.path.join(stacked_dir, f'final_rgb_image_combo_{combo_idx}.fits')
        # Create a FITS header
        header = fits.Header()
        header['OBJECT'] = f'FINAL_RGB_COMBO_{combo_idx}'
        header['COMMENT'] = f'RGB Combination {combo_idx}: ' + ', '.join([f"{k}:{v}" for k, v in color_combination.items()])
        hdu = fits.PrimaryHDU(rgb_image, header=header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(rgb_fits, overwrite=True)
        print(f"    - Final RGB FITS image saved to: {rgb_fits}")

# Function to generate color combinations
def get_color_combinations():
    """
    Defines multiple color combinations for final image creation.

    Returns:
        color_combinations (list): List of dictionaries mapping RGB channels to filter names.
    """
    # Define multiple color combinations
    color_combinations = [
        {'R': 'R', 'G': 'V', 'B': 'B'},          # Standard RGB
        {'R': 'Halpha', 'G': 'V', 'B': 'OIII'}  # Narrow-band RGB
    ]
    return color_combinations

# Main function to process images
def process_images(filters_to_process, process_dark_frames, science_data_path):
    """
    Orchestrates the image reduction, alignment, stacking, and color combination process.

    Parameters:
        filters_to_process (list): List of filter names to process.
        process_dark_frames (bool): Whether to process dark frames.
        science_data_path (str): Path to the directory containing science images.
    """
    print(">>> Starting Image Reduction, Alignment, Stacking, and Color Combination Process\n" + "="*50)

    # Define directories for output
    base_output_dir = "output"
    reduced_dir = os.path.join(base_output_dir, "reduced")
    aligned_dir = os.path.join(base_output_dir, "aligned")
    stacked_dir = os.path.join(base_output_dir, "stacked")

    create_directory(base_output_dir)
    create_directory(reduced_dir)
    create_directory(aligned_dir)
    create_directory(stacked_dir)

    # Step 1: Create master_bias
    bias_files = glob.glob(os.path.join('data', 'photo', 'offsets', '*.fits'))
    if not bias_files:
        print("  ! No bias files found. Exiting...")
        return

    master_bias_file = create_master_bias(bias_files, reduced_dir)

    # Step 2: Optionally create master_dark
    master_dark_file = None
    if process_dark_frames:
        dark_files = glob.glob(os.path.join('data', 'photo', 'dark', '*.fits'))
        if not dark_files:
            print("  ! No dark files found. Exiting...")
            return
        master_dark_file = create_master_dark(dark_files, master_bias_file, reduced_dir)
    else:
        # Create a zero dark frame
        data_example = read_raw_image(bias_files[0])
        master_dark = np.zeros_like(data_example)
        master_dark_file = os.path.join(reduced_dir, 'master_dark.fits')
        header = fits.getheader(bias_files[0])
        header['OBJECT'] = 'MASTER_DARK'
        write_fits_image(master_dark_file, master_dark, header)
        print(f"  - No dark frames provided. Using zero master dark: {master_dark_file}")

    reference_file = None
    all_reduced_files = []

    # Process each filter
    for filter_name in filters_to_process:
        print(f"\n>>> Processing filter: {filter_name.upper()}")

        # Find flats for the filter
        flat_dir = os.path.join('data', 'photo', 'flats', filter_name)
        flat_files = glob.glob(os.path.join(flat_dir, '*.fits'))
        if not flat_files:
            print(f"  ! No flat files found for filter {filter_name}. Skipping...")
            continue

        master_flat_file = create_master_flat(flat_files, master_bias_file, master_dark_file, filter_name, reduced_dir)

        # Find science images for the filter by checking filenames
        all_science_files = glob.glob(os.path.join(science_data_path, '*.fits'))
        # Filter science files that contain the filter name in their filename
        matching_science_files = [f for f in all_science_files if filename_contains_filter(os.path.basename(f), filter_name)]

        if not matching_science_files:
            print(f"  ! No science files found for filter {filter_name}. Skipping...")
            continue

        reduced_files = reduce_science_images(
            matching_science_files,
            master_bias_file,
            master_dark_file,
            master_flat_file,
            filter_name,
            reduced_dir
        )
        all_reduced_files.extend(reduced_files)

        # Set the reference file for alignment (e.g., first reduced image of the first filter)
        if not reference_file and reduced_files:
            reference_file = reduced_files[0]
            print(f"  - Reference file set to: {reference_file}")

    # Step 3: Align images
    if reference_file and all_reduced_files:
        # Exclude the reference file from target files
        target_files = [f for f in all_reduced_files if f != reference_file]
        align_images(reference_file, target_files, aligned_dir)
    else:
        print("  ! No reference or target files found for alignment.")

    # Step 4: Rename aligned files to simpler names
    renamed_files_dict = rename_aligned_files(aligned_dir, filters_to_process)

    # Step 5: Stack images per filter
    stacked_files = stack_images(renamed_files_dict, stacked_dir)

    # Step 6: Create final color images based on multiple color combinations
    color_combinations = get_color_combinations()
    create_color_images(stacked_files, color_combinations, stacked_dir)

    print("\n>>> Image Reduction, Alignment, Stacking, and Color Combination Process Completed\n" + "="*50)

# Helper function to determine if a filename contains the filter name
def filename_contains_filter(filename, filter_name):
    """
    Checks if the filename contains the filter name as a substring (case-insensitive).

    Parameters:
        filename (str): The name of the file.
        filter_name (str): The filter name to search for.

    Returns:
        bool: True if the filter name is found in the filename, False otherwise.
    """
    return filter_name.lower() in filename.lower()

# Function to generate color combinations
def get_color_combinations():
    """
    Defines multiple color combinations for final image creation.

    Returns:
        color_combinations (list): List of dictionaries mapping RGB channels to filter names.
    """
    # Define multiple color combinations
    color_combinations = [
        {'R': 'R', 'G': 'V', 'B': 'B'},          # Standard RGB
        {'R': 'Halpha', 'G': 'V', 'B': 'OIII'}  # Narrow-band RGB
    ]
    return color_combinations

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process astronomical images.')
    parser.add_argument(
        '--filters',
        nargs='+',
        default=['B', 'R', 'V', 'Halpha', 'OIII'],
        help='List of filters to process (e.g., --filters B V R Halpha OIII)'
    )
    parser.add_argument(
        '--dark',
        action='store_true',
        default=True,
        help='Include this flag to process dark frames'
    )
    parser.add_argument(
        '--science_path',
        default=os.path.join('data', 'photo', 'M1'),
        help='Path to science images (default: data/photo/M1)'
    )
    args = parser.parse_args()

    filters_to_process = args.filters
    process_dark_frames = args.dark
    science_data_path = args.science_path

    # Validate filter names
    valid_filters = ['B', 'R', 'V', 'Halpha', 'OIII']
    for filt in filters_to_process:
        if filt not in valid_filters:
            print(f"  ! Invalid filter name: {filt}. Valid options are: {valid_filters}")
            exit(1)

    process_images(filters_to_process, process_dark_frames, science_data_path)