__author__ = "Jordan McKee"
__email__ = "jwm109@uakron.edu"

import sys
import time
import numpy as np


def read_pgm(filename):
    """
    Reads a PGM (portable graymap) file in P2 format and returns a 2D array of pixel values.

    Params:
        filename (str): The path to the PGM file.

    Returns:
        A list of lists of ints representing the pixel values of the image.

    Raises:
        ValueError: If the file is not in P2 format.
    """
    with open(filename, "r") as f:
        # Check the PGM header
        magic_number = f.readline().strip()
        if magic_number != "P2":
            raise ValueError("File is not in P2 format.")

        # Parse the image width, height, and maximum gray value
        width, height = None, None
        max_gray_value = None
        while width is None or height is None or max_gray_value is None:
            line = f.readline().strip()
            if line.startswith("#"):
                continue  # Comment line, ignore
            if width is None and height is None:
                width = int(line.split()[0])
                height = int(line.split()[1])
            elif max_gray_value is None:
                max_gray_value = int(line)

        # Read the image data
        img_data = f.read().split()

    # Convert the text image data to a 2D array of pixel values
    pixels = [[0] * width for _ in range(height)]
    for i in range(height):
        for j in range(width):
            pixel_value = float(img_data[i * width + j])
            pixels[i][j] = pixel_value

    return pixels


def compute_energy(pixels):
    """
    Computes the energy of each pixel in the image.

    Params:
        pixels (list of lists of ints): The pixel values of the image.

    Returns:
        A 2D array of energy values for each pixel in the image.
    """
    height, width = len(pixels), len(pixels[0])
    energy = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            current = pixels[i][j]
            left = pixels[i][j - 1] if j > 0 else pixels[i][j]
            right = pixels[i][j + 1] if j < width - 1 else pixels[i][j]
            up = pixels[i - 1][j] if i > 0 else pixels[i][j]
            down = pixels[i + 1][j] if i < height - 1 else pixels[i][j]
            energy[i][j] = (
                abs(current - down)
                + abs(current - up)
                + abs(current - left)
                + abs(current - right)
            )

    return energy


def calculate_cumulative_energy(energy):
    """
    Calculates the cumulative energy of the image.

    Params:
        energy (list of lists of floats): The energy of each pixel in the image.

    Returns:
        A list of lists of floats representing the cumulative energy of the image.
    """
    # Initialize the cumulative energy matrix with the energy of the first row
    cumulative_energy = energy.copy()

    # Calculate the cumulative energy for each subsequent row
    for i in range(1, len(energy)):
        for j in range(len(energy[0])):
            # Find the minimum cumulative energy of the pixels above and to the left of the current pixel
            if j == 0:
                cumulative_energy[i][j] += min(
                    cumulative_energy[i - 1][j], cumulative_energy[i - 1][j + 1]
                )
            elif j == len(energy[0]) - 1:
                cumulative_energy[i][j] += min(
                    cumulative_energy[i - 1][j - 1], cumulative_energy[i - 1][j]
                )
            else:
                cumulative_energy[i][j] += min(
                    cumulative_energy[i - 1][j - 1],
                    cumulative_energy[i - 1][j],
                    cumulative_energy[i - 1][j + 1],
                )

    return cumulative_energy


def remove_vertical_seam(pixels, cumulative_energy):
    """
    Iterate cumulative_energy 2D list to find a vertical seam and remove it from the pixels image

    Params:
        pixels (list of lists): Represents the pixel values of the modified image.
        cumulative_energy (list of lists): The representation of the cumulative energy in the pixels list

    Returns:
        A list of lists of ints representing the pixel values of the image after removing a vertical seam.
    """
    height, width = cumulative_energy.shape
    start_pixel = np.argmin(cumulative_energy[-1])
    del pixels[-1][start_pixel]

    # Traverse upwards and delete the pixels that belong to the seam
    for i in range(height - 2, -1, -1):
        if start_pixel == 0:
            # Handle the left edge of the image
            next_pixel = np.argmin(cumulative_energy[i, start_pixel : start_pixel + 2])
            start_pixel += next_pixel
        elif start_pixel == width - 1:
            # Handle the right edge of the image
            next_pixel = np.argmin(
                cumulative_energy[i, start_pixel - 1 : start_pixel + 1]
            )
            start_pixel += next_pixel - 1
        else:
            # Handle the general case
            next_pixel = np.argmin(
                cumulative_energy[i, start_pixel - 1 : start_pixel + 2]
            )
            start_pixel += next_pixel - 1

        # Mark the next pixel as part of the seam
        del pixels[i][start_pixel]

    return pixels


def carve_seam(pixels, num_seams, vertical):
    """
    Carves num_seams vertical seams from the image that pixels represents

    Params:
        pixels (list of lists): Represents the pixel values of the modified image.
        num_seams (int): The number of vertical seams to remove

    Returns:
        A list of lists of ints representing the pixel values of the image.
    """
    # rotate pixels array if horizontal seam
    if not vertical:
        pixels = np.rot90(pixels, 1).tolist()

    for _ in range(num_seams):
        # Compute the energy of the image
        energy = compute_energy(pixels)

        # Compute the cumulative energy of the image
        cumulative_energy = calculate_cumulative_energy(energy)

        # remove seam
        pixels = remove_vertical_seam(pixels, cumulative_energy)

    # rotate pixels back to correct orientation
    if not vertical:
        pixels = np.rot90(pixels, -1).tolist()

    return pixels


# ___________________
# MAIN

start = time.time()
# verify correct args were passed to the program
if len(sys.argv) != 4 or not sys.argv[2].isdigit() or not sys.argv[3].isdigit():
    print(
        'Wrong format! Should be "python <filename>.py <imageName> <verticalSeams> <horizontalSeams>"'
    )
    exit(1)

vertical_seams = int(sys.argv[2])
horizontal_seams = int(sys.argv[3])
pgm_file = sys.argv[1]
pixels = read_pgm(pgm_file)  # load pgm file into a 2d array

if vertical_seams >= len(pixels[0]) or horizontal_seams >= len(pixels):
    print("Seams given to carve are too large for given image!\n")
    exit(1)

# carve vertical seams
pixels = carve_seam(pixels, vertical_seams, True)

# carve horizontal seams
pixels = carve_seam(pixels, horizontal_seams, False)

# save result to new .pgm file
height = len(pixels)
width = len(pixels[0])

new_filename = pgm_file.removesuffix(".pgm") + "_processed_{}_{}.pgm".format(
    vertical_seams, horizontal_seams
)
with open(new_filename, "w") as f:
    # Write the PGM header to the file
    f.write("P2\n# delicately carved by jordan\n{} {}\n255\n".format(width, height))

    # Write the pixel data to the file
    for row in pixels:
        for pixel in row:
            f.write("{} ".format(int(pixel)))
        f.write("\n")

    # Close the file
    f.close()

# get execution time
end = time.time()
print("runtime: ", end - start, "s")
