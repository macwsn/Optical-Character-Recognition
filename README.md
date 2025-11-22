To view evrything visit jupyter nootebook file(too big to do md readme)

# OCR Optical Character Recognition

## Project Description

This project contains two implementations of Optical Character Recognition (OCR):
- **Frequency domain method (FFT)** – uses Fourier transform for character comparison.
- **Image domain method (IOU)** – a classic approach based on Intersection over Union metric.

The code uses the following libraries: OpenCV, NumPy, SciPy, skimage, Matplotlib. Some helper functions are located in the `util` folder.

## Features

- Automatic generation of test images with text, various fonts, and rotations.
- Recognition of text from images containing English alphabet letters, digits, spaces, and special characters (?, !). The `~` character is reserved to indicate a recognition error.
- Support for multi-line texts.
- Text rotation correction (angle detection using the Hough method).
- Step-by-step visualizations of the OCR process.

## Usage Examples

- Recognition of automatically generated text images.
- Tests with different fonts and rotated images.
- Comparison of results with expected text – in all tested cases, 100% accuracy was achieved.

## Notebook Structure

1. **Import of libraries and helper functions**
2. **OCR implementation with FFT**
	- Function `simple_ocr_fft`
	- Example usage on a generated image
3. **OCR implementation with IOU**
	- Function `simple_ocr`
	- Example usage on a generated image
4. **Tests with other fonts**
5. **OCR with rotation correction**
	- Function `ocr_with_rotation_correction`
	- Example usage on a rotated image

## Requirements

- Python 3.x
- OpenCV
- NumPy
- SciPy
- skimage
- Matplotlib

## How to Run

1. Install the required libraries:
	```
	pip install opencv-python numpy scipy scikit-image matplotlib
	```
2. Run the notebook `ocr_sprawozdanie.ipynb` in Jupyter Notebook or VS Code.

## Results

Both methods achieve very high accuracy (100% in tests). The IOU method is more reliable for classic character recognition, while FFT allows for interesting analyses in the frequency domain.
