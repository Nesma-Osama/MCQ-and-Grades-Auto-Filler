# Grades Sheet Processing System

## Overview
The Grades Sheet Processing System is a software tool designed to process images of printed grade sheets and extract relevant data into an Excel spreadsheet. The system handles imperfections in the input images such as skewing, scaling, and handwriting variability to produce a structured and accurate digital representation of the grades.

## Features
- **Image Preprocessing:**
  - Handles skewed, scaled, or angled images (excluding upside-down).
  - Adjusts for various ink colors and handwritten grades.
  - Accommodates varying table sizes, row heights, and column widths.
  
- **Data Extraction:**
  - Detects and extracts printed student IDs using two methods:
    1. Pretrained OCR (Tesseract).
    2. Custom feature extraction and classification (user selectable).
  - Converts handwritten symbols into structured data with the following mappings:
    - `✓` (Checkmark) → **5**
    - `ፂ` or `-` (Cross or dash) → **0**
    - Empty cell → **Empty cell**
    - Stacked vertical lines (`|||`) → **i**, where `i` is the number of lines.
    - Stacked horizontal lines (`---`) → **(5 - i)`, where `i` is the number of lines.
    - `?` → **Empty cell with a red background**.

- **Output:**
  - Generates an Excel file replicating the original format, either by filling the original template or creating a new sheet.
  - Handles numeric handwritten values using OCR or a custom classifier (user selectable).

## Requirements
### Software Dependencies
- Python 3.x
- OpenCV
- NumPy
- Tesseract OCR
- scikit-learn
- OpenPyXL
- PIL (Pillow)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Tesseract OCR is installed and added to the system path.

## Usage
### Step 1: Prepare the Image
- Capture an image of the grades sheet with clear visibility of all cells.
- Ensure the image is not upside-down.

### Step 2: Run the Application
1. Launch the application:
   ```bash
   python main.py
   ```
2. Select the processing mode:
   - OCR for printed student IDs and handwritten grades.
   - Feature-based classifier for numeric values and student IDs.
3. Upload the image of the grades sheet.
4. Configure the output settings:
   - Specify the output file name and location.
   - Choose whether to fill the original template or create a new sheet.

### Step 3: View the Output
- The processed data will be saved as an Excel file in the specified location.

## Data Processing Workflow
1. **Image Preprocessing:**
   - Convert the image to grayscale.
   - Apply edge detection (Canny) and dilation to enhance features.
   - Detect and correct the largest rectangular contour to align the grid.

2. **Grid Extraction:**
   - Segment the grid into rows and columns.
   - Extract individual cells for further processing.

3. **Symbol and Numeric Value Recognition:**
   - Use OCR or a custom SVM-based classifier for numeric values.
   - Map handwritten symbols to corresponding numeric outputs.

4. **Excel Generation:**
   - Populate the extracted data into an Excel sheet.
   - Highlight empty cells with a red background where appropriate.

## Customization
Users can customize various aspects of the system, including:
- Threshold values for edge detection and contour finding.
- Classifier models for numeric and symbol recognition.
- Output formatting options for the Excel file.

## Limitations
- The system does not support upside-down images.
- Highly smudged or unclear handwriting may impact accuracy.

## Future Enhancements
- Add support for additional symbol mappings.
- Improve handwriting recognition for challenging cases.
- Implement real-time processing for video input.

## Acknowledgments
Special thanks to all contributors who supported the development of this project.

