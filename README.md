# MCQ Bubble Sheet Grader and Grades Auto-Filler

This project assists Teaching Assistants (TAs) and Professors in automating the correction of MCQ bubble sheet exams and digitizing handwritten grade sheets into structured formats like Excel.

---

## Features

### Module 1: Grades Sheet Processor
1. **Digitize Handwritten Grade Sheets**:
   - Input: Photos of handwritten grade sheets captured using mobile phones.
   - Handles variations in:
     - Skewing, orientation, and scale.
     - Ink colors and pencil marks.
     - Row and column sizes.

2. **Convert Marks into Digital Format**:
   - Interprets symbols like:
     - `✓` → `5`
     - `𐄂`, `-` → `0`
     - Empty cells remain empty.
     - Stacked vertical or horizontal lines converted into numeric equivalents.
     - `?` → Empty cell with a red background.
## Features

3. **Outputs Structured Excel Sheets**:
   - Creates a final Excel file replicating the original grade sheet but in a digital format.

### Module 2: MCQ Bubble Sheet Grader
1. **Bubble Sheet Processing**:
   - Input: Scanned images of bubble sheets.
   - Handles different formats:
     - Varying number of questions and options.
     - Vertically aligned bubbles.

2. **Automatic Grading**:
   - Input Model Answer:
     ```
     A
     B
     C
     A
     A
     ```
   - Outputs an Excel sheet with:
     - Columns for Student ID and answers (Q1, Q2, ...).
     - `1` for correct answers and `0` for wrong answers.

3. **Customization**:
   - Grading values (e.g., question weight, penalties) are configurable via a file.

---

## Prerequisites

- **Python 3.8+**
- Required Libraries:
  - `opencv-python`
  - `numpy`
  - `pandas`
  - `openpyxl`

Install dependencies using:
```bash
pip install -r requirements.txt
```
Run the UI:
```bash
pyhton3 MCQ-and-Grades-Auto-Filler\Code\MCQ\modular_mcq.py 
