import cv2 as cv
import numpy as np
import pandas as pd
from tkinter import Tk, Label, Button, filedialog, messagebox, Frame
from tkinter.font import Font
from utils import (
    find_biggest_contour, show_images, wrapped_paper, cropp_box_image,
    replace_image_with_white, finall_extract, correct_id_mcq, split_questions
)

class MCQGrader:
    def __init__(self, width=480, height=480):
        self.width = width
        self.height = height

    def read_image(self, path):
        return cv.resize(cv.imread(path), (self.width, self.height))

    def preprocess_image(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 1)
        edged = cv.Canny(blur, 75, 100)
        return gray, blur, edged

    def get_contours(self, edged, gray):
        edge_copy = edged.copy()
        contours, _ = cv.findContours(edge_copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        image_color = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        cv.drawContours(image_color, contours, -1, (0, 255, 0), 10)
        return contours, image_color

    def process_paper(self, blur, gray, biggest):
        wrapped_paper_image = wrapped_paper(self.width, self.height, biggest, blur)
        wrapped_paper_gray = wrapped_paper(self.width, self.height, biggest, gray)
        return wrapped_paper_image, wrapped_paper_gray

    def apply_threshold(self, wrapped_paper_image):
        return cv.adaptiveThreshold(
            wrapped_paper_image, 255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY, 11, 2
        )

    def remove_noise(self, thresh_image):
        contours, _ = cv.findContours(thresh_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        output = thresh_image.copy()
        replace_image_with_white(contours, output, 30)
        return output

    def grade_mcq(self, mcq_regions):
        results = []
        for q_num, question in enumerate(mcq_regions, 1):
            answers = split_questions(question, (7, 3), 2)
            for answer in answers:
                results.append(np.argmin(answer))
        return results

    def process_id(self, id_image):
        id_answers = correct_id_mcq(id_image)
        return ''.join(str(np.argmin(ans)) for ans in id_answers)


def grade_students(model_answers_file, students_results):
    # Read model answers
    with open(model_answers_file, 'r') as f:
        model_answers = [line.strip() for line in f.readlines()]

    model_answers = [ord(ans.lower()) - ord('a') for ans in model_answers]
    # Prepare results for Excel
    results_data = []

    for student_id, student_answers in students_results:
        graded_answers = [
            1 if student_answers[i] == model_answers[i] else 0
            for i in range(len(model_answers))
        ]
        results_data.append([student_id] + graded_answers)

    # Prepare DataFrame
    column_names = ["Student ID"] + [f"Q{i+1}" for i in range(len(model_answers))]
    df = pd.DataFrame(results_data, columns=column_names)

    # Write to Excel
    output_excel_file = 'grades.xlsx'  # Fixed output file name
    df.to_excel(output_excel_file, index=False)
    return output_excel_file


def run_gui():
    def select_image():
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
        if path:
            image_path_label["text"] = path

    def select_model_answers():
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if path:
            model_answers_path_label["text"] = path

    def start_grading():
        image_path = image_path_label["text"]
        model_answers_path = model_answers_path_label["text"]

        if not image_path or not model_answers_path:
            messagebox.showerror("Error", "Please select all required files.")
            return

        try:
            grader = MCQGrader()

            # Read and process the image
            image = grader.read_image(image_path)
            gray, blur, edged = grader.preprocess_image(image)
            contours, _ = grader.get_contours(edged, gray)

            # Extract the paper
            biggest, _ = find_biggest_contour(contours)
            wrapped_paper_image, wrapped_paper_gray = grader.process_paper(blur, gray, biggest)

            # Process the image
            thresh = grader.apply_threshold(wrapped_paper_image)
            output = grader.remove_noise(thresh)

            # Extract regions
            actual_box_addative, actual_box_gray = cropp_box_image(output, grader.width, grader.height, wrapped_paper_gray)
            mcq_regions, id_image, _ = finall_extract(actual_box_addative, actual_box_gray)

            # Grade
            student_id = grader.process_id(id_image)
            mcq_answers = grader.grade_mcq(mcq_regions)

            # Collect results
            students_results = [(student_id, mcq_answers)]

            # Save grades to Excel
            output_file_path = grade_students(model_answers_path, students_results)
            messagebox.showinfo("Success", f"Grades saved to {output_file_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    root = Tk()
    root.title("MCQ Grading System")
    root.geometry("600x400")

    # Styling
    title_font = Font(family="Helvetica", size=16, weight="bold")
    label_font = Font(family="Helvetica", size=12)

    # Title
    title_label = Label(root, text="MCQ Grading System", font=title_font, pady=10)
    title_label.pack()

    # Image selection
    frame1 = Frame(root, pady=10)
    frame1.pack(fill="x")

    image_label = Label(frame1, text="Select Bubble Sheet Image:", font=label_font)
    image_label.pack(side="left", padx=10)

    image_button = Button(frame1, text="Browse", command=select_image)
    image_button.pack(side="left", padx=10)

    image_path_label = Label(frame1, text="", font=label_font, wraplength=300, anchor="w", justify="left")
    image_path_label.pack(side="left", padx=10)

    # Model answers selection
    frame2 = Frame(root, pady=10)
    frame2.pack(fill="x")

    model_answers_label = Label(frame2, text="Select Model Answers File:", font=label_font)
    model_answers_label.pack(side="left", padx=10)

    model_answers_button = Button(frame2, text="Browse", command=select_model_answers)
    model_answers_button.pack(side="left", padx=10)

    model_answers_path_label = Label(frame2, text="", font=label_font, wraplength=300, anchor="w", justify="left")
    model_answers_path_label.pack(side="left", padx=10)

    # Start Grading button
    start_button = Button(root, text="Start Grading", command=start_grading, font=label_font, pady=10)
    start_button.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    run_gui()
