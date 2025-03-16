import tkinter as tk
from tkinter import filedialog as fd
from PIL import Image, ImageTk, ImageEnhance
import pandas as pd
import datetime as dt
import face_recognition as fr
import subprocess
import pickle
import concurrent.futures
import os
import numpy as np

def preprocess_image(image):
    """
    Enhance the input image for improved face detection.
    Uses brightness, contrast, and sharpness enhancements.
    """
    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(image)
    # Enhance brightness
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.2)  # Increase brightness by 20%
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.2)  # Increase contrast by 20%
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.1)  # Slightly sharpen the image
    # Optionally, you could resize the image if it's too high-res
    return np.array(pil_img)

class SmartAttendanceApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Smart Attendance System")
        self.master.geometry("900x900")
        self.master.config(background="#121212")

        try:
            self.master.iconphoto(True, tk.PhotoImage(file='GUI/logo.png'))
        except:
            print("[Warning] Icon file not found.")

        self.label = tk.Label(master, text="Smart Attendance System", font=('Elianto', 25, 'bold'), fg='white', bg='#121212')
        self.label.place(x=250, y=50)

        log_frame = tk.Frame(master, bg="#121212")
        log_frame.place(x=50, y=150, width=800, height=400)

        self.log_text = tk.Text(log_frame, font=('Elianto', 12), fg='#32CD32', bg='#121212', wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        try:
            self.button_image = tk.PhotoImage(file='GUI/click.png')
            self.download_image = tk.PhotoImage(file='GUI/download.png')
        except:
            print("[Warning] GUI button images not found. Using text instead.")
            self.button_image = None
            self.download_image = None

        self.button = tk.Button(master, text='Choose Image!', font=('Comic Sans', 18, 'bold'), fg='#00FF00', bg='black',
                                activeforeground='black', activebackground='white', image=self.button_image, compound='right',
                                command=self.process_image)
        self.button.place(x=330, y=600)

        self.download = tk.Button(master, text='Download Presentees and Absentees file!', font=('Comic Sans', 18, 'bold'),
                                  fg='#00FF00', bg='black', activeforeground='black', activebackground='white',
                                  image=self.download_image, compound='right', command=self.download_file)
        self.download.place(x=200, y=700)

        # DataFrames for final attendance results
        self.pr_df = pd.DataFrame(columns=['Reg No', 'Name'])
        self.abs_df = pd.DataFrame(columns=['Reg No', 'Name'])

        # Load or create student encodings and name mapping
        self.student_encodings = self.load_student_encodings()

    def update_log(self, text):
        """Update the log area and auto-scroll."""
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.yview_moveto(1)

    def get_image_path(self):
        """Use Zenity to select an image file."""
        try:
            result = subprocess.run(["zenity", "--file-selection", "--file-filter=*.jpg *.jpeg"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return None
        except Exception as e:
            self.update_log(f"[Error] Zenity failed: {e}")
            return None

    def process_image(self):
        unknown_fp = self.get_image_path()
        if not unknown_fp:
            self.update_log("[Log] No input file: Code Terminated")
            return

        self.update_log(f"Selected file: {unknown_fp}")

        try:
            unknown_image = fr.load_image_file(unknown_fp)
            # Preprocess the image for better detection
            unknown_image = preprocess_image(unknown_image)
            unknown_encodings = fr.face_encodings(unknown_image, model="cnn")
        except Exception as e:
            self.update_log(f"[Error] Failed to process image: {e}")
            return

        if not unknown_encodings:
            self.update_log("[Error] No faces detected in the image.")
            self.create_absentees_from_all()
            return

        recognized_reg_nos = set()

        # For each detected face, use a best-match approach
        for unknown_encoding in unknown_encodings:
            if not isinstance(unknown_encoding, np.ndarray):
                unknown_encoding = np.array(unknown_encoding)
            if unknown_encoding.ndim != 1:
                unknown_encoding = unknown_encoding.flatten()

            best_match = None
            best_distance = 1.0  # Start with a high distance
            # Compare this face against every student's known encodings
            for reg_no, known_encodings in self.student_encodings.items():
                distances = []
                for known_encoding in known_encodings:
                    if not isinstance(known_encoding, np.ndarray):
                        known_encoding = np.array(known_encoding)
                    if known_encoding.ndim != 1:
                        known_encoding = known_encoding.flatten()
                    d = fr.face_distance([known_encoding], unknown_encoding)[0]
                    distances.append(d)
                if distances:
                    student_min_distance = min(distances)
                    if student_min_distance < best_distance:
                        best_distance = student_min_distance
                        best_match = reg_no

            # If the best match is within the acceptable threshold, record it
            if best_distance < 0.5:
                recognized_reg_nos.add(best_match)
            else:
                self.update_log(f"Unknown face with distance: {best_distance:.3f}")

        # Build presentees list from recognized registration numbers
        presentees = []
        for reg_no in recognized_reg_nos:
            name = self.reg_no_to_name.get(reg_no, "Unknown")
            presentees.append([reg_no, name])

        # Build absentees list from all students not recognized
        absentees = []
        for reg_no, name in self.reg_no_to_name.items():
            if reg_no not in recognized_reg_nos:
                absentees.append([reg_no, name])

        self.pr_df = pd.DataFrame(presentees, columns=['Reg No', 'Name'])
        self.abs_df = pd.DataFrame(absentees, columns=['Reg No', 'Name'])

        self.update_log("[Log] Attendance updated successfully.")
        self.update_log(f"Total Present: {len(presentees)} | Total Absent: {len(absentees)}")

    def create_absentees_from_all(self):
        """Mark all students as absent if no faces are detected."""
        absentees = []
        for reg_no, name in self.reg_no_to_name.items():
            absentees.append([reg_no, name])
        self.abs_df = pd.DataFrame(absentees, columns=['Reg No', 'Name'])

    def load_student_encodings(self):
        """
        Load precomputed student encodings or create them if not found.
        Also builds a dictionary mapping Reg No -> Name.
        """
        df = pd.read_csv('Student.csv')
        self.reg_no_to_name = dict(zip(df['Reg No'], df['Name']))

        if os.path.exists("student_encodings.pkl"):
            with open("student_encodings.pkl", "rb") as f:
                return pickle.load(f)
        else:
            self.update_log("[Warning] No precomputed encodings found. Creating them now...")
            return self.precompute_student_encodings(df)

    def precompute_student_encodings(self, df):
        """
        Compute and save encodings for all students in 'Student.csv'.
        Applies preprocessing to each student image.
        """
        encodings_dict = {}

        def encode_face(image_path):
            try:
                image = fr.load_image_file(image_path)
                # Preprocess each student image before encoding
                image = preprocess_image(image)
                encodings = fr.face_encodings(image, model="cnn")
                return encodings if encodings else []
            except Exception as e:
                self.update_log(f"[Warning] Error processing {image_path}: {e}")
                return []

        for _, row in df.iterrows():
            stud_paths = row['File Paths'].split(',')
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(encode_face, stud_paths))
            # Flatten the list of lists into a single list of encodings
            encodings_dict[row['Reg No']] = [enc for sublist in results for enc in sublist]

        with open("student_encodings.pkl", "wb") as f:
            pickle.dump(encodings_dict, f)

        self.update_log("[Log] Student encodings precomputed and saved.")
        return encodings_dict

    def download_file(self):
        if self.pr_df.empty and self.abs_df.empty:
            self.update_log("[Error] No data to save. Process an image first.")
            return

        presentees_file = fd.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if presentees_file:
            self.pr_df.to_csv(presentees_file, sep=',', index=False, encoding='utf-8')
            self.update_log(f"[Log] Presentees saved to: {presentees_file}")

        absentees_file = fd.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if absentees_file:
            self.abs_df.to_csv(absentees_file, sep=',', index=False, encoding='utf-8')
            self.update_log(f"[Log] Absentees saved to: {absentees_file}")

def main():
    root = tk.Tk()
    app = SmartAttendanceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
