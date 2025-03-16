# main.py



import tkinter as tk
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import pandas as pd
import os
import subprocess
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from face_recognition_module import load_student_encodings, recognize_faces_in_image
#import attendance_db

class SmartAttendanceApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Smart Attendance System")
        self.master.geometry("900x900")
        self.master.config(background="#121212")

        try:
            self.master.iconphoto(True, tk.PhotoImage(file='GUI/logo.png'))
        except Exception as e:
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

        self.button = tk.Button(
            master,
            text='Choose Image!',
            font=('Comic Sans', 18, 'bold'),
            fg='#00FF00',
            bg='black',
            activeforeground='black',
            activebackground='white',
            command=self.process_image
        )
        self.button.place(x=330, y=600)

        self.download = tk.Button(
            master,
            text='Download Attendance Records',
            font=('Comic Sans', 18, 'bold'),
            fg='#00FF00',
            bg='black',
            activeforeground='black',
            activebackground='white',
            command=self.download_file
        )
        self.download.place(x=200, y=700)

        self.pr_df = pd.DataFrame(columns=['Reg No', 'Name'])
        self.abs_df = pd.DataFrame(columns=['Reg No', 'Name'])

        # Initialize the attendance database
        #attendance_db.init_db()

        # Load student encodings and mapping from registration number to name
        self.student_encodings, self.reg_no_to_name = load_student_encodings()

    def update_log(self, text):
        """Append log messages to the text widget and print to console."""
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.yview_moveto(1)
        print(text)

    def get_image_path(self):
        """Open a file dialog to select an image file."""
        if os.name == 'nt':
            return fd.askopenfilename(
                title="Select an image",
                filetypes=[("JPEG files", "*.jpg *.jpeg"), ("PNG files", "*.png")]
            )
        else:
            try:
                result = subprocess.run(
                    ["zenity", "--file-selection", "--file-filter=*.jpg *.jpeg"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception as e:
                self.update_log(f"[Error] Zenity failed: {e}")
            return fd.askopenfilename(
                title="Select an image",
                filetypes=[("JPEG files", "*.jpg *.jpeg"), ("PNG files", "*.png")]
            )

    def process_image(self):
        """Process the selected image, recognize faces, and update attendance."""
        image_path = self.get_image_path()
        if not image_path:
            self.update_log("[Log] No input file selected.")
            return

        self.update_log(f"Selected file: {image_path}")
        recognized_reg_nos, logs = recognize_faces_in_image(image_path, self.student_encodings, self.reg_no_to_name)
        for log in logs:
            self.update_log(log)

        if not recognized_reg_nos:
            self.update_log("[Log] No recognized faces. Marking all as absent.")
            self.create_absentees_from_all()
            return

        # Build lists for presentees and absentees
        presentees = []
        for reg_no in recognized_reg_nos:
            name = self.reg_no_to_name.get(reg_no, "Unknown")
            presentees.append([reg_no, name])
            # Insert attendance record into the database
            # from attendance_db import insert_attendance
            # insert_attendance(reg_no, name)

        absentees = []
        for reg_no, name in self.reg_no_to_name.items():
            if reg_no not in recognized_reg_nos:
                absentees.append([reg_no, name])

        self.pr_df = pd.DataFrame(presentees, columns=['Reg No', 'Name'])
        self.abs_df = pd.DataFrame(absentees, columns=['Reg No', 'Name'])

        self.update_log("[Log] Attendance updated successfully.")
        self.update_log(f"Total Present: {len(presentees)} | Total Absent: {len(absentees)}")

    def create_absentees_from_all(self):
        """Mark all students as absent."""
        absentees = []
        for reg_no, name in self.reg_no_to_name.items():
            absentees.append([reg_no, name])
        self.abs_df = pd.DataFrame(absentees, columns=['Reg No', 'Name'])
        self.update_log("[Log] All students marked as absent.")

    def download_file(self):
        """Save the attendance records to CSV files."""
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
