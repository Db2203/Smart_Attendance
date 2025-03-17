import tkinter as tk
from tkinter import filedialog as fd, messagebox
from tkinter import Toplevel, Label, Button, Frame
from PIL import Image, ImageTk
import pandas as pd
import os, subprocess, sys, pickle, numpy as np, face_recognition as fr

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from face_recognition_module import load_student_encodings, recognize_faces_in_image, preprocess_image
from my_config import FACE_RECOGNITION


class SmartAttendanceApp:

    def __init__(self, master):

        self.master = master
        self.master.title("Smart Attendance System")
        self.master.configure(bg="#121212")
        self.master.geometry("1000x800")

        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(0, weight=1)


        self.header_frame = tk.Frame(master, bg="#121212")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.header_label = tk.Label(self.header_frame,
                                     text="Smart Attendance System",
                                     font=("Helvetica", 28, "bold"),
                                     bg="#121212", fg="white")
        self.header_label.grid(row=0, column=0, sticky="nsew")

        self.refresh_icon = tk.Button(self.header_frame,
                                      text="⟳",
                                      font=("Helvetica", 12),
                                      bg="#121212", fg="#00FF00",
                                      bd=0, activebackground="#121212",
                                      command=self.refresh_encodings)
        self.refresh_icon.grid(row=0, column=1, sticky="ne", padx=10)


        self.log_frame = tk.Frame(master, bg="#121212")
        self.log_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.log_frame.grid_rowconfigure(0, weight=1)
        self.log_frame.grid_columnconfigure(0, weight=1)

        self.log_text = tk.Text(self.log_frame,
                                font=("Helvetica", 12),
                                bg="#121212", fg="#32CD32",
                                wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = tk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=self.scrollbar.set)


        self.button_frame = tk.Frame(master, bg="#121212")
        self.button_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        self.button_frame.grid_columnconfigure(2, weight=1)

        self.choose_button = tk.Button(self.button_frame,
                                       text="Choose Image!",
                                       font=("Helvetica", 16, "bold"),
                                       bg="black", fg="#00FF00",
                                       command=self.process_image,
                                       width=20)
        self.choose_button.grid(row=0, column=0, padx=10, pady=10)

        self.add_button = tk.Button(self.button_frame,
                                    text="Add New Student",
                                    font=("Helvetica", 16, "bold"),
                                    bg="black", fg="#00FF00",
                                    command=self.add_student,
                                    width=20)
        self.add_button.grid(row=0, column=1, padx=10, pady=10)

        self.download_button = tk.Button(self.button_frame,
                                         text="Download Attendance Records",
                                         font=("Helvetica", 16, "bold"),
                                         bg="black", fg="#00FF00",
                                         command=self.download_file,
                                         width=25)
        self.download_button.grid(row=0, column=2, padx=10, pady=10)


        self.pr_df = pd.DataFrame(columns=["Reg No", "Name"])
        self.abs_df = pd.DataFrame(columns=["Reg No", "Name"])

        self.student_encodings, self.reg_no_to_name = load_student_encodings()
        self.rejections = {}


    def update_log(self, text):

        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        print(text)


    def get_image_path(self):

        if os.name == "nt":
            return fd.askopenfilename(title="Select an image",
                                      filetypes=[("JPEG files", "*.jpg *.jpeg"),
                                                 ("PNG files", "*.png")])
        else:
            try:
                result = subprocess.run(["zenity", "--file-selection", "--file-filter=*.jpg *.jpeg *.png"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception as e:
                self.update_log(f"[Error] Zenity failed: {e}")
            return fd.askopenfilename(title="Select an image",
                                      filetypes=[("JPEG files", "*.jpg *.jpeg"),
                                                 ("PNG files", "*.png")])


    def ask_user_confirmation(self, cropped_face, prompt):

        dialog = Toplevel(self.master)
        dialog.title("Confirm Identity")
        dialog.configure(bg="#121212")

        photo = ImageTk.PhotoImage(cropped_face)
        dialog.photo = photo

        img_label = Label(dialog, image=photo, bg="#121212")
        img_label.pack(pady=10)

        prompt_label = Label(dialog, text=prompt,
                             font=("Helvetica", 14),
                             bg="#121212", fg="white")
        prompt_label.pack(pady=10)

        answer = {"result": None}

        def yes():
            answer["result"] = True
            dialog.destroy()

        def no():
            answer["result"] = False
            dialog.destroy()

        btn_frame = Frame(dialog, bg="#121212")
        btn_frame.pack(pady=10)

        yes_button = Button(btn_frame, text="Yes",
                            font=("Helvetica", 12, "bold"),
                            bg="black", fg="#00FF00",
                            command=yes, width=10)
        yes_button.pack(side="left", padx=10)

        no_button = Button(btn_frame, text="No",
                           font=("Helvetica", 12, "bold"),
                           bg="black", fg="#00FF00",
                           command=no, width=10)
        no_button.pack(side="left", padx=10)

        dialog.grab_set()
        self.master.wait_window(dialog)

        return answer["result"]


    def process_image(self):

        image_path = self.get_image_path()
        if not image_path:
            self.update_log("[Log] No input file selected.")
            return

        self.update_log(f"Selected file: {image_path}")

        recognized_reg_nos, logs, close_match_candidates, unknown_image = recognize_faces_in_image(
            image_path, self.student_encodings, self.reg_no_to_name)

        for log in logs:
            self.update_log(log)

        pil_unknown_image = Image.fromarray(unknown_image) if unknown_image is not None else None
        rejection_similarity_threshold = 0.05

        for candidate in close_match_candidates:
            unknown_encoding, candidate_reg_no, best_distance, face_location = candidate
            student_name = self.reg_no_to_name.get(candidate_reg_no, "Unknown")
            rejected = False

            if candidate_reg_no in self.rejections:
                for rejected_enc in self.rejections[candidate_reg_no]:
                    dist = fr.face_distance([rejected_enc], unknown_encoding)[0]
                    if dist < rejection_similarity_threshold:
                        self.update_log(f"[Log] Candidate for {student_name} previously rejected; skipping confirmation.")
                        rejected = True
                        break

            if rejected:
                continue

            if face_location and pil_unknown_image:
                top, right, bottom, left = face_location
                crop_box = (left, top, right, bottom)
                cropped_face = pil_unknown_image.crop(crop_box)
                prompt = f"A close match was found for {student_name} (distance: {best_distance:.3f}). Is this {student_name}?"
                answer = self.ask_user_confirmation(cropped_face, prompt)

                if answer:
                    self.student_encodings[candidate_reg_no].append(unknown_encoding)
                    recognized_reg_nos.add(candidate_reg_no)
                    self.update_log(f"[Log] {student_name} confirmed and encoding updated.")
                else:
                    if candidate_reg_no not in self.rejections:
                        self.rejections[candidate_reg_no] = []
                    self.rejections[candidate_reg_no].append(unknown_encoding)
                    self.update_log(f"[Log] {student_name} not confirmed; candidate rejected.")
            else:
                self.update_log(f"[Log] Unable to retrieve face location for {student_name} candidate.")

        if not recognized_reg_nos:
            self.update_log("[Log] No recognized faces. Marking all as absent.")
            self.create_absentees_from_all()
            return

        with open("student_encodings.pkl", "wb") as f:
            pickle.dump(self.student_encodings, f)

        presentees = []
        for reg_no in recognized_reg_nos:
            name = self.reg_no_to_name.get(reg_no, "Unknown")
            presentees.append([reg_no, name])

        absentees = []
        for reg_no, name in self.reg_no_to_name.items():
            if reg_no not in recognized_reg_nos:
                absentees.append([reg_no, name])

        self.pr_df = pd.DataFrame(presentees, columns=["Reg No", "Name"])
        self.abs_df = pd.DataFrame(absentees, columns=["Reg No", "Name"])

        self.update_log("[Log] Attendance updated successfully.")
        self.update_log(f"Total Present: {len(presentees)} | Total Absent: {len(absentees)}")


    def create_absentees_from_all(self):

        absentees = []
        for reg_no, name in self.reg_no_to_name.items():
            absentees.append([reg_no, name])
        self.abs_df = pd.DataFrame(absentees, columns=["Reg No", "Name"])
        self.update_log("[Log] All students marked as absent.")


    def download_file(self):

        if self.pr_df.empty and self.abs_df.empty:
            self.update_log("[Error] No data to save. Process an image first.")
            return

        presentees_file = fd.asksaveasfilename(defaultextension=".csv",
                                               filetypes=[("CSV files", "*.csv")])
        if presentees_file:
            self.pr_df.to_csv(presentees_file, sep=",", index=False, encoding="utf-8")
            self.update_log(f"[Log] Presentees saved to: {presentees_file}")

        absentees_file = fd.asksaveasfilename(defaultextension=".csv",
                                              filetypes=[("CSV files", "*.csv")])
        if absentees_file:
            self.abs_df.to_csv(absentees_file, sep=",", index=False, encoding="utf-8")
            self.update_log(f"[Log] Absentees saved to: {absentees_file}")


    def refresh_encodings(self):

        self.student_encodings, self.reg_no_to_name = load_student_encodings(force_refresh=True)
        self.update_log("[Log] Student encodings refreshed from CSV.")


    def add_student(self):

        add_window = Toplevel(self.master)
        add_window.title("Add New Student")
        add_window.geometry("400x300")
        add_window.configure(bg="#121212")

        Label(add_window, text="Reg No:", font=("Comic Sans", 12),
              bg="#121212", fg="white").pack(pady=5)
        reg_no_entry = tk.Entry(add_window, font=("Comic Sans", 12))
        reg_no_entry.pack(pady=5)

        Label(add_window, text="Name:", font=("Comic Sans", 12),
              bg="#121212", fg="white").pack(pady=5)
        name_entry = tk.Entry(add_window, font=("Comic Sans", 12))
        name_entry.pack(pady=5)

        image_path_var = tk.StringVar()

        def browse_image():
            path = ""
            if os.name == "nt":
                path = fd.askopenfilename(title="Select an image",
                                          filetypes=[("JPEG files", "*.jpg *.jpeg"),
                                                     ("PNG files", "*.png")])
            else:
                try:
                    result = subprocess.run(["zenity", "--file-selection", "--file-filter=*.jpg *.jpeg *.png"],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if result.returncode == 0:
                        path = result.stdout.strip()
                    else:
                        path = fd.askopenfilename(title="Select an image",
                                                  filetypes=[("JPEG files", "*.jpg *.jpeg"),
                                                             ("PNG files", "*.png")])
                except Exception as e:
                    self.update_log(f"[Error] Zenity failed: {e}")
                    path = fd.askopenfilename(title="Select an image",
                                              filetypes=[("JPEG files", "*.jpg *.jpeg"),
                                                         ("PNG files", "*.png")])
            image_path_var.set(path)
            if path:
                Label(add_window, text=f"Selected: {os.path.basename(path)}",
                      font=("Comic Sans", 10), bg="#121212", fg="white").pack()

        Button(add_window, text="Browse Image", font=("Comic Sans", 12),
               fg="#00FF00", bg="black", command=browse_image).pack(pady=10)

        def submit_student():
            reg_no = reg_no_entry.get().strip()
            name = name_entry.get().strip()
            image_path = image_path_var.get().strip()
            if not reg_no or not name or not image_path:
                messagebox.showerror("Error", "All fields are required!")
                return
            try:
                image = fr.load_image_file(image_path)
                image = preprocess_image(image)
                encodings = fr.face_encodings(image, model=FACE_RECOGNITION["model"])
                if not encodings:
                    messagebox.showerror("Error", "No face detected in the image.")
                    return
                encoding = encodings[0]
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {e}")
                return
            csv_file = "Student.csv"
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file, dtype=str)
                df["Reg No"] = df["Reg No"].str.strip()
                df["File Paths"] = df["File Paths"].fillna("")
            else:
                df = pd.DataFrame(columns=["Reg No", "Name", "File Paths"])
            existing_reg_nos = [x.strip() for x in df["Reg No"].tolist()]
            if reg_no in existing_reg_nos:
                idx = existing_reg_nos.index(reg_no)
                existing_paths = df.at[idx, "File Paths"]
                paths = [p.strip() for p in existing_paths.split(",") if p.strip()] if existing_paths else []
                if image_path not in paths:
                    paths.append(image_path)
                df.at[idx, "File Paths"] = ",".join(paths)
                self.update_log(f"[Log] Updated existing student record for reg no {reg_no}.")
            else:
                new_row = {"Reg No": reg_no, "Name": name, "File Paths": image_path}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                self.update_log(f"[Log] Added new student record for reg no {reg_no}.")
            try:
                df.to_csv(csv_file, index=False, encoding="utf-8")
                self.update_log("[Log] CSV file updated successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update CSV: {e}")
                return
            df_reload = pd.read_csv(csv_file, dtype=str)
            df_reload["Reg No"] = df_reload["Reg No"].str.strip()
            self.reg_no_to_name = dict(zip(df_reload["Reg No"], df_reload["Name"]))
            if reg_no in self.student_encodings:
                self.student_encodings[reg_no].append(encoding)
            else:
                self.student_encodings[reg_no] = [encoding]
            try:
                with open("student_encodings.pkl", "wb") as f:
                    pickle.dump(self.student_encodings, f)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update encodings file: {e}")
                return
            messagebox.showinfo("Success", "New student added/updated successfully!")
            add_window.destroy()

        Button(add_window, text="Submit", font=("Comic Sans", 12),
               fg="#00FF00", bg="black", command=submit_student).pack(pady=10)


def main():
    root = tk.Tk()
    app = SmartAttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
