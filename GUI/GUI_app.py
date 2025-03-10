import tkinter as tk
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import pandas as pd
import datetime as dt
import face_recognition as fr
import subprocess

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

        # Log frame with scrollbar
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

        self.pr_df = pd.DataFrame(columns=['Reg No', 'Name'])
        self.abs_df = pd.DataFrame(columns=['Reg No', 'Name'])

    def update_log(self, text):
        """ Updates the log area and auto-scrolls """
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.yview_moveto(1)  # Auto-scroll to bottom

    def get_image_path(self):
        """ Use Zenity to select an image instead of Tkinter file dialog """
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
            self.update_log("[Log 0] No input file: Code Terminated")
            return

        self.update_log(f"Selected file: {unknown_fp}")

        try:
            df = pd.read_csv('Student.csv', delimiter=',')
        except Exception as e:
            self.update_log(f"[Error] Unable to read Student.csv: {e}")
            return

        presentees = []
        absentees = []

        dateStr = str(dt.datetime.now().date())

        self.pfname = f'Presentees {dateStr}.csv'
        self.afname = f'Absentees {dateStr}.csv'

        self.update_log('[Log 0] Verifying')

        try:
            unknownImage = fr.load_image_file(unknown_fp)
            unknownEncoding = fr.face_encodings(unknownImage)
        except Exception as e:
            self.update_log(f"[Error] Failed to process image: {e}")
            return

        if not unknownEncoding:
            self.update_log("[Error] No faces detected in the image.")
            return

        peopleCount = list(range(len(unknownEncoding)))

        for index, row in df.iterrows():
            if index == 4:
                self.update_log("[Log 1] Half Done Successfully")

            StudPath = row['File Path']
            try:
                knownImage = fr.load_image_file(StudPath)
                knownEncoding = fr.face_encodings(knownImage)[0]
            except Exception as e:
                self.update_log(f"[Warning] Could not process {StudPath}: {e}")
                continue

            for i in peopleCount:
                results = fr.compare_faces([knownEncoding], unknownEncoding[i])

                if results[0]:
                    peopleCount.remove(i)
                    presentees.append([row['Reg No'], row['Name']])
                    break

            if [row['Reg No'], row['Name']] not in presentees:
                absentees.append([row['Reg No'], row['Name']])

        self.pr_df = pd.DataFrame(presentees, columns=['Reg No', 'Name'])
        self.abs_df = pd.DataFrame(absentees, columns=['Reg No', 'Name'])

        self.update_log('[Log 2] Created new data frames')
        self.update_log('[Log 3] Updated the data frames')

    def download_file(self):
        if self.pr_df.empty and self.abs_df.empty:
            self.update_log("[Error] No data to save. Process an image first.")
            return

        presentees_file = fd.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if presentees_file:
            self.pr_df.to_csv(presentees_file, sep=',', index=False, encoding='utf-8')
            self.update_log(f'[Log 4] Presentees saved to: {presentees_file}')

        absentees_file = fd.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if absentees_file:
            self.abs_df.to_csv(absentees_file, sep=',', index=False, encoding='utf-8')
            self.update_log(f'[Log 5] Absentees saved to: {absentees_file}')

def main():
    root = tk.Tk()
    app = SmartAttendanceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
