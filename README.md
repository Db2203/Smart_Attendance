# Smart Attendance System

An automated attendance solution that uses face recognition technology to identify students from images and manage attendance records efficiently.

## 🚀 Features
* **Face Recognition Attendance**: Automatically detects and identifies students in photos to mark them as present or absent.
* **Student Management**: Add new students directly through the GUI by providing their Registration Number, Name, and a reference image.
* **Identity Confirmation**: Includes a manual confirmation step for "close matches" to ensure high accuracy in identification.
* **Export Records**: Generate and download attendance reports (both Presentees and Absentees) in CSV format.
* **Image Preprocessing**: Automatically enhances input images (brightness, contrast, and sharpness) to improve recognition reliability.
* **Efficient Processing**: Uses precomputed face encodings stored in a `.pkl` file for faster recognition performance.

## 🛠️ Technologies Used
* **Language**: Python
* **Libraries**:
    * `face_recognition`: Core engine for detecting and identifying faces.
    * `tkinter`: Graphical user interface framework.
    * `pandas`: For managing student databases and exporting CSV records.
    * `PIL (Pillow)`: For image handling and enhancement.
    * `pickle`: For serializing and storing face encodings.
    * `NumPy`: For numerical operations on image data.

## 📂 Key Files
* `GUI/main.py`: The primary entry point for the application with the full GUI and student management tools.
* `GUI/face_recognition_module.py`: Contains the logic for face detection, encoding, and recognition.
* `GUI/my_config.py`: Configuration settings for recognition thresholds and image enhancement.
* `Student.csv`: Local database storing student registration numbers, names, and image paths.
* `student_encodings.pkl`: Serialized file containing processed face encodings for faster matching.

## ⚙️ Setup & Installation
1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd Smart_Attendance
    ```
2.  **Install dependencies**:
    ```bash
    pip install face-recognition pandas pillow numpy
    ```
    *Note: `tkinter` is usually included with Python, but you may need to install `python3-tk` on some Linux distributions.*
3.  **Run the Application**:
    ```bash
    python GUI/main.py
    ```

## 📖 How to Use
1.  **Initialize Encodings**: On the first run, the system will process images listed in `Student.csv` to create the `student_encodings.pkl` file.
2.  **Mark Attendance**: 
    * Click **"Choose Image!"** and select a photo of the class or individual.
    * The system will process the image and log recognized students in the UI.
    * If a face is a close match, a dialog will ask you to confirm the identity.
3.  **Add New Students**: Click **"Add New Student"** to register a new person with their details and a reference photo.
4.  **Download Logs**: Click **"Download Attendance Records"** to save CSV files of present and absent students.