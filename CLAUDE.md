# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
python src/main.py
```

Run from the project root directory.

## Dependencies

```bash
pip install face-recognition pandas pillow numpy
```

Note: `tkinter` is included with Python on Windows. On Linux, install `python3-tk`.

The `face_recognition` library requires dlib, which may need CMake and a C++ compiler on some systems.

## Project Structure

```
Smart_Attendance/
├── src/                    # Python source code
│   ├── main.py             # Entry point
│   ├── face_recognition_module.py
│   ├── config.py           # Centralized configuration
│   ├── attendance_db.py    # SQLite helper
│   └── GUI_app.py          # Legacy GUI version
├── assets/                 # GUI images (logo, buttons)
├── data/                   # Data files
│   ├── Student.csv         # Student registry
│   ├── student_encodings.pkl
│   └── attendance.db
├── images/                 # Student face photos (gitignored)
└── reference_images/       # Test group photos (gitignored)
```

## Architecture

### Core Flow

1. **Startup**: `main.py` loads precomputed face encodings from `data/student_encodings.pkl` or generates them from `data/Student.csv`
2. **Recognition**: User selects an image → faces are detected and encoded → compared against stored encodings → matches marked present
3. **Close Match Handling**: Faces within the confirmation margin (threshold + 0.1) trigger a manual confirmation dialog; confirmed faces update the encodings
4. **Output**: Attendance exported as CSV files (presentees/absentees)

### Key Modules

- **`src/main.py`**: Primary entry point. Contains `SmartAttendanceApp` class with full GUI, image processing flow, student management, and encoding persistence
- **`src/face_recognition_module.py`**: Face detection/encoding logic. Separate from GUI for modularity
- **`src/config.py`**: Configuration constants for paths, recognition thresholds, and image enhancement parameters
- **`src/attendance_db.py`**: SQLite helper for attendance records

### Configuration (`src/config.py`)

- `PATHS`: Centralized file paths for all data files
- `FACE_RECOGNITION['threshold']`: 0.5 - distance below this is a match
- `FACE_RECOGNITION['confirmation_margin']`: 0.1 - faces between threshold and threshold+margin trigger confirmation
- `FACE_RECOGNITION['model']`: 'cnn' - face detection model (cnn is more accurate, hog is faster)
- `IMAGE_ENHANCEMENT`: Brightness/contrast/sharpness multipliers applied before face detection

### Important Patterns

- Encodings are persisted to pkl after each confirmed close match to improve future recognition
- Rejected face encodings are tracked in `self.rejections` to avoid re-prompting for the same face
- Image preprocessing (brightness, contrast, sharpness enhancement) is applied to both reference and input images
- All paths use relative paths from project root via `PATHS` config
