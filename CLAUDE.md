# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

### Desktop GUI (Tkinter)
```bash
python src/main.py
```

### Web Application (Flask)
```bash
cd web
python run.py
```
Access at http://127.0.0.1:5000

Run from the project root directory.

## Dependencies

### Core (InsightFace - GPU Accelerated)
```bash
pip install insightface onnxruntime-gpu opencv-python numpy pillow pandas
```

### For CUDA GPU Support (Optional but recommended)
```bash
pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
```

### Web Application
```bash
pip install flask flask-login flask-wtf flask-sqlalchemy email-validator werkzeug
```

Note: InsightFace models (~500MB) are downloaded automatically on first run to `~/.insightface/models/buffalo_l/`

## Project Structure

```
Smart_Attendance/
├── src/                    # Desktop app source code
│   ├── main.py             # Desktop GUI entry point
│   ├── face_recognition_module.py  # InsightFace implementation
│   ├── config.py           # Centralized configuration
│   ├── attendance_db.py    # SQLite helper
│   └── GUI_app.py          # Legacy GUI version
├── web/                    # Flask web application
│   ├── run.py              # Web app entry point
│   ├── app/
│   │   ├── __init__.py     # App factory
│   │   ├── models.py       # SQLAlchemy models
│   │   ├── auth/           # Authentication blueprint
│   │   ├── teacher/        # Teacher routes & forms
│   │   ├── student/        # Student routes
│   │   ├── face_recognition/  # InsightFace service wrapper
│   │   ├── templates/      # Jinja2 templates
│   │   └── static/         # CSS, JS, images
│   └── instance/           # Database (gitignored)
├── assets/                 # Desktop GUI images
├── data/                   # Data files
│   ├── Student.csv         # Student registry
│   ├── student_encodings.pkl  # 512-dim InsightFace encodings
│   └── attendance.db
├── images/                 # Student face photos (gitignored)
├── regenerate_encodings.py # Script to regenerate all encodings
└── reference_images/       # Test group photos (gitignored)
```

## Architecture

### Face Recognition System (InsightFace)

**Technology Stack:**
- **Detection**: SCRFD model (det_10g.onnx) - GPU accelerated
- **Recognition**: ArcFace model (w600k_r50.onnx) - 512-dimensional embeddings
- **Similarity**: Cosine similarity (higher = better match)
- **Runtime**: ONNX Runtime with CUDA support (falls back to CPU)

**Key Differences from Old dlib System:**
| Aspect | Old (dlib) | New (InsightFace) |
|--------|------------|-------------------|
| Embedding size | 128-dim | 512-dim |
| Similarity metric | Euclidean distance (lower=better) | Cosine similarity (higher=better) |
| Threshold | 0.5 (distance) | 0.35 (similarity) |
| Speed | ~500ms/face (CPU) | ~20ms/face (GPU) |
| Accuracy | 99.38% | 99.6% |

### Configuration (`src/config.py`)

```python
FACE_RECOGNITION = {
    'model': 'insightface',      # Using InsightFace with GPU
    'threshold': 0.35,           # Cosine similarity threshold (0-1, higher=stricter)
    'confirmation_margin': 0.08, # Faces between 0.27-0.35 need manual confirmation
    'det_size': (640, 640),      # Detection input size
    'det_threshold': 0.5,        # Face detection confidence
}

IMAGE_ENHANCEMENT = {
    'brightness': 1.2,
    'contrast': 1.2,
    'sharpness': 1.1,
}
```

### Core Flow

1. **Startup**: Models loaded from `~/.insightface/models/buffalo_l/`
2. **Recognition**: Image uploaded → SCRFD detects faces → ArcFace generates 512-dim embeddings → Cosine similarity matching
3. **Close Match Handling**: Faces with similarity 0.27-0.35 trigger manual confirmation with checkboxes
4. **Attendance**: Recognized students marked present, enrolled students not in photo marked absent
5. **Multiple Sessions**: Each save creates new records (supports multiple classes per day)

### Key Modules

- **`web/app/face_recognition/service.py`**: InsightFace service with SCRFD detector and ArcFace recognizer
- **`src/face_recognition_module.py`**: Desktop app face recognition (same InsightFace implementation)
- **`src/config.py`**: Thresholds, paths, and enhancement settings
- **`regenerate_encodings.py`**: Utility to regenerate all student encodings

## Web Application

### Features
- **Authentication**: Login/register with role-based access (teacher/student)
- **Teacher Dashboard**: Mark attendance, view per-class statistics, manage students
- **Student Dashboard**: View attendance history and per-subject percentages
- **Face Recognition**: InsightFace with GPU acceleration
- **Close Match Confirmation**: Checkboxes to manually confirm borderline matches
- **Class Records**: View attendance by session with present/absent counts
- **Multiple Sessions**: Supports multiple attendance sessions per day per class

### Key Routes
- `/auth/login`, `/auth/register` - Authentication
- `/teacher/dashboard` - Teacher home with per-class stats
- `/teacher/attendance` - Upload photo & mark attendance
- `/teacher/class/<id>/records` - View class attendance history
- `/teacher/students` - CRUD student management
- `/student/dashboard` - Student attendance view

### Attendance Logic
```
1. Teacher uploads class photo
2. Face recognition identifies students:
   - High confidence (>35%) → Auto-recognized (green cards)
   - Medium confidence (27-35%) → Need confirmation (orange cards with checkboxes)
   - Low confidence (<27%) → Unknown
3. Teacher confirms close matches via checkboxes
4. On save:
   - Recognized + confirmed students → marked PRESENT
   - Enrolled students not in photo → marked ABSENT
5. Multiple saves per day create separate session records
```

### Database Models
- **User**: Authentication (email, password, role)
- **Student**: Profile (reg_no, name, face_encoding as pickled 512-dim list)
- **Class**: Subject (name, code, teacher_id)
- **ClassEnrollment**: Student-class relationship
- **Attendance**: Records (student_id, class_id, date, status, confidence) - NO unique constraint

### UI Theme - SLCM Clone
Styled to match Manipal University's SLCM portal with red header, green tabs, orange breadcrumbs.

### Templates
- `slcm_base.html` - Base template
- `teacher/slcm_dashboard.html` - Teacher dashboard with per-class stats
- `teacher/slcm_attendance.html` - Mark attendance with close match checkboxes
- `teacher/slcm_class_records.html` - Class attendance history
- `teacher/slcm_students.html` - Student management
- `student/slcm_dashboard.html` - Student attendance view
