IMAGE_ENHANCEMENT = {
    'brightness': 1.2,
    'contrast': 1.2,
    'sharpness': 1.1,
}

# InsightFace uses cosine similarity (higher = better match)
# Threshold 0.35 means faces with similarity > 0.35 are considered matches
# This is different from old dlib Euclidean distance (lower = better)
FACE_RECOGNITION = {
    'model': 'insightface',  # Using InsightFace with GPU
    'threshold': 0.35,       # Cosine similarity threshold (0-1 scale, higher = stricter)
    'resize_scale': 0.25,    # For display purposes
    'confirmation_margin': 0.08,  # Faces between threshold-margin and threshold need confirmation
    'det_size': (640, 640),  # Detection input size
    'det_threshold': 0.5,    # Face detection confidence threshold
}

# All paths relative to project root
PATHS = {
    'student_csv': 'data/Student.csv',
    'encodings_pkl': 'data/student_encodings.pkl',
    'database': 'data/attendance.db',
    'images_dir': 'images',
    'assets_dir': 'assets',
    'models_dir': '~/.insightface/models/buffalo_l',  # InsightFace models directory
}

# Legacy alias for backward compatibility
DATABASE = {
    'db_file': PATHS['database']
}
