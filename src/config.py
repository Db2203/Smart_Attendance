IMAGE_ENHANCEMENT = {
    'brightness': 1.2,
    'contrast': 1.2,
    'sharpness': 1.1,
}

FACE_RECOGNITION = {
    'model': 'cnn',
    'threshold': 0.5,
    'resize_scale': 0.25,
    'confirmation_margin': 0.1
}

# All paths relative to project root
PATHS = {
    'student_csv': 'data/Student.csv',
    'encodings_pkl': 'data/student_encodings.pkl',
    'database': 'data/attendance.db',
    'images_dir': 'images',
    'assets_dir': 'assets',
}

# Legacy alias for backward compatibility
DATABASE = {
    'db_file': PATHS['database']
}
