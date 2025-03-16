# config.py

# Settings for image enhancement
IMAGE_ENHANCEMENT = {
    'brightness': 1.2,
    'contrast': 1.2,
    'sharpness': 1.1,
}

# Settings for face recognition
FACE_RECOGNITION = {
    'model': 'cnn',          # Options: 'cnn' or 'hog'
    'threshold': 0.5,        # Lower values require a closer match
    'resize_scale': 0.25     # Factor to downscale images for faster processing
}

# Settings for the SQLite database
DATABASE = {
    'db_file': 'attendance.db'
}
