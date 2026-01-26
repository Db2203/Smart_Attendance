import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(os.path.dirname(basedir), 'instance', 'attendance.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Session
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)

    # File uploads
    UPLOAD_FOLDER = os.path.join(os.path.dirname(basedir), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    # Face recognition paths (relative to project root)
    IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.dirname(basedir)), 'images')
    DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(basedir)), 'data')
    ENCODINGS_FILE = os.path.join(DATA_FOLDER, 'student_encodings.pkl')
    STUDENT_CSV = os.path.join(DATA_FOLDER, 'Student.csv')


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
