"""Regenerate all student face encodings using InsightFace.

This script converts all existing face encodings from the old dlib format (128-dim)
to the new InsightFace/ArcFace format (512-dim).

Run from project root:
    python regenerate_encodings.py
"""

import os
import sys
import pickle
import sqlite3
from pathlib import Path

# Add paths for imports
sys.path.insert(0, 'web/app/face_recognition')
sys.path.insert(0, 'src')

from service import FaceRecognitionService


def regenerate_web_encodings():
    """Regenerate encodings for all students in the web app database."""
    print("=" * 60)
    print("Regenerating Web App Encodings (SQLite Database)")
    print("=" * 60)

    db_path = 'web/instance/attendance.db'
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    # Initialize face recognition service
    print("\nInitializing InsightFace models...")
    service = FaceRecognitionService()
    service._load_models()

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all students with photos
    cursor.execute('SELECT id, reg_no, name, photo_path FROM students WHERE photo_path IS NOT NULL')
    students = cursor.fetchall()

    print(f"\nFound {len(students)} students with photos")
    print("-" * 60)

    success_count = 0
    error_count = 0

    for student_id, reg_no, name, photo_path in students:
        # Try different base paths for the photo
        possible_paths = [
            photo_path,  # As stored
            os.path.join('web/app/static', photo_path),  # Web static folder
            os.path.join('web/app/static/uploads', os.path.basename(photo_path)),  # Uploads folder
            photo_path.replace('images/', 'web/app/static/uploads/'),  # Alternative
        ]

        actual_path = None
        for p in possible_paths:
            if os.path.exists(p):
                actual_path = p
                break

        if not actual_path:
            print(f"  [{reg_no}] {name}: Photo not found - {photo_path}")
            error_count += 1
            continue

        # Generate new encoding
        encoding, error = service.generate_encoding_from_file(actual_path)

        if encoding is not None:
            # Store as pickled list (for compatibility with multiple encodings)
            encoding_bytes = pickle.dumps([encoding])
            cursor.execute(
                'UPDATE students SET face_encoding = ? WHERE id = ?',
                (encoding_bytes, student_id)
            )
            print(f"  [{reg_no}] {name}: OK (512-dim encoding)")
            success_count += 1
        else:
            print(f"  [{reg_no}] {name}: FAILED - {error}")
            error_count += 1

    conn.commit()
    conn.close()

    print("-" * 60)
    print(f"Success: {success_count}, Failed: {error_count}")
    return success_count, error_count


def regenerate_desktop_encodings():
    """Regenerate encodings from Student.csv for the desktop app."""
    print("\n" + "=" * 60)
    print("Regenerating Desktop App Encodings (student_encodings.pkl)")
    print("=" * 60)

    csv_path = 'data/Student.csv'
    pkl_path = 'data/student_encodings.pkl'

    if not os.path.exists(csv_path):
        print(f"Student.csv not found: {csv_path}")
        return

    import pandas as pd

    # Initialize face recognition service
    print("\nInitializing InsightFace models...")
    service = FaceRecognitionService()
    service._load_models()

    # Read student CSV
    df = pd.read_csv(csv_path, dtype=str)
    print(f"\nFound {len(df)} students in CSV")
    print("-" * 60)

    encodings_dict = {}
    success_count = 0
    error_count = 0

    for _, row in df.iterrows():
        reg_no = row['Reg No'].strip()
        name = row.get('Name', reg_no)
        file_paths = row.get('File Paths', '')

        if not file_paths:
            print(f"  [{reg_no}] {name}: No file paths")
            error_count += 1
            continue

        all_encodings = []
        paths = [p.strip() for p in file_paths.split(',')]

        for image_path in paths:
            if not os.path.exists(image_path):
                continue

            encoding, error = service.generate_encoding_from_file(image_path)
            if encoding is not None:
                all_encodings.append(encoding)

        if all_encodings:
            encodings_dict[reg_no] = all_encodings
            print(f"  [{reg_no}] {name}: OK ({len(all_encodings)} encodings)")
            success_count += 1
        else:
            print(f"  [{reg_no}] {name}: FAILED - No valid encodings")
            error_count += 1

    # Save to pickle
    with open(pkl_path, 'wb') as f:
        pickle.dump(encodings_dict, f)

    print("-" * 60)
    print(f"Success: {success_count}, Failed: {error_count}")
    print(f"Saved to: {pkl_path}")
    return success_count, error_count


def main():
    print("\n" + "=" * 60)
    print("  InsightFace Encoding Regeneration Script")
    print("  Converting from dlib (128-dim) to ArcFace (512-dim)")
    print("=" * 60)

    # Regenerate web app encodings
    web_success, web_error = regenerate_web_encodings()

    # Regenerate desktop app encodings if CSV exists
    if os.path.exists('data/Student.csv'):
        desktop_success, desktop_error = regenerate_desktop_encodings()
    else:
        print("\n[Info] data/Student.csv not found, skipping desktop encodings")
        desktop_success, desktop_error = 0, 0

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Web App:     {web_success} success, {web_error} failed")
    print(f"  Desktop App: {desktop_success} success, {desktop_error} failed")
    print("=" * 60)
    print("\nDone! All encodings are now in 512-dim InsightFace format.")


if __name__ == '__main__':
    main()
