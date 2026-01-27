#!/usr/bin/env python
"""Import existing student data from CSV and encodings from PKL into the database."""

import os
import sys
import pickle
import pandas as pd

# Add the web app to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import create_app, db
from app.models import Student, User, Class


def import_students(csv_path, pkl_path=None):
    """Import students from CSV file into database.

    Args:
        csv_path: Path to Student.csv
        pkl_path: Path to student_encodings.pkl (optional)
    """
    app = create_app('development')

    with app.app_context():
        # Read CSV
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            return

        df = pd.read_csv(csv_path, dtype=str)
        df['Reg No'] = df['Reg No'].str.strip()

        # Load encodings if available
        encodings_dict = {}
        if pkl_path and os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                encodings_dict = pickle.load(f)
            print(f"Loaded encodings for {len(encodings_dict)} students")

        # Import each student
        imported = 0
        updated = 0
        skipped = 0

        for _, row in df.iterrows():
            reg_no = row['Reg No'].strip()
            name = row['Name'].strip()
            file_paths = row.get('File Paths', '')

            # Check if student already exists
            existing = Student.query.filter_by(reg_no=reg_no).first()

            if existing:
                # Update existing student
                existing.name = name
                existing.photo_path = file_paths

                # Update encoding if available
                if reg_no in encodings_dict:
                    existing.face_encoding = pickle.dumps(encodings_dict[reg_no])

                updated += 1
                print(f"Updated: {reg_no} - {name}")
            else:
                # Create new student
                student = Student(
                    reg_no=reg_no,
                    name=name,
                    photo_path=file_paths
                )

                # Add encoding if available
                if reg_no in encodings_dict:
                    student.face_encoding = pickle.dumps(encodings_dict[reg_no])

                db.session.add(student)
                imported += 1
                print(f"Imported: {reg_no} - {name}")

        db.session.commit()

        print(f"\nSummary:")
        print(f"  Imported: {imported}")
        print(f"  Updated: {updated}")
        print(f"  Total: {imported + updated}")


def create_default_class(teacher_email=None):
    """Create a default class for testing."""
    app = create_app('development')

    with app.app_context():
        # Check if default class exists
        existing = Class.query.filter_by(code='DEFAULT').first()
        if existing:
            print(f"Default class already exists: {existing.name}")
            return existing

        # Find a teacher to assign
        teacher = None
        if teacher_email:
            teacher = User.query.filter_by(email=teacher_email, role='teacher').first()

        if not teacher:
            teacher = User.query.filter_by(role='teacher').first()

        if not teacher:
            print("No teacher found. Please create a teacher account first.")
            return None

        # Create default class
        default_class = Class(
            name='Default Class',
            code='DEFAULT',
            teacher_id=teacher.id
        )
        db.session.add(default_class)
        db.session.commit()

        print(f"Created default class assigned to: {teacher.email}")
        return default_class


if __name__ == '__main__':
    # Paths relative to project root
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    csv_path = os.path.join(project_root, 'data', 'Student.csv')
    pkl_path = os.path.join(project_root, 'data', 'student_encodings.pkl')

    print("=" * 50)
    print("Student Data Import Script")
    print("=" * 50)

    print(f"\nCSV Path: {csv_path}")
    print(f"PKL Path: {pkl_path}")

    print("\n--- Importing Students ---")
    import_students(csv_path, pkl_path)

    print("\n--- Creating Default Class ---")
    create_default_class()

    print("\nDone!")
