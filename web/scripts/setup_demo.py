"""
Demo Setup Script
Creates 6 subjects, student user accounts, and enrolls all students in all classes.
Run with: python web/scripts/setup_demo.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import create_app, db
from app.models import User, Student, Class, ClassEnrollment

# 6 Subjects for the demo
SUBJECTS = [
    ("OE", "OPEN ELECTIVE"),
    ("CV", "COMPUTER VISION"),
    ("SCP", "SOFT COMPUTING PARADIGMS"),
    ("NLP", "NATURAL LANGUAGE PROCESSING"),
    ("DL", "DEEP LEARNING"),
    ("EHCS", "ETHICAL HACKING AND CYBER SECURITY"),
]

DEFAULT_STUDENT_PASSWORD = "student123"


def setup_demo():
    app = create_app()

    with app.app_context():
        print("=" * 50)
        print("DEMO SETUP SCRIPT")
        print("=" * 50)

        # 1. Get or create teacher
        teacher = User.query.filter_by(role='teacher').first()
        if not teacher:
            print("\nNo teacher found. Please register a teacher account first.")
            print("Go to http://127.0.0.1:5000/auth/register and create a teacher account.")
            return

        print(f"\nUsing teacher: {teacher.name} ({teacher.email})")

        # 2. Create 6 classes/subjects
        print("\n--- Creating Classes ---")
        for code, name in SUBJECTS:
            existing = Class.query.filter_by(code=code).first()
            if existing:
                print(f"  [EXISTS] {code}: {name}")
            else:
                new_class = Class(code=code, name=name, teacher_id=teacher.id)
                db.session.add(new_class)
                print(f"  [CREATED] {code}: {name}")

        db.session.commit()

        # 3. Get all classes
        classes = Class.query.all()
        print(f"\nTotal classes: {len(classes)}")

        # 4. Get all students
        students = Student.query.all()
        print(f"Total students: {len(students)}")

        if not students:
            print("\nNo students found. Please import students first.")
            print("Run: python web/scripts/import_data.py")
            return

        # 5. Create User accounts for students (if not exists)
        print("\n--- Creating Student User Accounts ---")
        created_count = 0
        for student in students:
            # Check if user already exists for this student
            existing_user = User.query.filter_by(student_id=student.id).first()
            if existing_user:
                print(f"  [EXISTS] {student.name}")
                continue

            # Generate email if student doesn't have one
            email = student.email if student.email else f"{student.reg_no}@student.edu"

            # Check if email is already used
            if User.query.filter_by(email=email).first():
                email = f"{student.reg_no}@student.edu"

            # Create user account
            user = User(
                email=email,
                name=student.name,
                role='student',
                student_id=student.id
            )
            user.set_password(DEFAULT_STUDENT_PASSWORD)
            db.session.add(user)
            created_count += 1
            print(f"  [CREATED] {student.name} - {email}")

        db.session.commit()
        print(f"\nCreated {created_count} new student accounts")

        # 6. Enroll all students in all classes
        print("\n--- Enrolling Students in Classes ---")
        enrollment_count = 0
        for student in students:
            for cls in classes:
                existing = ClassEnrollment.query.filter_by(
                    student_id=student.id,
                    class_id=cls.id
                ).first()
                if not existing:
                    enrollment = ClassEnrollment(student_id=student.id, class_id=cls.id)
                    db.session.add(enrollment)
                    enrollment_count += 1

        db.session.commit()
        print(f"Created {enrollment_count} new enrollments")

        # Summary
        print("\n" + "=" * 50)
        print("DEMO SETUP COMPLETE!")
        print("=" * 50)
        print(f"\nTeacher Account:")
        print(f"  Email: {teacher.email}")
        print(f"  (Use the password you registered with)")

        print(f"\nStudent Accounts:")
        print(f"  Total: {len(students)}")
        print(f"  Password for all: {DEFAULT_STUDENT_PASSWORD}")

        print(f"\nClasses Created: {len(classes)}")
        for cls in classes:
            print(f"  - {cls.code}: {cls.name}")

        print(f"\nAll {len(students)} students enrolled in all {len(classes)} classes.")
        print("\nYou can now:")
        print("1. Login as teacher and mark attendance")
        print("2. Login as any student to view their attendance")


if __name__ == "__main__":
    setup_demo()
