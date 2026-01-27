"""Routes for teacher module."""

import os
import uuid
from datetime import date
from flask import render_template, redirect, url_for, flash, request, current_app, jsonify
from flask_login import login_required, current_user
from functools import wraps
from werkzeug.utils import secure_filename

from . import teacher_bp
from .forms import AttendanceForm, ConfirmAttendanceForm, StudentForm, AddPhotoForm
import pickle
from ..models import Class, Student, Attendance, db
from ..face_recognition import FaceRecognitionService


def teacher_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_teacher():
            flash('Access denied. Teachers only.', 'danger')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


@teacher_bp.route('/dashboard')
@login_required
@teacher_required
def dashboard():
    # Get teacher's classes
    classes = Class.query.filter_by(teacher_id=current_user.id).all()

    # Get recent attendance records
    recent_attendance = []
    for cls in classes:
        records = Attendance.query.filter_by(class_id=cls.id)\
            .order_by(Attendance.date.desc())\
            .limit(5).all()
        recent_attendance.extend(records)

    # Stats
    total_students = Student.query.count()
    total_classes = len(classes)

    return render_template('teacher/dashboard.html',
                           classes=classes,
                           total_students=total_students,
                           total_classes=total_classes,
                           recent_attendance=recent_attendance[:10])


@teacher_bp.route('/attendance', methods=['GET', 'POST'])
@login_required
@teacher_required
def attendance():
    form = AttendanceForm()

    # Populate class choices - show all classes for now
    classes = Class.query.all()
    if not classes:
        # Create a default class if none exist
        default_class = Class(
            name='Default Class',
            code='DEFAULT',
            teacher_id=current_user.id
        )
        db.session.add(default_class)
        db.session.commit()
        classes = [default_class]

    form.class_id.choices = [(c.id, c.name) for c in classes]

    results = None

    if form.validate_on_submit():
        # Save uploaded image
        image = form.image.data
        filename = secure_filename(f"{uuid.uuid4().hex}_{image.filename}")
        upload_folder = os.path.join(current_app.root_path, 'static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        image_path = os.path.join(upload_folder, filename)
        image.save(image_path)

        # Initialize face recognition service
        fr_service = FaceRecognitionService()

        # Load encodings from database
        students = Student.query.filter(Student.face_encoding.isnot(None)).all()
        fr_service.load_encodings_from_db(students)

        # Process the image
        recognition_results = fr_service.process_image(image_path)

        if recognition_results['error']:
            flash(recognition_results['error'], 'warning')
        else:
            # Get student details for recognized faces
            recognized_students = []
            for match in recognition_results['recognized']:
                student = Student.query.filter_by(reg_no=match['reg_no']).first()
                if student:
                    recognized_students.append({
                        'id': student.id,
                        'reg_no': student.reg_no,
                        'name': student.name,
                        'confidence': match['confidence'],
                        'status': 'confirmed'
                    })

            # Get student details for close matches
            close_match_students = []
            for match in recognition_results['close_matches']:
                student = Student.query.filter_by(reg_no=match['reg_no']).first()
                if student:
                    close_match_students.append({
                        'id': student.id,
                        'reg_no': student.reg_no,
                        'name': student.name,
                        'confidence': match['confidence'],
                        'status': 'pending'
                    })

            results = {
                'recognized': recognized_students,
                'close_matches': close_match_students,
                'unknown_count': recognition_results['unknown_count'],
                'total_faces': len(recognized_students) + len(close_match_students) + recognition_results['unknown_count'],
                'image_url': url_for('static', filename=f'uploads/{filename}'),
                'class_id': form.class_id.data
            }

            if recognized_students:
                flash(f'Successfully recognized {len(recognized_students)} student(s)!', 'success')
            if close_match_students:
                flash(f'{len(close_match_students)} student(s) need confirmation.', 'info')
            if recognition_results['unknown_count'] > 0:
                flash(f'{recognition_results["unknown_count"]} face(s) could not be identified.', 'warning')

    return render_template('teacher/attendance.html', form=form, results=results)


@teacher_bp.route('/attendance/confirm', methods=['POST'])
@login_required
@teacher_required
def confirm_attendance():
    """Save confirmed attendance records to database."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    class_id = data.get('class_id')
    student_ids = data.get('student_ids', [])
    attendance_date = date.today()

    if not class_id or not student_ids:
        return jsonify({'error': 'Missing class_id or student_ids'}), 400

    try:
        saved_count = 0
        for item in student_ids:
            student_id = item.get('id')
            confidence = item.get('confidence', 100)

            # Check if attendance already exists for this student/class/date
            existing = Attendance.query.filter_by(
                student_id=student_id,
                class_id=class_id,
                date=attendance_date
            ).first()

            if not existing:
                attendance = Attendance(
                    student_id=student_id,
                    class_id=class_id,
                    date=attendance_date,
                    status='present',
                    confidence=confidence,
                    marked_by=current_user.id
                )
                db.session.add(attendance)
                saved_count += 1

        db.session.commit()
        return jsonify({
            'success': True,
            'message': f'Attendance saved for {saved_count} student(s)',
            'saved_count': saved_count
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@teacher_bp.route('/students')
@login_required
@teacher_required
def students():
    # Get all students
    all_students = Student.query.order_by(Student.name).all()
    form = StudentForm()
    add_photo_form = AddPhotoForm()
    return render_template('teacher/students.html', students=all_students, form=form, add_photo_form=add_photo_form)


@teacher_bp.route('/students/add', methods=['POST'])
@login_required
@teacher_required
def add_student():
    """Add a new student."""
    form = StudentForm()

    if form.validate_on_submit():
        # Create student
        student = Student(
            reg_no=form.reg_no.data.strip(),
            name=form.name.data.strip(),
            email=form.email.data.strip() if form.email.data else None
        )

        # Handle photo uploads
        photos = request.files.getlist('photos')
        if photos and photos[0].filename:
            # Create upload directory for this student
            student_folder = os.path.join(
                current_app.root_path, 'static', 'uploads', 'students', form.reg_no.data.strip()
            )
            os.makedirs(student_folder, exist_ok=True)

            # Process photos and generate encodings
            fr_service = FaceRecognitionService()
            all_encodings = []
            errors = []

            for photo in photos:
                if photo and photo.filename:
                    filename = secure_filename(f"{uuid.uuid4().hex}_{photo.filename}")
                    photo_path = os.path.join(student_folder, filename)
                    photo.save(photo_path)

                    # Generate encoding
                    encoding, error = fr_service.generate_encoding_from_file(photo_path)
                    if encoding is not None:
                        all_encodings.append(encoding)
                        # Store first photo path
                        if not student.photo_path:
                            student.photo_path = f"uploads/students/{form.reg_no.data.strip()}/{filename}"
                    elif error:
                        errors.append(error)
                        # Delete photo if no face detected
                        os.remove(photo_path)

            # Save encodings
            if all_encodings:
                student.face_encoding = pickle.dumps(all_encodings)
                flash(f'Successfully generated {len(all_encodings)} face encoding(s).', 'success')
            elif errors:
                flash(f'Photo errors: {"; ".join(errors)}', 'warning')

        db.session.add(student)
        db.session.commit()
        flash(f'Student "{student.name}" added successfully!', 'success')
    else:
        for field, errs in form.errors.items():
            for err in errs:
                flash(f'{field}: {err}', 'danger')

    return redirect(url_for('teacher.students'))


@teacher_bp.route('/students/<int:id>', methods=['GET'])
@login_required
@teacher_required
def get_student(id):
    """Get student details as JSON for edit modal."""
    student = Student.query.get_or_404(id)
    return jsonify({
        'id': student.id,
        'reg_no': student.reg_no,
        'name': student.name,
        'email': student.email or '',
        'photo_path': student.photo_path,
        'has_encoding': student.face_encoding is not None
    })


@teacher_bp.route('/students/<int:id>/edit', methods=['POST'])
@login_required
@teacher_required
def edit_student(id):
    """Edit an existing student."""
    student = Student.query.get_or_404(id)
    form = StudentForm(original_reg_no=student.reg_no)

    if form.validate_on_submit():
        student.reg_no = form.reg_no.data.strip()
        student.name = form.name.data.strip()
        student.email = form.email.data.strip() if form.email.data else None

        # Handle new photo uploads if provided
        photos = request.files.getlist('photos')
        if photos and photos[0].filename:
            student_folder = os.path.join(
                current_app.root_path, 'static', 'uploads', 'students', student.reg_no
            )
            os.makedirs(student_folder, exist_ok=True)

            fr_service = FaceRecognitionService()
            new_encodings = []
            errors = []

            for photo in photos:
                if photo and photo.filename:
                    filename = secure_filename(f"{uuid.uuid4().hex}_{photo.filename}")
                    photo_path = os.path.join(student_folder, filename)
                    photo.save(photo_path)

                    encoding, error = fr_service.generate_encoding_from_file(photo_path)
                    if encoding is not None:
                        new_encodings.append(encoding)
                        if not student.photo_path:
                            student.photo_path = f"uploads/students/{student.reg_no}/{filename}"
                    elif error:
                        errors.append(error)
                        os.remove(photo_path)

            if new_encodings:
                # Merge with existing encodings
                student.face_encoding = fr_service.merge_encodings(
                    student.face_encoding, new_encodings
                )
                flash(f'Added {len(new_encodings)} new face encoding(s).', 'success')
            elif errors:
                flash(f'Photo errors: {"; ".join(errors)}', 'warning')

        db.session.commit()
        flash(f'Student "{student.name}" updated successfully!', 'success')
    else:
        for field, errs in form.errors.items():
            for err in errs:
                flash(f'{field}: {err}', 'danger')

    return redirect(url_for('teacher.students'))


@teacher_bp.route('/students/<int:id>/delete', methods=['POST'])
@login_required
@teacher_required
def delete_student(id):
    """Delete a student."""
    student = Student.query.get_or_404(id)
    name = student.name

    # Delete student's photo folder if exists
    student_folder = os.path.join(
        current_app.root_path, 'static', 'uploads', 'students', student.reg_no
    )
    if os.path.exists(student_folder):
        import shutil
        shutil.rmtree(student_folder)

    # Delete attendance records for this student
    Attendance.query.filter_by(student_id=id).delete()

    db.session.delete(student)
    db.session.commit()
    flash(f'Student "{name}" deleted successfully.', 'success')

    return redirect(url_for('teacher.students'))


@teacher_bp.route('/students/<int:id>/add-photo', methods=['POST'])
@login_required
@teacher_required
def add_student_photo(id):
    """Add additional photos to an existing student."""
    student = Student.query.get_or_404(id)
    form = AddPhotoForm()

    if form.validate_on_submit():
        photos = request.files.getlist('photos')
        if photos and photos[0].filename:
            student_folder = os.path.join(
                current_app.root_path, 'static', 'uploads', 'students', student.reg_no
            )
            os.makedirs(student_folder, exist_ok=True)

            fr_service = FaceRecognitionService()
            new_encodings = []
            errors = []

            for photo in photos:
                if photo and photo.filename:
                    filename = secure_filename(f"{uuid.uuid4().hex}_{photo.filename}")
                    photo_path = os.path.join(student_folder, filename)
                    photo.save(photo_path)

                    encoding, error = fr_service.generate_encoding_from_file(photo_path)
                    if encoding is not None:
                        new_encodings.append(encoding)
                        if not student.photo_path:
                            student.photo_path = f"uploads/students/{student.reg_no}/{filename}"
                    elif error:
                        errors.append(error)
                        os.remove(photo_path)

            if new_encodings:
                student.face_encoding = fr_service.merge_encodings(
                    student.face_encoding, new_encodings
                )
                db.session.commit()
                flash(f'Added {len(new_encodings)} new photo(s) for {student.name}.', 'success')
            elif errors:
                flash(f'Errors: {"; ".join(errors)}', 'warning')
        else:
            flash('No photos selected.', 'warning')

    return redirect(url_for('teacher.students'))


@teacher_bp.route('/classes')
@login_required
@teacher_required
def classes():
    """View and manage classes."""
    teacher_classes = Class.query.filter_by(teacher_id=current_user.id).all()
    return render_template('teacher/classes.html', classes=teacher_classes)


@teacher_bp.route('/reports')
@login_required
@teacher_required
def reports():
    """View attendance reports."""
    classes = Class.query.filter_by(teacher_id=current_user.id).all()
    return render_template('teacher/reports.html', classes=classes)
