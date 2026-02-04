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
from ..models import Class, Student, Attendance, ClassEnrollment, db
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
    from sqlalchemy import func

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

    # Per-class attendance stats
    class_stats = {}
    for cls in classes:
        # Count unique dates (sessions) for this class
        sessions = db.session.query(func.count(func.distinct(Attendance.date)))\
            .filter(Attendance.class_id == cls.id).scalar() or 0

        # Count total attendance records and present records
        total_records = Attendance.query.filter_by(class_id=cls.id).count()
        present_records = Attendance.query.filter_by(class_id=cls.id, status='present').count()

        # Calculate average attendance percentage
        avg_percentage = round((present_records / total_records * 100), 1) if total_records > 0 else 0

        class_stats[cls.id] = {
            'sessions': sessions,
            'total_records': total_records,
            'present_records': present_records,
            'avg_percentage': avg_percentage
        }

    return render_template('teacher/slcm_dashboard.html',
                           classes=classes,
                           class_stats=class_stats,
                           total_students=total_students,
                           total_classes=total_classes,
                           recent_attendance=recent_attendance[:10],
                           active_tab='dashboard')


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
            # Get student details for recognized faces (deduplicated by reg_no, keeping highest confidence)
            recognized_by_reg = {}
            for match in recognition_results['recognized']:
                reg_no = match['reg_no']
                if reg_no not in recognized_by_reg or match['confidence'] > recognized_by_reg[reg_no]['confidence']:
                    student = Student.query.filter_by(reg_no=reg_no).first()
                    if student:
                        recognized_by_reg[reg_no] = {
                            'id': student.id,
                            'reg_no': student.reg_no,
                            'name': student.name,
                            'confidence': match['confidence'],
                            'status': 'confirmed'
                        }
            recognized_students = list(recognized_by_reg.values())

            # Get student details for close matches (deduplicated, excluding already recognized)
            close_match_by_reg = {}
            for match in recognition_results['close_matches']:
                reg_no = match['reg_no']
                # Skip if already in recognized list
                if reg_no in recognized_by_reg:
                    continue
                if reg_no not in close_match_by_reg or match['confidence'] > close_match_by_reg[reg_no]['confidence']:
                    student = Student.query.filter_by(reg_no=reg_no).first()
                    if student:
                        close_match_by_reg[reg_no] = {
                            'id': student.id,
                            'reg_no': student.reg_no,
                            'name': student.name,
                            'confidence': match['confidence'],
                            'status': 'pending'
                        }
            close_match_students = list(close_match_by_reg.values())

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

    from datetime import date as date_module
    return render_template('teacher/slcm_attendance.html', form=form, results=results, today=date_module.today(), active_tab='attendance')


@teacher_bp.route('/attendance/confirm', methods=['POST'])
@login_required
@teacher_required
def confirm_attendance():
    """Save confirmed attendance records to database.

    Marks recognized students as present and enrolled non-recognized students as absent.
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    class_id = data.get('class_id')
    student_ids = data.get('student_ids', [])
    attendance_date = date.today()

    if not class_id:
        return jsonify({'error': 'Missing class_id'}), 400

    try:
        # Get all students enrolled in this class
        enrollments = ClassEnrollment.query.filter_by(class_id=class_id).all()
        enrolled_student_ids = {e.student_id for e in enrollments}

        # Get the IDs of students marked as present (from face recognition)
        present_student_ids = {item.get('id') for item in student_ids}

        # Each save creates a NEW session - always create new records
        # This allows multiple attendance sessions per day

        # Mark present students
        for item in student_ids:
            student_id = item.get('id')
            confidence = item.get('confidence', 100)

            attendance = Attendance(
                student_id=student_id,
                class_id=class_id,
                date=attendance_date,
                status='present',
                confidence=confidence,
                marked_by=current_user.id
            )
            db.session.add(attendance)

        # Mark absent students (enrolled but not in the photo)
        absent_student_ids = enrolled_student_ids - present_student_ids
        for student_id in absent_student_ids:
            attendance = Attendance(
                student_id=student_id,
                class_id=class_id,
                date=attendance_date,
                status='absent',
                confidence=None,
                marked_by=current_user.id
            )
            db.session.add(attendance)

        db.session.commit()

        # Return actual counts for this session
        present_count = len(present_student_ids)
        absent_count = len(absent_student_ids)

        return jsonify({
            'success': True,
            'message': f'Attendance saved: {present_count} present, {absent_count} absent',
            'present_count': present_count,
            'absent_count': absent_count
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
    return render_template('teacher/slcm_students.html', students=all_students, form=form, add_photo_form=add_photo_form, active_tab='students')


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


@teacher_bp.route('/class/<int:class_id>/records')
@login_required
@teacher_required
def class_records(class_id):
    """View attendance records for a specific class."""
    from sqlalchemy import func
    from collections import defaultdict

    # Get the class
    cls = Class.query.get_or_404(class_id)

    # Verify the teacher owns this class (or show all classes for now)
    # if cls.teacher_id != current_user.id:
    #     flash('Access denied.', 'danger')
    #     return redirect(url_for('teacher.dashboard'))

    # Get all attendance records for this class
    attendance_records = Attendance.query.filter_by(class_id=class_id)\
        .order_by(Attendance.date.desc(), Attendance.student_id).all()

    # Group attendance by date (each date = one session)
    sessions = defaultdict(list)
    for record in attendance_records:
        sessions[record.date].append(record)

    # Convert to list of session data with stats
    session_list = []
    for session_date, records in sorted(sessions.items(), reverse=True):
        present_count = sum(1 for r in records if r.status == 'present')
        total_count = len(records)
        session_list.append({
            'date': session_date,
            'records': records,
            'present_count': present_count,
            'total_count': total_count,
            'percentage': round((present_count / total_count * 100), 1) if total_count > 0 else 0
        })

    # Overall stats
    total_sessions = len(session_list)
    total_records = len(attendance_records)
    total_present = sum(1 for r in attendance_records if r.status == 'present')
    avg_attendance = round((total_present / total_records * 100), 1) if total_records > 0 else 0

    return render_template('teacher/slcm_class_records.html',
                           cls=cls,
                           sessions=session_list,
                           total_sessions=total_sessions,
                           total_records=total_records,
                           total_present=total_present,
                           avg_attendance=avg_attendance,
                           active_tab='records')


@teacher_bp.route('/attendance/<int:attendance_id>/toggle', methods=['POST'])
@login_required
@teacher_required
def toggle_attendance(attendance_id):
    """Toggle attendance status between present and absent."""
    record = Attendance.query.get_or_404(attendance_id)

    # Toggle status
    if record.status == 'present':
        record.status = 'absent'
        record.confidence = None  # Clear confidence when manually changed
    else:
        record.status = 'present'
        record.confidence = None

    record.marked_by = current_user.id
    db.session.commit()

    return jsonify({
        'success': True,
        'new_status': record.status,
        'message': f'Attendance updated to {record.status}'
    })


@teacher_bp.route('/attendance/<int:attendance_id>/delete', methods=['POST'])
@login_required
@teacher_required
def delete_attendance(attendance_id):
    """Delete an attendance record."""
    record = Attendance.query.get_or_404(attendance_id)
    class_id = record.class_id

    db.session.delete(record)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': 'Attendance record deleted'
    })


@teacher_bp.route('/class/<int:class_id>/export')
@login_required
@teacher_required
def export_attendance(class_id):
    """Export attendance records to CSV."""
    import csv
    from io import StringIO
    from flask import Response

    cls = Class.query.get_or_404(class_id)

    # Get all attendance records for this class
    records = Attendance.query.filter_by(class_id=class_id)\
        .order_by(Attendance.date.desc(), Attendance.student_id).all()

    # Create CSV in memory
    output = StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(['Date', 'Student Reg No', 'Student Name', 'Status', 'Confidence', 'Marked At'])

    # Write data
    for record in records:
        writer.writerow([
            record.date.strftime('%Y-%m-%d'),
            record.student.reg_no if record.student else 'N/A',
            record.student.name if record.student else 'N/A',
            record.status.capitalize(),
            f'{record.confidence:.1f}%' if record.confidence else '-',
            record.created_at.strftime('%Y-%m-%d %H:%M') if record.created_at else '-'
        ])

    # Create response
    output.seek(0)
    filename = f'attendance_{cls.code}_{date.today().strftime("%Y%m%d")}.csv'

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )


@teacher_bp.route('/class/<int:class_id>/export-summary')
@login_required
@teacher_required
def export_attendance_summary(class_id):
    """Export attendance summary (per student) to CSV."""
    import csv
    from io import StringIO
    from flask import Response
    from collections import defaultdict

    cls = Class.query.get_or_404(class_id)

    # Get all attendance records for this class
    records = Attendance.query.filter_by(class_id=class_id).all()

    # Group by student
    student_stats = defaultdict(lambda: {'present': 0, 'absent': 0, 'total': 0})
    for record in records:
        if record.student:
            key = (record.student.reg_no, record.student.name)
            student_stats[key]['total'] += 1
            if record.status == 'present':
                student_stats[key]['present'] += 1
            else:
                student_stats[key]['absent'] += 1

    # Create CSV
    output = StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(['Reg No', 'Student Name', 'Total Classes', 'Present', 'Absent', 'Attendance %'])

    # Write data sorted by reg_no
    for (reg_no, name), stats in sorted(student_stats.items()):
        percentage = round((stats['present'] / stats['total'] * 100), 2) if stats['total'] > 0 else 0
        writer.writerow([
            reg_no,
            name,
            stats['total'],
            stats['present'],
            stats['absent'],
            f'{percentage}%'
        ])

    output.seek(0)
    filename = f'attendance_summary_{cls.code}_{date.today().strftime("%Y%m%d")}.csv'

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )
