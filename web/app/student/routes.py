from flask import render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from functools import wraps
from sqlalchemy import func
from . import student_bp
from ..models import Student, Attendance, Class, ClassEnrollment, db


def student_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_student():
            flash('Access denied. Students only.', 'danger')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


@student_bp.route('/dashboard')
@login_required
@student_required
def dashboard():
    # Get the student record linked to current user
    student = Student.query.get(current_user.student_id)

    if not student:
        flash('Student profile not found.', 'danger')
        return redirect(url_for('auth.logout'))

    # Get enrolled classes
    enrollments = ClassEnrollment.query.filter_by(student_id=student.id).all()
    enrolled_classes = [e.class_ for e in enrollments]
    classes_count = len(enrolled_classes)

    # Calculate attendance stats
    total_records = Attendance.query.filter_by(student_id=student.id).count()
    present_count = Attendance.query.filter_by(student_id=student.id, status='present').count()
    absent_count = total_records - present_count

    # Calculate attendance percentage
    attendance_rate = round((present_count / total_records * 100), 1) if total_records > 0 else 0

    # Get recent attendance records (last 10)
    recent_attendance = Attendance.query.filter_by(student_id=student.id)\
        .order_by(Attendance.date.desc(), Attendance.created_at.desc())\
        .limit(10).all()

    return render_template('student/slcm_dashboard.html',
                           student=student,
                           attendance_rate=attendance_rate,
                           present_count=present_count,
                           absent_count=absent_count,
                           classes_count=classes_count,
                           recent_attendance=recent_attendance,
                           enrolled_classes=enrolled_classes,
                           active_tab='academics')


@student_bp.route('/history')
@login_required
@student_required
def history():
    student = Student.query.get(current_user.student_id)

    if not student:
        flash('Student profile not found.', 'danger')
        return redirect(url_for('auth.logout'))

    # Get all enrolled classes
    enrollments = ClassEnrollment.query.filter_by(student_id=student.id).all()
    enrolled_classes = [e.class_ for e in enrollments]

    # Get attendance grouped by class
    attendance_by_class = {}
    for cls in enrolled_classes:
        records = Attendance.query.filter_by(
            student_id=student.id,
            class_id=cls.id
        ).order_by(Attendance.date.desc()).all()

        present = sum(1 for r in records if r.status == 'present')
        total = len(records)
        percentage = round((present / total * 100), 1) if total > 0 else 0

        attendance_by_class[cls] = {
            'records': records,
            'present': present,
            'total': total,
            'percentage': percentage
        }

    return render_template('student/history.html',
                           student=student,
                           attendance_by_class=attendance_by_class)


@student_bp.route('/stats')
@login_required
@student_required
def stats():
    student = Student.query.get(current_user.student_id)

    if not student:
        flash('Student profile not found.', 'danger')
        return redirect(url_for('auth.logout'))

    # Get enrolled classes
    enrollments = ClassEnrollment.query.filter_by(student_id=student.id).all()
    enrolled_classes = [e.class_ for e in enrollments]

    # Calculate stats per class
    class_stats = []
    for cls in enrolled_classes:
        records = Attendance.query.filter_by(
            student_id=student.id,
            class_id=cls.id
        ).all()

        present = sum(1 for r in records if r.status == 'present')
        total = len(records)
        percentage = round((present / total * 100), 1) if total > 0 else 0

        class_stats.append({
            'class': cls,
            'present': present,
            'absent': total - present,
            'total': total,
            'percentage': percentage
        })

    # Overall stats
    total_records = Attendance.query.filter_by(student_id=student.id).count()
    present_count = Attendance.query.filter_by(student_id=student.id, status='present').count()
    overall_percentage = round((present_count / total_records * 100), 1) if total_records > 0 else 0

    return render_template('student/stats.html',
                           student=student,
                           class_stats=class_stats,
                           overall_percentage=overall_percentage,
                           total_present=present_count,
                           total_classes=total_records)
