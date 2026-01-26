from flask import render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from functools import wraps
from . import teacher_bp


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
    return render_template('teacher/dashboard.html')


@teacher_bp.route('/attendance')
@login_required
@teacher_required
def attendance():
    return render_template('teacher/attendance.html')


@teacher_bp.route('/students')
@login_required
@teacher_required
def students():
    return render_template('teacher/students.html')
