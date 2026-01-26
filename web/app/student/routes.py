from flask import render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from functools import wraps
from . import student_bp


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
    return render_template('student/dashboard.html')


@student_bp.route('/history')
@login_required
@student_required
def history():
    return render_template('student/history.html')


@student_bp.route('/stats')
@login_required
@student_required
def stats():
    return render_template('student/stats.html')
