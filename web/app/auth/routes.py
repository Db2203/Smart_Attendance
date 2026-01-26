from flask import render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from . import auth_bp
from .forms import LoginForm, RegistrationForm
from ..models import User, Student
from .. import db


@auth_bp.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.is_teacher():
            return redirect(url_for('teacher.dashboard'))
        else:
            return redirect(url_for('student.dashboard'))
    return redirect(url_for('auth.login'))


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        if current_user.is_teacher():
            return redirect(url_for('teacher.dashboard'))
        return redirect(url_for('student.dashboard'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.lower()).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember_me.data)
            next_page = request.args.get('next')
            if not next_page:
                if user.is_teacher():
                    next_page = url_for('teacher.dashboard')
                else:
                    next_page = url_for('student.dashboard')
            flash('Login successful!', 'success')
            return redirect(next_page)
        flash('Invalid email or password.', 'danger')

    return render_template('auth/login.html', form=form)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('auth.index'))

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            email=form.email.data.lower(),
            name=form.name.data,
            role=form.role.data
        )
        user.set_password(form.password.data)

        # If student, create or link student record
        if form.role.data == 'student' and form.reg_no.data:
            student = Student.query.filter_by(reg_no=form.reg_no.data).first()
            if not student:
                student = Student(
                    reg_no=form.reg_no.data,
                    name=form.name.data,
                    email=form.email.data.lower()
                )
                db.session.add(student)
                db.session.flush()  # Get the student ID
            user.student_id = student.id

        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/register.html', form=form)


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))
