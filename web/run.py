#!/usr/bin/env python
"""Entry point for the Smart Attendance Web Application."""

import os
from app import create_app, db
from app.models import User, Student, Class, Attendance

app = create_app(os.environ.get('FLASK_ENV', 'development'))


@app.shell_context_processor
def make_shell_context():
    """Make database models available in flask shell."""
    return {
        'db': db,
        'User': User,
        'Student': Student,
        'Class': Class,
        'Attendance': Attendance
    }


@app.cli.command()
def init_db():
    """Initialize the database."""
    db.create_all()
    print('Database initialized.')


@app.cli.command()
def create_admin():
    """Create an admin/teacher account."""
    email = input('Enter email: ')
    name = input('Enter name: ')
    password = input('Enter password: ')

    if User.query.filter_by(email=email).first():
        print('User already exists!')
        return

    user = User(email=email, name=name, role='teacher')
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    print(f'Teacher account created for {email}')


@app.context_processor
def inject_now():
    """Inject current datetime into templates."""
    from datetime import datetime
    return {'now': datetime.utcnow}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
