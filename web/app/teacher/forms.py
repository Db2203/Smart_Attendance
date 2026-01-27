"""Forms for teacher module."""

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired, MultipleFileField
from wtforms import SelectField, SubmitField, HiddenField, StringField
from wtforms.validators import DataRequired, Optional, Email, ValidationError
from ..models import Student


class AttendanceForm(FlaskForm):
    """Form for uploading class photo and marking attendance."""

    class_id = SelectField('Select Class', coerce=int, validators=[DataRequired()])
    image = FileField('Class Photo', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')
    ])
    submit = SubmitField('Process & Mark Attendance')


class ConfirmAttendanceForm(FlaskForm):
    """Form for confirming recognized students."""

    class_id = HiddenField()
    date = HiddenField()
    # Student IDs will be added dynamically via JavaScript
    submit = SubmitField('Confirm Attendance')


class StudentForm(FlaskForm):
    """Form for adding/editing students."""

    reg_no = StringField('Registration Number', validators=[DataRequired()])
    name = StringField('Full Name', validators=[DataRequired()])
    email = StringField('Email', validators=[Optional(), Email()])
    photos = MultipleFileField('Face Photos', validators=[
        FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')
    ])
    submit = SubmitField('Save Student')

    def __init__(self, original_reg_no=None, *args, **kwargs):
        super(StudentForm, self).__init__(*args, **kwargs)
        self.original_reg_no = original_reg_no

    def validate_reg_no(self, field):
        """Check if registration number is unique."""
        if field.data != self.original_reg_no:
            student = Student.query.filter_by(reg_no=field.data).first()
            if student:
                raise ValidationError('This registration number is already in use.')


class AddPhotoForm(FlaskForm):
    """Form for adding additional photos to existing student."""

    photos = MultipleFileField('Additional Photos', validators=[
        FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')
    ])
    submit = SubmitField('Add Photos')
