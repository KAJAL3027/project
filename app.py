'''
This is the main file for the application. 
It contains the routes and views for the application.
'''

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from database import opendb, DB_URL
from database import User, Profile, SatelliteImage
from db_helper import *
from validators import *
from logger import log
from werkzeug.utils import secure_filename
import os
import torch
from predictor import predict_image, ImageClassifier



def session_add(key, value):
    session[key] = value

def save_file(file):
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    return path

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg','tiff']


def load_model():
    model_path = 'model_90pct_acc_v1.pt'
    model = torch.load(model_path)
    model.eval()
    return model

# global variables
app = Flask(__name__)
app.secret_key  = '()*(#@!@#)'
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload/satellite/image', methods=['POST'])
def upload_satellite_image():
    if 'isauth' not in session:
        flash('You need to login first', 'danger')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('dashboard'))
    if file and allowed_file(file.filename):
        path = save_file(file)
        session_add('last_satellite_image_path', path)
        flash('File uploaded successfully', 'success')
        db = opendb()
        user_id = session['user_id']
        satellite_image = SatelliteImage(path=path, added_by=user_id)
        db.add(satellite_image)
        db.commit()
        return redirect(url_for('gallery'))
    
@app.route('/delete/satellite/image/<int:id>')
def delete_satellite_image(id):
    if 'isauth' not in session:
        flash('You need to login first', 'danger')
        return redirect(url_for('index'))
    db = opendb()
    image = db.query(SatelliteImage).filter_by(id=id).first()
    if image is None:
        flash('Image not found', 'danger')
        return redirect(url_for('gallery'))
    else:
        path = image.path
        db.delete(image)
        db.commit()
        if os.path.exists(path):
            os.remove(path)
        flash('Image deleted successfully', 'success')
        return redirect(url_for('gallery'))

@app.route('/predict/satellite/image/<int:id>')
def predict_sattelite_image(id):
    if 'isauth' not in session:
        flash('You need to login first', 'danger')
        return redirect(url_for('index'))
    db = opendb()
    image = db.query(SatelliteImage).filter_by(id=id).first()
    message = ""
    if image is None:
        flash('Image not found', 'danger')
        return redirect(url_for('gallery'))
    else:
        result = predict_image(image.path, model)
        if result == 0:
            message = "No forest fire detected"
        else:
            message = "Forest fire detected"
    return render_template('predict.html', id=id, message=message, result=result, image=image)

@app.route('/gallery')
def gallery():
    db = opendb()
    images = db.query(SatelliteImage).all()
    return render_template('gallery.html', images=images)

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    if not validate_email(email):
        flash('Invalid email', 'danger')
        return redirect(url_for('index'))
    if not validate_password(password):
        flash('Invalid password', 'danger')
        return redirect(url_for('index'))
    db = opendb()
    user = db.query(User).filter_by(email=email).first()
    if user is not None and user.verify_password(password):
        session_add('user_id', user.id)
        session_add('user_name', user.name)
        session_add('user_email', user.email)
        session_add('isauth', True)
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid email or password', 'danger')
        return redirect(url_for('index'))
    
@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    cpassword = request.form.get('cpassword')
    db = opendb()
    if not validate_username(name):
        flash('Invalid username', 'danger')
        return redirect(url_for('index'))
    if not validate_email(email):
        flash('Invalid email', 'danger')
        return redirect(url_for('index'))
    if not validate_password(password):
        flash('Invalid password', 'danger')
        return redirect(url_for('index'))
    if password != cpassword:
        flash('Passwords do not match', 'danger')
        return redirect(url_for('index'))
    if db.query(User).filter_by(email=email).first() is not None    :
        flash('Email already exists', 'danger')
        return redirect(url_for('index'))
    elif db.query(User).filter_by(name=name).first() is not None:
        flash('Username already exists', 'danger')
        return redirect(url_for('index'))
    else:
        db_save(User(name=name, email=email, password=password))
        flash('User registered successfully', 'success')
        return redirect(url_for('index'))
    
@app.route('/dashboard')
def dashboard():
    if session.get('isauth'):
        return render_template('dashboard.html')
    else:
        return redirect(url_for('index'))

@app.route('/profile/add', methods=['POST'])
def add_profile():
    if session.get('isauth'):
        user_id = session.get('user_id')
        city = request.form.get('city')
        gender = request.form.get('gender')
        avatar = request.files.get('avatar')
        db = opendb()
        if not validate_city(city):
            flash('Invalid city', 'danger')
            return redirect(url_for('dashboard'))
        if not validate_avatar(avatar):
            flash('Invalid avatar file', 'danger')
            return redirect(url_for('dashboard'))
        if db.query(Profile).filter_by(user_id=user_id).first() is not None:
            flash('Profile already exists', 'danger')
            return redirect(url_for('view_profile'))
        else:
            db_save(Profile(user_id = user_id, city=city, gender=gender, avatar=save_file(avatar)))
            flash('Profile added successfully', 'success')
            return redirect(url_for('dashboard'))
    else:
        flash('Please login to continue', 'danger')
        return redirect(url_for('index'))
        
@app.route('/profile/edit', methods=['POST'])
def edit_profile():
    if session.get('isauth'):
        profile = db_get_by_field(Profile, user_id=session.get('user_id'))
        if profile is not None:
            profile.city = request.form.get('city')
            profile.gender = request.form.get('gender')
            avatar = request.files.get('avatar')
            if avatar is not None:
                profile.avatar = save_file(avatar)
            db_save(profile)
            flash('Profile updated successfully', 'success')
            return redirect(url_for('dashboard'))
    else:
        flash('Please login to continue', 'danger')
        return redirect(url_for('index'))    

@app.route('/profile')
def view_profile():
    if session.get('isauth'):
        profile = db_get_by_field(Profile, user_id=session.get('user_id'))
        if profile is not None:
            return render_template('profile.html', profile=profile)
        else:
            flash(f'<a class="text-danger" href="#" data-bs-toggle="modal" data-bs-target="#profileModal">Create a profile</a>', 'danger')
            return redirect(url_for('dashboard'))
    else:
        flash('Please login to continue', 'danger')
        return redirect(url_for('index'))


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000, debug=True)
 