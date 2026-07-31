from app import db
from flask_login import UserMixin
from sqlalchemy.sql import func

class Users(db.Model, UserMixin):
    __tablename__ = 'Users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(150), nullable=True)  # Added to capture full name from signup
    status = db.Column(db.String, default='Free')       # Replaced role column with premium status boolean
    created_at = db.Column(db.DateTime, server_default=func.now())
    level_of_study = db.Column(db.String(100), nullable=True)  # New column for level of study
    current_class = db.Column(db.String(100), nullable=True)   # New column for current class
    
    # Relationship to courses
    courses = db.relationship('Course', backref='owner', lazy=True)

class Course(db.Model):
    __tablename__ = 'courses'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('Users.id'), nullable=False)
    is_mathematical = db.Column(db.Boolean, default=False)

class Book(db.Model):
    __tablename__ = 'books'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False, index=True)  # Indexed for fast searching
    author = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(100), nullable=False, index=True)  # Indexed for category filters
    description = db.Column(db.Text, nullable=True)
    file_path = db.Column(db.String(512), nullable=False)  # Server path for digital assets (PDFs, EPUBs, etc.)
    cover_url = db.Column(db.String(512), nullable=True)   # Book cover image URL or server path
    rating = db.Column(db.Float, default=0.0)
    price = db.Column(db.Float, default=0.0)
    is_free = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, server_default=func.now())
