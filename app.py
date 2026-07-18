import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_cors import CORS

db = SQLAlchemy()

def create_app():
    app = Flask(__name__, template_folder='templates')
    CORS(app)
    app.secret_key = 'key'
    
    # 1. Fetch the Render database URL from the environment, fallback to local SQLite
    database_url = os.getenv('DATABASE_URL', 'sqlite:///testdb.db')
    
    # 2. Fix Render's "postgres://" prefix for SQLAlchemy 1.4+ compatibility
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
        
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    
    db.init_app(app)
    login_manager = LoginManager()
    login_manager.init_app(app)
    
    from models import Users
    @login_manager.user_loader
    def loadUser(uid):
        return Users.query.get(uid)
    
    from routes import register_routes
    register_routes(app, db)
    migrate = Migrate(app, db)
    
    # 3. Automatically generate empty database tables if they don't exist yet
    with app.app_context():
        db.create_all()
        
    return app
