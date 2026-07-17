from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash



db = SQLAlchemy()



def create_app():
	app = Flask(__name__, template_folder = 'templates')
	CORS(app)
	app.secret_key ='key'
	app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///testdb.db'
	db.init_app(app)
	login_manager = LoginManager()
	login_manager.init_app(app)
	from models import Users
	@login_manager.user_loader
	def loadUser(uid):
		Users.query.get(uid)
	
	from routes import register_routes
	register_routes(app, db)
	migrate = Migrate(app, db)
	return app