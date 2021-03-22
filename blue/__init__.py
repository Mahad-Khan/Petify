from flask import Flask
from flask_cors import CORS, cross_origin
from flask_restful import Api,Resource
from blue.api import routes
from blue.api.routes import db
from blue.api.routes import mod  
import os


dirname = os.path.dirname(os.path.abspath(__file__))
dirname_list = dirname.split("/")[:-1]
dirname = "/".join(dirname_list)

print("inside init")
print("dir +",dirname)


app = Flask(__name__)
CORS(app)

db.init_app(app)

db_path = 'sqlite:///'+dirname+'/petmypal.db'
print("Database path: "+db_path)
app.config['SQLALCHEMY_DATABASE_URI'] = db_path
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True
with app.app_context():
    # Imports
    db.create_all()

app.register_blueprint(api.routes.mod,url_prefix = '/api')
