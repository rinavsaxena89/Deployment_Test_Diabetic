# importing the lib
from flask import Flask , render_template, request
import joblib

app = Flask(__name__)

#load the model
model = joblib.load('C:/Users/DELL/Desktop/VS_Code_Deployment/Saving_Model/joblib/diabetic_75.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/data', methods=['post'])
def data():
    preg = (int)(request.form.get("preg"))
    plas = (int)(request.form.get("plas"))
    pres = (int)(request.form.get("pres"))
    skin = (int)(request.form.get("skin"))
    test = (int)(request.form.get("test"))
    mass = (int)(request.form.get("mass"))
    pedi = (int)(request.form.get("pedi"))
    age = (int)(request.form.get("age"))

    result = model.predict([[preg, plas, pres, skin, test, mass, pedi, age]])

    if result[0]==1:
        data = 'person is diabatic'
    else:
        data = 'person is not diabatic'

    print(data)
    
    return 'data received'

app.run(debug = True) # should be always at the end