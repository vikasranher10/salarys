from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

##import pkl file
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Experience = int(request.form.get('Experience'))
        Team_Lead_Experience = int(request.form.get('Team_Lead_Experience'))
        Project_Manager_Experience = int(request.form.get('Project_Manager_Experience'))
        Certifications = int(request.form.get('Certifications'))
        
        new_data=standard_scaler.transform([[Experience,Team_Lead_Experience,Project_Manager_Experience,Certifications]])
        result=ridge_model.predict(new_data)
        
        return render_template('home.html',result=result[0])
        
        
    
    else:
        return render_template('home.html')
        
    


if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0")