from flask import Flask,render_template,request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            age=int(request.form.get('age')),
            workclass=(request.form.get('workclass')),
            fnlwgt=int(request.form.get('fnlwgt')),
            education=(request.form.get('education')),
            education_num=int(request.form.get('education_num')),
            married_status=(request.form.get('married_status')),
            occupation=(request.form.get('occupation')),
            relationship=(request.form.get('relationship')),
            race=(request.form.get('race')),
            sex=(request.form.get('sex')),
            capital_gain=int(request.form.get('capital_gain')),
            capital_loss=int(request.form.get('capital_loss')),
            hrs_pr_week=int(request.form.get('hrs_pr_week')),
            native_country=(request.form.get('native_country'))
            )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)