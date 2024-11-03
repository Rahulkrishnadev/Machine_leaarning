from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app=FastAPI()

model=joblib.load('model.joblib')

class modelinput(BaseModel):
     crim:float
     zn :float
     indus:float
     chas:int
     nox:float
     rm:float
     age:float
     dis:float
     rad:int  
     tax:int  
     ptratio:float
     b:float
     lstat:float
    

@app.get('/')
def welcome():
     return " welcome to Housing price prediction model"


@app.post('/predict')
def pred(input:modelinput):
     data=[[input.crim,input.zn,input.indus,input.chas,input.nox,input.rm,input.age,input.dis,input.rad,input.tax,input.ptratio,input.b,input.lstat]]
     prediction=model.predict(data)
     predicted_value = float(prediction[0])
     return {'prediction':predicted_value}
