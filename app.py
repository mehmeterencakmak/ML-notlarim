import pandas as pd
import numpy as np
from flask import Flask,render_template,request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
df=pd.read_csv(r'finalcleaned5.csv')
le = preprocessing.LabelEncoder()
crop=le.fit_transform(df['locality'])
df['loc']=crop
df.drop('locality',axis='columns')


app=Flask(__name__)
@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def pre():
    lr=LinearRegression()
    
    a = float(request.form.get('p1'))
    b = float(request.form.get('p2'))
    c = float(request.form.get('p3'))
    d = float(request.form.get('p4'))
    lr.fit(df[['loc','bathroom','property_size','type_bhk',]],df.rent_amount)
    k=lr.predict([[a,b,c,d]])
    return render_template('index.html',p=k[0]/200)



if __name__=='__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run( debug =True)

