# import Flask and jsonify
from flask import Flask, jsonify, request
# import Resource, Api and reqparser
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy as np 
import pickle

# App definition
app = Flask(__name__)

api=Api(app)

class DataframeFunctionTransformer():
    def __init__(self, func):
        self.func = func
    def transform(self, input_df, **transform_params):
        return self.func(input_df)
    def fit(self, X, y=None, **fit_params):
       # print(X) # used this for testing
        return self
# this function takes a dataframe as input and
# returns a modified version thereof
def process_dataframe(input_df):
    
    input_df["Total_Income"] = (input_df["ApplicantIncome"]+input_df['CoapplicantIncome']).map(lambda i: np.log(i) if i>0 else 0)
    input_df['LoanAmount']=input_df['LoanAmount'].map(lambda i: np.log(i) if i >0 else 0)
    return input_df
    

class FeatureSelector:
    def __init__(self, feats):
        self.feats = feats

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        return X[self.feats]

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
feats=['Dependents','Education','Self_Employed', 
'LoanAmount',
'Loan_Amount_Term',
'Credit_History',
'Property_Area', 'Total_Income']

    
model = pickle.load( open( "pipe1.pkl", "rb" ) )
# assign endpoint

class Scoring(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        # getting predictions from our model.
        # it is much simpler because we used pipelines during development
        res = model.predict(df)
        # we cannot send numpt array as a result
        return res.tolist() 

api.add_resource(Scoring, '/scoring')

if __name__=='__main__':
      
    app.run(host='0.0.0.0', port=5000, debug=True)
      
 