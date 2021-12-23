import json, os, joblib, confuse
import numpy as np
import pandas as pd
from azureml.core.model import Model
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

input_sample = pd.DataFrame({
    "user": pd.Series(['------Username------'],dtype="object"),
    "subreddit": pd.Series(['AskReddit'],dtype="object"),
})

def init():
    global model
    config = confuse.Configuration('SubredditEndpoint')
    
    model_path = Model.get_model_path('subreddit-rec')
    model = joblib.load(model_path)

@input_schema('data',PandasParameterType(input_sample))
def run(data):
    result = model.predict(uid=data['user'].values[0],iid=data['subreddit'].values[0])
    result_dict = {
        'uid': result.uid,
        'iid': result.iid,
        'est': result.est,
        'details': result.details
    }
    return json.dumps({
        "prediction": result_dict
    })