# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
#import StringIO
import io
from io import StringIO
import sys
import signal
import traceback
import sklearn
import flask
import csv
import pandas as pd

#prefix = '/opt/ml/'
#model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            #with open(os.path.join(model_path, 'testmodel.pkl'), 'r') as inp:
            #with open('testmodel.pkl','rb') as inp:
                #cls.model = pickle.loads(inp)
            cls.model = pickle.load(open('testmodel.pkl','rb'))
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf.predict(input)

# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route('/')
def index():
    return "hello!!"

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    #return "hello"
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        #s = StringIO.StringIO(data)
        s= io.BytesIO(data.encode()) 
        #s = StringIO.read(data)
        #s = csv.reader('test.csv')
        data = pd.read_csv(s, header=None)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    #print('Invoked with {} records'.format(data.shape[0]))
    print('Invoked with {} records'.format(int(data[0])))

    # Do the prediction
    #predictions = ScoringService.predict(data)
    bmi=[int(data[0])]
    bmi=[bmi]
    predictions = ScoringService.predict(bmi)

    return flask.Response(response=predictions, status=200, mimetype='text/csv')

    # Convert from numpy back to CSV
    #out = StringIO.StringIO()
    #out = io.BytesIO()
    #pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    #result = out.getvalue()

    #return flask.Response(response=result, status=200, mimetype='text/csv')

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=8080)
