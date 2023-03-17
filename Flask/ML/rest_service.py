from flask import Flask, request
import pickle
import numpy as np

# Load Models
local_model1 = pickle.load(open('Model1.pickle', 'rb'))
local_model2 = pickle.load(open('Model2.pickle', 'rb'))
local_model3 = pickle.load(open('Model3.pickle', 'rb'))
local_model4 = pickle.load(open('Model4.pickle', 'rb'))
local_model5 = pickle.load(open('Model5.pickle', 'rb'))
# Load Standard Scaler Transform
# Create flask application
app = Flask(__name__)

@app.route('/model', methods=['POST'])
def my_model():
    request_data = request.get_json(force=True)
    classifier=request_data['classifier']
    data_transformm=request_data['data_transform']

    print(f'Classifier: {classifier} and Data Transform: {data_transformm}')
    # Create a 2D array
    data = np.array([[classifier, data_transformm]])
    if classifier=='LogisticRegression':
        results=local_model1
    elif classifier=='SVM':
        results==local_model5
    elif classifier=='RandomForest':
        results==local_model3
    elif classifier=='DecisionTree':
        results=local_model2
    else:
        results=local_model4
        
    return {"here are the results :":
             float(results)
        
    }


if __name__ == "__main__":
    app.run(port=8009, debug=True)    



