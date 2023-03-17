from flask import Flask, request
from distutils import debug

app = Flask(__name__) # create flask object
# Define routes
@app.route('/model', methods=['POST'])
def hello_world():
    request_data = request.get_json(force=True)
    model_name = request_data['model']
    data_transform_name=request_data['data_transform']
    return "You are requesting for a {0} model".format(model_name),"And data transform".format(data_transform_name)


if __name__ == '__main__':
    app.run(port=8006, debug=True)

import numpy

my_list = [2, 8, 7, -5, 3]

def sqr(x):
    return x*2

def main():
    print([sqr(x) for x in my_list])

if __name__ == "__main__":
    main()
