
# 
# Module dependencies.
# 

import os
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template
from predict import predict_breed


app = Flask(__name__)


@app.route("/")
def index():
  return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
  data = {
    "success": False
  }

  # Grab file.
  file = request.files["file"]
  
  # Save the file to ./uploads
  basepath = os.path.dirname(__file__)
  file_path = os.path.join(basepath, "./static/uploads", secure_filename(file.filename))
  file.save(file_path)

  # Make prediction
  breed = predict_breed(file_path)

  data = {
    "success": True,
    "breed": breed
  }

  return jsonify(data)


if __name__ == '__main__':
  app.run(debug=True)



  