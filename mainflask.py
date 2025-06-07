from flask import Flask, request, render_template,jsonify
import joblib


app=Flask(__name__)

model=joblib.load('model.pkl')


@app.route('/predict',Methods=['POST'])
def predict():
    data=request.get_json(force=True)
    prediction=model.predict([list(data.values())])
    return jsonify({'prediction':prediction[0]})



if __name__=='__main__':
    app.run(host='0.0.0.0',port=9080,debug=True)
