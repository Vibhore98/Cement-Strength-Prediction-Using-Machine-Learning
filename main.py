from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('RFMODEL.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('modelindex.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form['material_qty']
    val2 = request.form['additive_cat']
    val3 = request.form['ash_com']
    val4 = request.form['platicizer']
    val5 = request.form['formulation_durt']
    arr = np.array([val1, val2, val3, val4,val5])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    return render_template('modelindex.html', Calculating=float(pred))

app.config['TEMPLATES_AUTO_RELOAD'] = True
if __name__ == '__main__':
    app.run(debug=True,port=5000)

