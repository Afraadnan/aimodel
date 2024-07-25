from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from ai_model import reg
from waitress import serve
# Save the model to a file
pickle.dump(reg, open('ai_model.pkl', 'wb'))

# Load the model from the file
model = pickle.load(open('ai_model.pkl', 'rb'))

print(model.predict([[20, 40]]))  # Example prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.form.to_dict()
    int_features = [int(data[feature]) for feature in data]
    final_features = [np.array(int_features)]  # Convert to array

    # Make prediction
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Time = {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
