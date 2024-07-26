from flask import Flask, render_template, request, jsonify
import pickle

# Load the trained model
with open('ai_model.pkl', 'rb') as model_file:
    reg = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        x = float(request.form['x'])
        y = float(request.form['y'])
        # Prepare the input data for prediction
        input_data = [[x, y]]
        # Make prediction using the loaded model
        prediction = reg.predict(input_data)
        result = prediction[0]  # Assuming the model returns an array
        return f"Prediction: {result}"
    except ValueError:
        return "Invalid input. Please enter valid float values."
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
