from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("app/car_price_predicition_a3.model", "rb") as f:
    model = pickle.load(f)

# Load scaler and class boundaries
with open("app/meta.pkl", "rb") as f:
    meta = pickle.load(f)

scaler = meta['scaler']
classes = meta['classes']  # e.g., [100000, 300000, 500000, 700000, 1000000]

@app.route('/')
def home():
    return render_template("final_model.html", prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        year = int(request.form['year'])
        mileage = float(request.form['mileage'])
        max_power = float(request.form['max_power'])

        # Prepare and scale input
        features = np.array([[year, mileage, max_power]])
        features[:, 0:3] = scaler.transform(features[:, 0:3])
        features = np.insert(features, 0, 1, axis=1)  # Add bias term if needed

        # Predict class index (0 to 3)
        class_idx = model.predict(features)[0]

        # Convert class index to readable price range
        prediction = f"Estimated price range: ${classes[class_idx]} - ${classes[class_idx + 1]}"

        return render_template("final_model.html", prediction=prediction)

    except Exception as e:
        return f"❌ Error during prediction: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
