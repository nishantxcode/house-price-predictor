from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd

model = joblib.load('predictor/ml/model.pkl')
encoder = joblib.load('predictor/ml/location_encoder.pkl')

# Get original location names from encoder classes
locations = list(encoder.classes_)

def predict_price(request):
    predicted_price = None
    error_message = None

    if request.method == 'POST':
        try:
            location = request.POST.get('location')
            total_sqft = float(request.POST.get('total_sqft'))
            bath = int(request.POST.get('bath'))
            bhk = int(request.POST.get('bhk'))

            if not (300 <= total_sqft <= 10000):
                raise ValueError("Total Sqft must be between 300 and 10000")
            if not (1 <= bath <= 8):
                raise ValueError("Bathrooms must be between 1 and 8")
            if not (1 <= bhk <= 10):
                raise ValueError("BHK must be between 1 and 10")

            location_index = encoder.transform([location])[0]
            prediction = model.predict(np.array([[location_index, total_sqft, bath, bhk]]))
            predicted_price = round(prediction[0], 2)

        except Exception as e:
            error_message = str(e)

    return render(request, 'webapp/index.html', {
        'predicted_price': predicted_price,
        'error_message': error_message,
        'locations': locations
    })
