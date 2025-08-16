from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import pickle
from datetime import datetime

# Load model, encoders, and scaler
with open('best_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open('encoder.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler_data = pickle.load(scaler_file)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Feature importance
FEATURE_IMPORTANCE = {
    'Contract': 0.15, 'tenure': 0.12, 'MonthlyCharges': 0.11,
    'TotalCharges': 0.10, 'OnlineSecurity': 0.08, 'TechSupport': 0.07,
    'InternetService': 0.06, 'PaymentMethod': 0.05, 'PaperlessBilling': 0.04,
    'SeniorCitizen': 0.03, 'Dependents': 0.03, 'Partner': 0.02,
    'DeviceProtection': 0.02, 'StreamingTV': 0.01, 'StreamingMovies': 0.01,
    'gender': 0.01
}

def make_prediction(input_data):
    prediction_data = {k: v for k, v in input_data.items() if k != 'customerID'}
    input_df = pd.DataFrame([prediction_data])

    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numerical_cols] = scaler_data.transform(input_df[numerical_cols])

    prediction = loaded_model.predict(input_df)[0]
    probability = loaded_model.predict_proba(input_df)[0, 1]
    return "Churn" if prediction == 1 else "No Churn", round(probability * 100, 2)

def get_recommendations(prediction, probability, input_data):
    recommendations = []
    if prediction == "Churn":
        if probability > 70:
            recommendations.append("Offer loyalty discount (10-15%)")
            recommendations.append("Provide personalized customer service outreach")
        if input_data['Contract'] == "Month-to-month":
            recommendations.append("Suggest switching to annual contract with discount")
        if input_data['OnlineSecurity'] == "No":
            recommendations.append("Offer free trial of online security features")
        if not recommendations:
            recommendations.append("Consider offering a small service credit or perk")
    else:
        recommendations.append("Continue current engagement strategies")
        if input_data['tenure'] < 12:
            recommendations.append("Consider onboarding satisfaction check")
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = probability = error = None
    recommendations = []
    feature_importance_data = []
    history_data = []
    customer_id = ""

    if request.method == 'POST':
        customer_id = request.form.get('customerID', '').strip()
        if not customer_id:
            error = "Customer ID is required"
        else:
            try:
                input_data = {
                    'customerID': customer_id,
                    'gender': request.form.get('gender', 'Male'),
                    'SeniorCitizen': int(request.form.get('SeniorCitizen', 0)),
                    'Partner': request.form.get('Partner', 'No'),
                    'Dependents': request.form.get('Dependents', 'No'),
                    'tenure': int(request.form.get('tenure', 0)),
                    'PhoneService': request.form.get('PhoneService', 'No'),
                    'MultipleLines': request.form.get('MultipleLines', 'No'),
                    'InternetService': request.form.get('InternetService', 'No'),
                    'OnlineSecurity': request.form.get('OnlineSecurity', 'No'),
                    'OnlineBackup': request.form.get('OnlineBackup', 'No'),
                    'DeviceProtection': request.form.get('DeviceProtection', 'No'),
                    'TechSupport': request.form.get('TechSupport', 'No'),
                    'StreamingTV': request.form.get('StreamingTV', 'No'),
                    'StreamingMovies': request.form.get('StreamingMovies', 'No'),
                    'Contract': request.form.get('Contract', 'Month-to-month'),
                    'PaperlessBilling': request.form.get('PaperlessBilling', 'No'),
                    'PaymentMethod': request.form.get('PaymentMethod', 'Electronic check'),
                    'MonthlyCharges': float(request.form.get('MonthlyCharges', 0)),
                    'TotalCharges': float(request.form.get('TotalCharges', 0)),
                }

                prediction, probability = make_prediction(input_data)
                recommendations = get_recommendations(prediction, probability, input_data)
                
                feature_importance_data = sorted(
                    [{'feature': k, 'importance': v} for k, v in FEATURE_IMPORTANCE.items()],
                    key=lambda x: x['importance'], reverse=True
                )
                
                if 'history' not in session:
                    session['history'] = []
                
                session['history'] = [{
                    'customerID': customer_id,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'prediction': prediction,
                    'probability': probability,
                    'tenure': input_data['tenure'],
                    'monthly_charges': input_data['MonthlyCharges']
                }] + session['history'][:4]
                
                history_data = session['history']

            except Exception as e:
                error = f"Error processing request: {str(e)}"

    return render_template(
        'index.html',
        prediction=prediction,
        probability=probability,
        recommendations=recommendations,
        feature_importance=feature_importance_data,
        history_data=history_data,
        customer_id=customer_id,
        error=error
    )

@app.route('/customer-analytics/<customer_id>')
def customer_analytics(customer_id):
    try:
        df = pd.read_csv('sample.csv')
        customer_data = df[df['customerID'] == customer_id].iloc[0].to_dict()
        
        analytics = {
            'total_months': customer_data.get('tenure', 0),
            'total_spent': customer_data.get('TotalCharges', 0),
            'services_used': [srv for srv in ['PhoneService', 'InternetService', 'StreamingTV', 'StreamingMovies'] 
                            if customer_data.get(srv, 'No') != 'No'],
            'risk_factors': [
                f for f in [
                    'Month-to-month contract' if customer_data.get('Contract') == 'Month-to-month' else None,
                    'No OnlineSecurity' if customer_data.get('OnlineSecurity') == 'No' else None,
                    'Electronic check payment' if customer_data.get('PaymentMethod') == 'Electronic check' else None,
                    'High monthly charges' if customer_data.get('MonthlyCharges', 0) > 70 else None
                ] if f
            ],
            'churn_status': customer_data.get('Churn', 'Unknown')
        }
        
        return render_template(
            'customer_analytics.html',
            customer_id=customer_id,
            customer_data=customer_data,
            analytics=analytics
        )
        
    except Exception as e:
        return render_template('customer_not_found.html', customer_id=customer_id)

if __name__ == '__main__':
    app.run(debug=True)