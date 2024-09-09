# EmpAttr_Detection
import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import os

# Load the trained model using joblib
model = joblib.load('stack_tuning_model.pkl')

# Define the Streamlit app
def main():

    st.title("Employee Attrition Prediction")

    # Collect user input
    st.header("Enter the following features to predict employee attrition:")

    name = st.text_input('Name'),

    id = st.text_input('Employee ID'),

    department = st.selectbox('Department', options=['IT', 'admin', 'engineering', 'finance',
    'management', 'marketing', 'procurement', 'product', 'sales', 'support', 'temp'])

    filed_complaint = st.selectbox('Filed Complaint in Last Three Years',options=['Yes','No'])

    recently_promoted = st.selectbox('Receive Promotion in Last Three Years',options=['Yes','No'])

    salary = st.number_input('Salary',min_value=0)

    satisfaction = st.number_input('Satisfaction Level (e.g., 0.1 - 1.0)', min_value=0.0, max_value=1.0, step=0.01)

    average_monthly_hour = st.number_input('Average Monthly Hours', min_value=0, step=1)

    tenure = st.number_input('Tenure (in years)', min_value=0, step=1)

    project_no = st.number_input('Number of Projects', min_value=0, step=1)

    last_evaluation = st.number_input('Last Evaluation (e.g., 0.1 - 1.0)', min_value=0.0, max_value=1.0, step=0.01)

    # Calculate 'work_life_balance' from 'satisfaction' and 'average_working_hour'
    work_life_balance = satisfaction * average_monthly_hour

    # Display the calculated 'project_evaluation' as a read-only input
    st.text_input('Work Life Balance', value=f"{work_life_balance:.2f}", disabled=True)

    # Encode the department as an integer
    department_mapping = {
        'IT': 0, 'admin': 1, 'engineering': 2, 'finance': 3, 'management': 4,
        'marketing': 5, 'other' : 6, 'procurement': 7, 'product': 8, 'sales': 9, 'support': 10, 'temp': 11
    }
    department_encoded = department_mapping.get(department, -1)

    complaint_mapping = {'Yes':1,'No':0}
    complaint_encoded = complaint_mapping.get(filed_complaint, -1)

    promotion_mapping = {'Yes':1,'No':0}
    promotion_encoded = promotion_mapping.get(recently_promoted, -1)

    # Initialize the SHAP explainer with a lambda function to ensure correct interpretation
    explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], np.zeros((1, 7)))

    # Preprocess the input (if needed)
    input_features = np.array([[satisfaction, average_monthly_hour, work_life_balance, tenure, project_no, promotion_encoded,complaint_encoded]])

    feature_names = ['Satisfaction Level', 'Average Monthly Hours', 'Work Life Balance',
    'Tenure', 'Number of Projects', 'Promotion', 'Filed Complaint']

    # Create a button for prediction
    if st.button("Predict"):

        # Predict the probability of attrition
        prediction_proba = model.predict_proba(input_features)
        likelihood = prediction_proba[0][1] * 100

        # Probability Gauge or Progress Bar
        st.subheader("Attrition Likelihood Gauge")
        st.write(f"The likelihood of the employee leaving the company is : **{likelihood:.2f}%**")
        st.progress(int(likelihood))

        # Perform SHAP analysis
        shap_values = explainer.shap_values(input_features)

        # Convert SHAP values to a DataFrame for better readability
        shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
        st.write("SHAP Values:", shap_values_df)

        # Display the SHAP values for the input
        st.header("SHAP Analysis")
        st.write("The SHAP values explain the contribution of each feature to the prediction.")

        # SHAP Summary Plot
        st.subheader("SHAP Summary Plot")
        fig_summary, ax_summary = plt.subplots()
        shap.summary_plot(shap_values, input_features, feature_names=feature_names,plot_type="bar", show=False)
        ax_summary.set_xlabel("SHAP Value (Impact on Model Output)")
        st.pyplot(fig_summary)

        # SHAP Decision Plot
        st.subheader("SHAP Decision Plot")
        fig_decision = plt.figure()
        shap.decision_plot(explainer.expected_value, shap_values, input_features, feature_names=feature_names)
        st.pyplot(fig_decision)

        st.subheader("SHAP Force Plot")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            # Generate the force plot
            fig, ax = plt.subplots(figsize=(12, 5))
            shap.force_plot(explainer.expected_value, shap_values, input_features, feature_names=feature_names, matplotlib=True, show=False)

            # Add labels for feature values
            for i in range(len(feature_names)):
                ax.text(0.5, 0.8 - (i * 0.1), f"{feature_names[i]} = {input_features[0][i]:.2f}", transform=ax.transAxes)

            plt.savefig(tmpfile.name, bbox_inches='tight')
            plt.close()
            st.image(tmpfile.name)

        recommendations = []
        if satisfaction < 0.55:
            recommendations.append("**Recommendation:** The employee's satisfaction level is low. Consider offering career development"
            "opportunities, recognition programs, or flexible working arrangements to improve their satisfaction.")
        if average_monthly_hour > 166.67:
            recommendations.append("**Recommendation:** The employee has high working hours. Consider discussing work-life balance options"
             "or redistributing workload to prevent burnout.")
        if recently_promoted == 'No' and tenure > 2:
            recommendations.append("**Recommendation:** The employee has not been promoted in the last three years despite a long tenure."
             "Consider providing opportunities for advancement or discussing career growth plans with them.")
        if filed_complaint == 'Yes':
            recommendations.append("**Recommendation:** The employee has filed a complaint in the last three years."
            "Ensure that the issue has been resolved satisfactorily and provide ongoing support to address any remaining concerns.")
        if work_life_balance > 60:
            recommendations.append("**Recommendation:** The employee's work-life balance appears to be poor. Consider reducing workload "
            "or offering more flexible working hours to improve their overall well-being.")

        recommendations_text = "\n\n".join([f"{rec}" for rec in recommendations]) if recommendations else "No specific recommendations."
        recommendations_csv = "; ".join(recommendations) if recommendations else "No specific recommendations."

        # Save the input features to a CSV file
        save_data = pd.DataFrame(input_features, columns=feature_names)
        save_data['Name'] = name
        save_data['Employee ID'] = id
        save_data['Department'] = department
        save_data['Satisfaction Level'] = satisfaction
        save_data['Average Monthly Hours'] = average_monthly_hour
        save_data['Work Life Balance'] = work_life_balance
        save_data['Tenure'] = tenure
        save_data['Number of Projects'] = project_no
        save_data['Promotion'] = promotion_encoded
        save_data['Filed Complaint'] = complaint_encoded
        save_data['Last Evaluation'] = last_evaluation
        save_data['Prediction Probability'] = likelihood
        save_data['Recommendations'] = recommendations_csv

        if os.path.exists('employee_predictions.csv'):
            save_data.to_csv('employee_predictions.csv', mode='a', header=False, index=False)
        else:
            save_data.to_csv('employee_predictions.csv', mode='w', header=True, index=False)

        # Recommendations based on key features
        st.subheader("Recommendations")
        st.write(recommendations_text)


if __name__ == "__main__":
    main()
