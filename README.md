Dynamic Delinquency Risk Simulator 

A sophisticated, interactive business intelligence tool built with Streamlit and Scikit-Learn designed to predict credit card delinquency risk. This application allows financial analysts to move beyond "black-box" machine learning by providing a transparent, interactive simulator to stress-test credit profiles and audit large portfolios.

Overview
Traditional risk models often provide a score without context. This application bridges the gap between predictive modeling and actionable insight by offering two distinct operational modes:
1.  Manual Scenario Simulator: Move sliders to adjust key financial features in real-time and see how the model's risk prediction updates instantly.
2.  Automated Portfolio Audit: Batch-process entire datasets to categorize customers into Critical, High, Medium, or Low risk tiers for prioritized collections and intervention.

Key Features
 Dynamic Data Agnostic: Automatically detects column names, handles missing values, and encodes categorical data without requiring code changes for different CSV structures.
 AI-Learned Drivers: Automatically identifies the top 4 predictive drivers (e.g., payment status, credit limit, age) from your dataset and builds interactive sliders based on those specific variables.
 Actionable Intelligence: Provides an automated "Flagged Customer List" with export capabilities (CSV) to streamline operations for collections and risk management teams.
 Transparent Analytics: Uses interactive Plotly visualizations to show "Model Importance," giving users visibility into why a profile is flagged as high-risk.

Tech Stack
 Frontend: Streamlit
 ML Engine: Scikit-Learn (Random Forest Classifier, StandardScaler)
 Visualization: Plotly
 Data Handling: Pandas, NumPy

Launch the application:

streamlit run simulator.py

Privacy & Security Best Practices
This project has been configured with a .gitignore file to ensure security.

Data Safety: Real customer datasets are excluded from repository version control.

Deployment: When deploying this app (e.g., to Streamlit Cloud), always ensure your training data is stored securely in a private environment or anonymized before upload.

Usage Instructions
Upload: Drop your credit dataset (CSV format) into the uploader.<img width="1009" height="692" alt="Screenshot 2026-05-22 090141" src="https://github.com/user-attachments/assets/d71a4f64-7da3-4c40-afa9-e8f71a592c80" />
<img width="910" height="579" alt="Screenshot 2026-05-22 090157" src="https://github.com/user-attachments/assets/4ecd2889-e5fa-46cd-b44e-cb414e85eac7" />

Train: The model automatically handles missing values and categorical encoding.

Select Mode: Choose between Manual Simulation (for scenario testing) or Auto-Calculate (for bulk portfolio auditing).<img width="320" height="685" alt="Screenshot 2026-05-22 090253" src="https://github.com/user-attachments/assets/9c5e0042-f8c9-4878-bf35-f526b8bd0d5b" />
<img width="327" height="446" alt="Screenshot 2026-05-22 090343" src="https://github.com/user-attachments/assets/539bc421-c3c2-4e65-b01d-9bd9cd052dff" />

Export: Download high-risk customer lists directly as a CSV to hand off for business review.
