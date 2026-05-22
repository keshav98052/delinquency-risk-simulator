import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- PAGE SETUP ---
st.set_page_config(layout="centered", page_title="Smart Risk Simulator")
st.title("Dynamic Delinquency Risk Simulator")
st.markdown("Upload your historical credit dataset. The AI will learn the patterns and build a custom risk simulator for you.")

# ==========================================
# 1. DATA UPLOADER & BULLETPROOF CLEANING
# ==========================================
uploaded_file = st.file_uploader("Upload your Credit Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1')
        
    # STRIP HIDDEN SPACES: This prevents 99% of KeyErrors
    df.columns = df.columns.str.strip()
    
    st.success("✅ Data Loaded. Training AI Brain...")
    
    df_clean = df.copy()
    
    # Drop ID to prevent model confusion
    if 'ID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['ID'])
        
    # BULLETPROOF TARGET SELECTION: Always grab the very last column automatically
    target_col = df_clean.columns[-1]
    
    # Split Features (X) and Target (y)
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    feature_names = X.columns.tolist()

    # Automatic Data Imputation & Encoding
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        X[col] = X[col].fillna(median_val)
    
    le = LabelEncoder()
    for col in cat_cols:
        mode_val = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(mode_val)
        X[col] = X[col].fillna(mode_val)
        
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        X[col] = le.fit_transform(X[col].astype(str))

    # ==========================================
    # 2. AI TRAINING
    # ==========================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # THE FIX: .tolist() converts the data to standard Python integers, 
    # completely bypassing the Windows/NumPy 2.0 compatibility bug!
    y_encoded = le.fit_transform(y.astype(str)).tolist()
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y_encoded)
    
    importances = rf_model.feature_importances_
    top_4_idx = np.argsort(importances)[-4:][::-1]
    top_4_features = [feature_names[i] for i in top_4_idx]
    top_4_importances = [importances[i] for i in top_4_idx]

    # ==========================================
    # 3. OPERATION MODE SELECTOR
    # ==========================================
    st.write("---")
    st.subheader("⚙️ Select Operation Mode")
    operation_mode = st.radio(
        "How would you like to analyze the risk?",
        ("🎯 Manual Scenario Simulator", "🚀 Auto-Calculate Entire Portfolio"),
        horizontal=True
    )

    # ==========================================
    # MODE A: MANUAL SCENARIO SIMULATOR
    # ==========================================
    if operation_mode == "🎯 Manual Scenario Simulator":
        st.write("---")
        chart_box = st.container()
        st.write("---")
        metric_box = st.container()
        st.write("---")
        slider_box = st.container()

        user_inputs = {}
        with slider_box:
            st.subheader("🎛️ Adjust Client Profile")
            col1, col2 = st.columns(2)
            
            with col1:
                for i in range(2):
                    feat = top_4_features[i]
                    min_val = float(df_clean[feat].min())
                    max_val = float(df_clean[feat].max())
                    if min_val == max_val: max_val += 1.0 
                    user_inputs[feat] = st.slider(f"{feat}", min_value=min_val, max_value=max_val, value=float(df_clean[feat].median()))
                    
            with col2:
                for i in range(2, 4):
                    feat = top_4_features[i]
                    min_val = float(df_clean[feat].min())
                    max_val = float(df_clean[feat].max())
                    if min_val == max_val: max_val += 1.0 
                    user_inputs[feat] = st.slider(f"{feat}", min_value=min_val, max_value=max_val, value=float(df_clean[feat].median()))

        # Live AI Prediction
        mock_row = []
        for feature in feature_names:
            if feature in user_inputs:
                mock_row.append(user_inputs[feature])
            else:
                mock_row.append(df_clean[feature].median())
                
        mock_row_scaled = scaler.transform([mock_row])
        # Predict probability of the positive class (usually class '1' or 'Y')
        risk_probability = rf_model.predict_proba(mock_row_scaled)[0][-1] * 100 
        
        if risk_probability < 20: tier = "Low Risk"
        elif risk_probability < 45: tier = "Medium Risk"
        elif risk_probability < 70: tier = "High Risk"
        else: tier = "Critical Risk"

        with chart_box:
            st.subheader("📊 Primary Risk Factors (AI Learned)")
            df_chart = pd.DataFrame({
                "Key Drivers": top_4_features,
                "Importance (Model Weight)": top_4_importances
            }).sort_values(by="Importance (Model Weight)", ascending=True)

            fig = px.bar(
                df_chart, x="Importance (Model Weight)", y="Key Drivers", 
                orientation='h', color_discrete_sequence=['#006400']
            )
            fig.update_layout(plot_bgcolor="white", height=250, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with metric_box:
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(label="Predicted Probability of Default", value=f"{risk_probability:.1f}%")
            with metric_col2:
                st.metric(label="Risk Tier", value=tier)

        # Dynamic Clean Action List 
        st.write("---")
        st.subheader("📋 Flagged Customer Action List")
        
        flagged_indices = df_clean.index
        for feature, slider_value in user_inputs.items():
            if len(df_clean[feature].unique()) < 15:
                condition = df_clean.loc[flagged_indices, feature] >= slider_value
                flagged_indices = flagged_indices[condition]
            else:
                lower_bound = slider_value * 0.85
                upper_bound = slider_value * 1.15
                condition = (df_clean.loc[flagged_indices, feature] >= lower_bound) & (df_clean.loc[flagged_indices, feature] <= upper_bound)
                flagged_indices = flagged_indices[condition]

        flagged_df = df.loc[flagged_indices].copy()

        if len(flagged_df) > 0:
            try:
                cols = flagged_df.columns.tolist()
                if target_col in cols:
                    cols.remove(target_col)
                    cols.insert(0, target_col)
                    flagged_df = flagged_df[cols]
            except Exception:
                pass 
                
            st.dataframe(flagged_df, use_container_width=True)
            
            csv_export = flagged_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Roster as CSV",
                data=csv_export,
                file_name='flagged_risk_customers.csv',
                mime='text/csv',
            )

    # ==========================================
    # MODE B: AUTO-CALCULATE ENTIRE PORTFOLIO
    # ==========================================
    elif operation_mode == "🚀 Auto-Calculate Entire Portfolio":
        st.write("---")
        st.subheader("🚀 Automated Portfolio Audit")
        st.markdown("Click the button below to force the AI to evaluate your entire database and sort all customers into action lists.")
        
        if st.button("Auto-Calculate All Customers", use_container_width=True):
            with st.spinner("The AI is scoring all customers... Please wait."):
                
                # Predict probability for everyone
                all_probabilities = rf_model.predict_proba(X_scaled)[:, -1] * 100
                
                portfolio_df = df.copy()
                portfolio_df['Risk Probability (%)'] = all_probabilities.round(1)
                
                # Determine Tiers
                conditions = [
                    (portfolio_df['Risk Probability (%)'] < 20),
                    (portfolio_df['Risk Probability (%)'] >= 20) & (portfolio_df['Risk Probability (%)'] < 45),
                    (portfolio_df['Risk Probability (%)'] >= 45) & (portfolio_df['Risk Probability (%)'] < 70),
                    (portfolio_df['Risk Probability (%)'] >= 70)
                ]
                choices = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
                
                # THE CRITICAL NUMPY FIX IS RIGHT HERE: default='Unknown'
                portfolio_df['AI Risk Tier'] = np.select(conditions, choices, default='Unknown')
                
                # Rearrange columns to show AI scores first
                cols = portfolio_df.columns.tolist()
                cols = ['AI Risk Tier', 'Risk Probability (%)'] + [c for c in cols if c not in ['AI Risk Tier', 'Risk Probability (%)']]
                portfolio_df = portfolio_df[cols]
                
                st.success("✅ Audit Complete! Select a tab below to view the rosters.")
                
                # Build Tabs
                tab1, tab2, tab3, tab4 = st.tabs(["🔴 Critical Risk", "🟠 High Risk", "🟡 Medium Risk", "🟢 Low Risk"])
                
                with tab1:
                    crit_df = portfolio_df[portfolio_df['AI Risk Tier'] == 'Critical Risk'].sort_values(by='Risk Probability (%)', ascending=False)
                    st.error(f"**Found {len(crit_df)} Critical Risk Customers** (>= 70% Default Probability)")
                    st.dataframe(crit_df, use_container_width=True)
                    
                with tab2:
                    high_df = portfolio_df[portfolio_df['AI Risk Tier'] == 'High Risk'].sort_values(by='Risk Probability (%)', ascending=False)
                    st.warning(f"**Found {len(high_df)} High Risk Customers** (45% - 69% Default Probability)")
                    st.dataframe(high_df, use_container_width=True)
                    
                with tab3:
                    med_df = portfolio_df[portfolio_df['AI Risk Tier'] == 'Medium Risk'].sort_values(by='Risk Probability (%)', ascending=False)
                    st.info(f"**Found {len(med_df)} Medium Risk Customers** (20% - 44% Default Probability)")
                    st.dataframe(med_df, use_container_width=True)
                    
                with tab4:
                    low_df = portfolio_df[portfolio_df['AI Risk Tier'] == 'Low Risk'].sort_values(by='Risk Probability (%)', ascending=True)
                    st.success(f"**Found {len(low_df)} Low Risk Customers** (< 20% Default Probability)")
                    st.dataframe(low_df, use_container_width=True)

else:
    st.info("Awaiting dataset. Please upload your dataset to generate the simulator.")