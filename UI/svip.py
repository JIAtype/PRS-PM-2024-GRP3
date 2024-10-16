import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler
from math import pi
import streamlit as st
import plotly.express as px

Ka = joblib.load('model/kmeans_A6.pkl')

def generate_cluster_radar_chart(df, model, name):
    """
    生成雷达图展示聚类结果。
    """
    numerical_features = df.select_dtypes(include='number').columns
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numerical_features])
    cluster_labels = model.predict(df_scaled)

    data = pd.DataFrame(df_scaled, columns=numerical_features)
    data['Cluster'] = cluster_labels
    mean_values = data.groupby('Cluster').mean().reset_index()
    feature_names_mapping = {
        'MemGen_x': 'Gender',
        'ASPT_x': 'Average Spending Per Transaction',
        'MaxSPT_x': 'Maximum Spending',
        'MinSPT_x': 'Minimum Spending',
        'ANT_x': 'Average Number of Transactions',
        'APDR_x': 'Average Purchase Frequency',
        'APinFavShop_x': 'Average Spending in Favorite Shop',
        'ATRinFavShop_x': 'Average Transactions in Favorite Shop',
        'NGinFavShop_x': 'Number of Visits to Favorite Shop',
        'NFavinFavShop_x': 'Number of Favorites in Favorite Shop',
        'MemAge_x': 'Age',
        'MemDuration_M_x': 'Duration (Months)',
    }
    features = mean_values.columns[1:]
    new_feature_names = [feature_names_mapping[feat] for feat in features]
    num_vars = len(features)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig = go.Figure()
    for i in range(len(mean_values)):
        values = mean_values.loc[i].drop('Cluster').values.tolist()
        values += values[:1]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=new_feature_names,
            fill='toself',
            name=f'Cluster {i}'
        ))
    fig.update_layout(
        title= name ,
        polar=dict(
            radialaxis=dict(visible=True),
            angularaxis=dict(tickvals=list(range(len(features))), ticktext=new_feature_names,)
        )
    )
    return fig, cluster_labels

def load_all_models(base_model_filenames, stacked_model_filename):
    all_models = list()
    # Load base models
    for filename in base_model_filenames:
        with open(filename, 'rb') as file:
            model_data = joblib.load(file)  # Load the base model data
        all_models.append(model_data)  # Store the entire model data

    # Load stacked model and scaler
    with open(stacked_model_filename, 'rb') as f:
        scaler, stacking_model = joblib.load(f)
    
    return all_models, scaler, stacking_model

def stacked_prediction(models, stacking_model, X_scaled):
    # Generate predictions from base models
    base_predictions = [model.predict(X_scaled) for model in models]
    # Stack the base predictions
    base_predictions = np.column_stack(base_predictions)
    # Make final predictions with the stacking model
    final_predictions = stacking_model.predict(base_predictions)
    return final_predictions

def stacked_dataset(members, inputX_scaled, inputX_unscaled):
    stackX = None
    for i, model_data in enumerate(members):
        model = model_data['model']  # Access the model instance
        if i < 4:  # For Logistic Regression, SVM, KNN, and MLP models
            yhat = model.predict(inputX_scaled)
        else:  # For the Random Forest model
            yhat = model.predict(inputX_unscaled)
        
        yhat = yhat.reshape(-1, 1)  # Reshape into a 2D array (n_samples, 1)
        
        # Stack predictions into [rows, members]
        if stackX is None:
            stackX = yhat
        else:
            stackX = np.hstack((stackX, yhat))  # Horizontally stack predictions
    return stackX

# 主界面部分
st.title("💎 SVIP Spotlight​")
st.header("Hybrid System to Predict High Spending Customers​")
st.markdown("""
    Welcome to our prediction tool! This tool aims to identify which of the newly joined members belong to the top 30% of high-value customers based on their historical spending data.

To achieve this goal, we employ a Multiple Classifier System that integrates five advanced models:
- **Logistic Regression**: Estimates the probability of customer spending.
- **Support Vector Machine**: Effectively handles high-dimensional data to enhance classification performance.
- **K-NN (K-Nearest Neighbors)**: Predicts spending based on the behavior of similar customers.
- **Multilayer Perceptron**: A deep learning model that captures complex spending patterns.
- **Random Forest**: Utilizes an ensemble of decision trees to improve prediction stability.

The Hybrid System can analyze the outputs of these models collectively, increasing the accuracy and reliability of predictions. By precisely identifying high-spending customers, you can better tailor your marketing strategies to enhance customer satisfaction and loyalty.

Start using the tool now to uncover potential high-value customers!
""")

UPLOAD_FOLDER = "UI/data"
# File upload section
if os.path.isdir(UPLOAD_FOLDER) and os.listdir(UPLOAD_FOLDER):
    uploaded_files = [f for f in os.listdir(UPLOAD_FOLDER) if f != ".DS_Store"]
    uploaded_files = ["Select File"] + uploaded_files
    selected_file = st.selectbox("Please select your file to start the analysis:", uploaded_files)
    df = None
    if selected_file != "Select File":
        file_path = os.path.join(UPLOAD_FOLDER, selected_file)
        if selected_file.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif selected_file.endswith(".xlsx"):
            df = pd.read_excel(file_path)

        # Required columns for analysis
        required_columns = ["MemID", "MemGen_x", "MemAge_x", "MemDuration_M_x", "ASPT_x", "MaxSPT_x", "MinSPT_x", 
                            "ANT_x", "APDR_x", "APinFavShop_x", "ATRinFavShop_x", "NGinFavShop_x", 
                            "NFavinFavShop_x"]

        if df is not None:
            if all(col in df.columns for col in required_columns) and len(df.columns) == len(required_columns):
                st.header("🌟 View Analysis Results")

                X = df[['ASPT_x', 'MemGen_x', 'MemAge_x', 'MemDuration_M_x', 'MaxSPT_x', 'MinSPT_x', 'ANT_x', 'APDR_x', 'APinFavShop_x', 'ATRinFavShop_x', 'NGinFavShop_x', 'NFavinFavShop_x']]

                base_model_filenames = [
                    'model/logreg_model_with_features.pkl',      
                    'model/svm_model_with_features.pkl',
                    'model/knn_model_with_features.pkl',   
                    'model/mlp_model_with_features.pkl',   
                    'model/rf_model_with_features.pkl'    
                ]
                stacked_model_filename = 'model/stacked_model.pkl'
                members, scaler, stacking_model = load_all_models(base_model_filenames, stacked_model_filename)
                X_scaled = scaler.transform(X)
                stacked_predictions = stacked_dataset(members, X_scaled, X)
                final_predictions = stacking_model.predict(stacked_predictions)

                fig_all, cluster_all = generate_cluster_radar_chart(X, Ka, 'Clustering Results')

                df['Cluster'] = cluster_all
                df['Predictions'] = final_predictions
                df['User_Importance'] = df['Predictions'].apply(lambda x: 'SVIP' if x == 1 else 'Member')
                df['MemGen_x'] = df['MemGen_x'].apply(lambda x: 'Male' if x == 1 else 'Female')                
                results = df[['MemID', 'MemGen_x', 'MemAge_x', 'MemDuration_M_x', 'User_Importance', 'Cluster']].rename(columns={
                                'MemName': 'Name',
                                'MemGen_x': 'Gender',
                                'MemAge_x': 'Age',
                                'MemDuration_M_x': 'Duration (Months)',
                                'User_Importance': 'Importance'
                            })
                def highlight_svip(row):
                    return ['background-color: yellow' if row['Importance'] == 'SVIP' else '' for _ in row]

                styled_results = results.style.apply(highlight_svip, axis=1).set_table_attributes('style="width:100%; border-collapse: collapse;"')
                
                # 性别选择
                gender_option = st.selectbox("Select Gender:", options=["All", "Male", "Female"])
                importance_option = st.selectbox("Select User Importance:", options=["All", "SVIP", "Member"])

                # 过滤结果
                filtered_results = results.copy()
                if gender_option != "All":
                    filtered_results = filtered_results[filtered_results['Gender'] == gender_option]
                if importance_option != "All":
                    filtered_results = filtered_results[filtered_results['Importance'] == importance_option]

                if not filtered_results.empty:
                    st.dataframe(filtered_results.style.apply(highlight_svip, axis=1).set_table_attributes('style="width:100%; border-collapse: collapse;"'))
                else:
                    st.dataframe(styled_results)
                
                st.plotly_chart(fig_all, use_container_width=True)        

                #性别比例图    
                svip_count = df[df['User_Importance'] == 'SVIP'].shape[0]
                member_count = df[df['User_Importance'] == 'Member'].shape[0]
                total_count = svip_count + member_count

                if total_count > 0:
                    pie_data = pd.DataFrame({
                        'User Type': ['SVIP', 'Member'],
                        'Count': [svip_count, member_count]
                    })
                    color_sequence = ['#ef5350', '#ffee58']
                    fig = px.pie(pie_data, values='Count', names='User Type', title='SVIP and Member Proportions', hole=0.2, color_discrete_sequence=color_sequence)
                    fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=2)))
                    fig.update_layout(legend_title_text='User Type', legend=dict(orientation="h"))
                    st.plotly_chart(fig)
            else:
                st.warning(f"The selected file should only have the following columns: {', '.join(required_columns)}. Please upload a file with the correct format.", icon="⚠️")
        else:
            st.warning("Please select a valid file before clicking 'Analyze'.", icon="⚠️")
else:
    st.warning("Please upload one or more files in the upload files section to get started!", icon="⚠️")