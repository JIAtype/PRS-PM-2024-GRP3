import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

dt = joblib.load('model/decision_tree_model.pkl')
loaded_rules = joblib.load('model/association_rules.pkl')

st.title("🛍️ Consumer Insight")
st.markdown("""
    Welcome to the SVIP Spotlight application!  
    Here, you can analyze SVIP users and obtain tailored product recommendations based on their shopping behavior.
""")

UPLOAD_FOLDER = "data"
# File upload section
if os.path.isdir(UPLOAD_FOLDER) and os.listdir(UPLOAD_FOLDER):
    uploaded_files = os.listdir(UPLOAD_FOLDER)
    uploaded_files = ["Select File"] + uploaded_files
    selected_file = st.selectbox("Please select the file to visualize here:", uploaded_files)
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
                            "NFavinFavShop_x", "MemGen_y", "MemAge_y", "MemDuration_M_y", "ASPT_y", 
                            "MaxSPT_y", "MinSPT_y", "ANT_y", "APDR_y", "APinFavShop_y", "ATRinFavShop_y", 
                            "NGinFavShop_y", "NFavinFavShop_y", "ProdName"]

        # Analysis button with style
        # if st.button("🔍 Analyze", key="analyze_button"):
        st.header("🌟 SVIP Spotlight and Product Recommendations")
        if df is not None:
            if all(col in df.columns for col in required_columns):
                # Data preprocessing
                df['MemAge_y_binned'] = pd.cut(df['MemAge_y'], bins=[15, 20, 30, 40, 50, 60, 70, 85], labels=['15-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-85'], include_lowest=True)
                df['MemAge_y_binned_code'] = df['MemAge_y_binned'].cat.codes
                
                # Binning logic for ASPT_x and ANT_x
                top_30_asptx_threshold = 2113.916
                top_30_antx_threshold = 0.363636364
                
                df['ASPT_x_binned_code'] = df['ASPT_x'].apply(lambda x: 1 if x >= top_30_asptx_threshold else 0)
                df['ANT_x_binned_code'] = df['ANT_x'].apply(lambda x: 1 if x >= top_30_antx_threshold else 0)

                features_to_extract = ['MemAge_y_binned_code', 'ASPT_x_binned_code', 'ANT_x_binned_code']
                if all(feature in df.columns for feature in features_to_extract):
                    data_points = df[features_to_extract].values
                    predictions = dt.predict(data_points)
                    df['Predicted_Class'] = predictions
                    df['User_Importance'] = df['Predicted_Class'].apply(lambda x: 'SVIP' if x == 1 else 'Member')
                    df['MemGen_x'] = df['MemGen_x'].apply(lambda x: 'Male' if x == 1 else 'Female')

                    results = df[['MemID', 'MemName', 'MemGen_x', 'MemAge_x', 'MemDuration_M_x', 'User_Importance']].rename(columns={
                        'MemName': 'Name',
                        'MemGen_x': 'Gender',
                        'MemAge_x': 'Age',
                        'MemDuration_M_x': 'Duration (Months)',
                        'User_Importance': 'Importance'
                    })

                    # Product recommendations
                    trans = df.copy()
                    trans['Item'] = trans['ProdName'].apply(lambda item: [i.strip() for i in item.split(',')])
                    trans = trans.explode('Item')
                    baskets = trans.groupby('MemID')['Item'].apply(list)

                    def execrules_anymatch(itemset, rules, topN=10):
                        preds = {}
                        for LHS, RHS, conf in rules:
                            if LHS.issubset(itemset):
                                for pitem in RHS:
                                    if pitem not in itemset:
                                        preds[pitem] = max(preds.get(pitem, 0), conf)
                        recs = sorted(preds.items(), key=lambda kv: kv[1], reverse=True)
                        return recs[:topN]

                    recommendations = []
                    for userID in range(len(baskets)):
                        basket = set(baskets.iloc[userID])
                        user_id = baskets.index[userID]
                        recommended_items = execrules_anymatch(basket, loaded_rules)
                        recommended_items_str = '\n'.join([item for item, _ in recommended_items])
                        recommendations.append({'MemID': user_id, 'Recommended Items': recommended_items_str})

                    recommendations_df = pd.DataFrame(recommendations)
                    final_results = pd.merge(results, recommendations_df, on='MemID', how='left')

                    # Highlight SVIP users
                    # def highlight_svip(row):
                    #     return ['background-color: yellow' if row['Importance'] == 'SVIP' else '' for _ in row]
                    def highlight_svip(row):
                        return ['background-color: yellow' if row['Importance'] == 'SVIP' else '' for _ in row]

                    # styled_results = final_results.style.apply(highlight_svip, axis=1).set_table_attributes('style="overflow-x:auto;"')
                    styled_results = final_results.style.apply(highlight_svip, axis=1).set_table_attributes('style="width:100%; border-collapse: collapse;"')
                    # Show styled dataframe
                    # st.subheader("Analysis Results")
                    st.dataframe(styled_results)

                    # Pie chart for SVIP proportions
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
                    st.warning("Data is missing one or more of the features needed for analysis.", icon="⚠️")
            else:
                st.warning(f"The selected file does not contain the required columns: {', '.join(required_columns)}. Please upload a file with the correct format.", icon="⚠️")
        else:
            st.warning("Please select a valid file before clicking 'Analyze'.", icon="⚠️")


else:
    st.warning("Please upload one or more files in the upload files section to get started!", icon="⚠️")

