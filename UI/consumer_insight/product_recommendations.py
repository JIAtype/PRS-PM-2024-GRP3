import streamlit as st
import pandas as pd
import joblib
import os

# Load the model
loaded_rules = joblib.load('model/association_rules.pkl')

# Streamlit app title
st.title("Product Recommendations for Consumer")

if st.session_state.role == "Admin":
    UPLOAD_FOLDER = "a_data"
elif st.session_state.role == "Clerk":
    UPLOAD_FOLDER = "c_data"

# Check if the upload folder exists and contains files
if os.path.isdir(UPLOAD_FOLDER) and os.listdir(UPLOAD_FOLDER):
    uploaded_files = os.listdir(UPLOAD_FOLDER)
    uploaded_files = ["Select File"] + uploaded_files
    selected_file = st.selectbox("Please select the file to visualize here:", uploaded_files)
    df = None
    if selected_file != "Select File":  # Only read data if a valid file is selected
        file_path = os.path.join(UPLOAD_FOLDER, selected_file)
        if selected_file.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif selected_file.endswith(".xlsx"):
            df = pd.read_excel(file_path)
else:
    st.warning("Please upload one or more files in the upload files section to get started!", icon="⚠️")

required_columns = ["MemID", "MemGen_x", "MemAge_x", "MemDuration_M_x", "ASPT_x", "MaxSPT_x", "MinSPT_x", 
                    "ANT_x", "APDR_x", "APinFavShop_x", "ATRinFavShop_x", "NGinFavShop_x", "NFavinFavShop_x", 
                    "MemGen_y", "MemAge_y", "MemDuration_M_y", "ASPT_y", "MaxSPT_y", "MinSPT_y", 
                    "ANT_y", "APDR_y", "APinFavShop_y", "ATRinFavShop_y", "NGinFavShop_y", "NFavinFavShop_y", 
                    "ProdName"]

if st.button("Analyze"):
    if df is not None:
        if all(col in df.columns for col in required_columns):
            trans = df.copy()
            memid_column = [col for col in trans.columns if 'MemID' in col][0]
            prodname_column = [col for col in trans.columns if 'ProdName' in col][0]
            trans = trans[[memid_column, prodname_column]]
            trans.columns = ['User', 'Item']  # Rename to User and Item

            # Preprocess items
            def preprocess_item(item):
                return [i.strip() for i in item.split(',')]

            trans['Item'] = trans['Item'].apply(preprocess_item)
            trans = trans.explode('Item')
            baskets = trans.groupby('User')['Item'].apply(list)

            # Display the total number of rules
            st.subheader(f"Total Number of Rules: {len(loaded_rules)}")

            # Function to execute rules
            def execrules_anymatch(itemset, rules, topN=10):
                preds = {}
                for LHS, RHS, conf in rules:
                    if LHS.issubset(itemset):
                        for pitem in RHS:
                            if pitem not in itemset:
                                preds[pitem] = max(preds.get(pitem, 0), conf)
                recs = sorted(preds.items(), key=lambda kv: kv[1], reverse=True)
                return recs[:topN]

            # Iterate through each user and display recommendations
            for userID in range(len(baskets)):
                basket = set(baskets.iloc[userID])
                st.markdown(f"### User ID: {baskets.index[userID]}")
                st.markdown(f"**User's Basket:** {basket}")
                recommended_items = execrules_anymatch(basket, loaded_rules)
                st.markdown("**Recommended Items:**")
                for item, conf in recommended_items:
                    st.markdown(f"- {item} (Confidence: {conf:.2f})")
                st.markdown("---")  # Separator for clarity between user outputs
        else:
            st.warning(f"The selected file does not contain the required columns: {', '.join(required_columns)}. Please upload a file with the correct format.", icon="⚠️")
    else:
        st.warning("Please select a valid file before clicking 'Analyze'.", icon="⚠️")