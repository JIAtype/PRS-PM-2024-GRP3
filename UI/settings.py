import streamlit as st
from datetime import datetime

current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

# Set the page title
st.title("⚙️ Settings")

st.header(f"Welcome {st.session_state.role}！")
st.write(f"You logged in to Xsell at {current_time}.")

st.header("Account Information")

# Username input
username = st.text_input("Username", value=st.session_state.role, max_chars=20)

# Password input
password = st.text_input("Password", type="password", value="123456")

# Other settings
st.header("Other Settings")
theme = st.selectbox("Theme", options=["Light", "Dark"])
notifications = st.checkbox("Enable Notifications", value=st.session_state.get("notifications", True))

# Save settings button
if st.button("Save Settings"):
    # Save settings to session_state
    st.session_state.username = username
    st.session_state.password = password
    st.session_state.theme = theme
    st.session_state.notifications = notifications

    st.success("Settings saved!")

# Display current settings
st.subheader("Current Settings")
st.write(f"Username: {st.session_state.get('username', 'Not set')}")
st.write(f"Theme: {st.session_state.get('theme', 'Not set')}")
st.write(f"Enable Notifications: {'Yes' if st.session_state.get('notifications', False) else 'No'}")