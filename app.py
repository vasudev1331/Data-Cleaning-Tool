import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import io
from data_cleaner import DataCleaner
from sklearn.preprocessing import LabelEncoder

# ---------------------- Page Configuration ----------------------
st.set_page_config(page_title="🧹 The Data Broom", layout="wide")

# ---------------------- Sidebar ----------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/10179/10179118.png", width=120)
st.sidebar.title("🧹 The Data Broom")
st.sidebar.markdown("### Clean. Explore. Prepare.")



# File Upload
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    filepath = os.path.join("uploaded.csv")
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "cleaner" not in st.session_state:
        st.session_state.cleaner = DataCleaner(filepath)

    cleaner = st.session_state.cleaner

    # Main Header
    st.markdown("""
        <div style="background-color:#0066cc;padding:20px;border-radius:12px;margin-bottom:20px">
        <h1 style="color:white;text-align:center;">🧹 The Data Broom</h1>
        <h3 style="color:white;text-align:center;">Next Level Cleaning & EDA TaaS Tool</h3>
        </div>
        """, unsafe_allow_html=True)

    # Tabs
    tabs = st.tabs(["📊 Summary", "🔍 Preview", "🛠 Manual Clean", "⚡ Auto Clean", "📈 EDA", "📉 Outliers", "💾 Download"])

    # Summary Report
    with tabs[0]:
        st.subheader("📊 Data Summary Report")
        st.write(f"**Shape:** {cleaner.data.shape[0]} rows, {cleaner.data.shape[1]} columns")
        st.write(f"**Duplicate Rows:** {cleaner.data.duplicated().sum()}")
        st.write("**Missing Values:**")
        st.dataframe(cleaner.show_missing())
        st.write("**Column Types:**")
        st.write(cleaner.data.dtypes.value_counts())
        st.write("**Descriptive Stats:**")
        st.dataframe(cleaner.data.describe())

    # Preview
    with tabs[1]:
        st.subheader("🔍 Dataset Preview")
        st.dataframe(cleaner.get_data().head())

    # Manual Clean
    with tabs[2]:
        st.subheader("🛠 Manual Cleaning")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Remove Duplicates"):
                cleaner.remove_duplicates()
                st.success("✅ Duplicates removed")

            if st.button("Standardize Column Names"):
                cleaner.standardize_columns()
                st.success("✅ Column names standardized")

        with col2:
            method = st.selectbox("Fill Missing (numeric)", ["mean", "median"])
            if st.button("Fill Missing"):
                cleaner.handle_missing(method)
                st.success(f"✅ Missing filled with {method}")

        with col3:
            encoding = st.radio("Encoding", ["Label Encoding", "One-Hot Encoding"])
            if st.button("Encode Categorical Columns"):
                if encoding == "Label Encoding":
                    cleaner.encode_categoricals()
                else:
                    cleaner.data = pd.get_dummies(cleaner.data)
                st.success(f"✅ {encoding} applied")

    # Auto Clean
    with tabs[3]:
        st.subheader("⚡ Auto Clean")
        if st.button("Run Auto Clean 🧹"):
            with st.spinner("Running full Auto Cleaning..."):
                cleaner.auto_clean()
            st.success("✅ Auto Cleaning Complete")

    # EDA
    with tabs[4]:
        st.subheader("📈 Exploratory Data Analysis")

        st.write("Missing Values Heatmap")
        plt.figure(figsize=(10, 5))
        sns.heatmap(cleaner.data.isnull(), cbar=False, cmap='plasma')
        st.pyplot(plt.gcf())
        plt.clf()

        st.write("Correlation Heatmap")
        plt.figure(figsize=(10, 5))
        corr = cleaner.data.select_dtypes(include=np.number).corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot(plt.gcf())
        plt.clf()

        numeric_cols = cleaner.data.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            col = st.selectbox("Distribution Plot", numeric_cols)
            plt.figure(figsize=(8, 4))
            sns.histplot(cleaner.data[col], kde=True, color="#007bff")
            st.pyplot(plt.gcf())
            plt.clf()

    # Outlier Removal
    with tabs[5]:
        st.subheader("📉 Outlier Detection & Removal")
        numeric_cols = cleaner.data.select_dtypes(include=np.number).columns
        col_to_check = st.multiselect("Select columns", numeric_cols)

        def remove_outliers_iqr(df, cols):
            for col in cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
            return df

        if st.button("Remove Outliers"):
            cleaner.data = remove_outliers_iqr(cleaner.data, col_to_check)
            st.success("✅ Outliers removed")

    # Download
    with tabs[6]:
        st.subheader("💾 Download Cleaned Dataset")
        cleaner.save()
        with open("cleaned_output.csv", "rb") as f:
            st.download_button("📥 Download CSV", f, file_name="cleaned_output.csv")

else:
    st.markdown("""
        <div style="background-color:#fde68a;padding:20px;border-radius:12px;margin-top:20px;">
        <h2 style="color:#92400e;text-align:center;">👋 Upload a CSV file to get started</h2>
        </div>
    """, unsafe_allow_html=True)
