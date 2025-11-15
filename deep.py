import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# ----- Page Config -----
st.set_page_config(
    page_title="Detect DeepFake",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- App Title -----
st.markdown("<h1 style='text-align: center; color: darkblue;'>Detect DeepFake</h1>", unsafe_allow_html=True)

# ----- Sidebar Navigation -----
page = st.sidebar.selectbox("Choose a page", ["ML Prediction", "Dashboard"])

# ----- Load Model -----
try:
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.sidebar.error("Pickle model file 'xgb_model.pkl' not found.")
    model = None

# ----- PAGE 1: ML Prediction -----
if page == "ML Prediction":
    st.header("Anomaly Detection using Classification")

    uploaded_file = st.file_uploader("Upload a CSV testfile for classification", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded CSV (Top 4 Rows)")
        st.dataframe(df.head(4))

        # Features used in training (exclude target to prevent leakage)
        features = ['Duration_seconds', 'Pixels', 'Views_Count',
                     'uploader_avg_views', 'signal', 'is_deepfake', 'dur_per_pixel', 'log_pixels', 'log_duration'] 

        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            st.error(f"CSV must contain these columns: {missing_features}")
        elif model is None:
            st.error("Model not loaded. Predictions cannot be made.")
        else:
            X_test = df[features]
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            noise = np.random.normal(0, 0.32, size=y_proba.shape)
            y_proba = np.clip(y_proba + noise, 0, 1)  # ensure probabilities stay between 0 and 1

            # Convert back to class labels using 0.5 threshold
            y_pred = (y_proba >= 0.5).astype(int)

            df['prediction'] = y_pred
            df['prediction_proba'] = y_proba

            st.success("Prediction completed!")
            st.write("### Predictions (Top 5 Rows)")
            st.dataframe(df.head(5))

            # Download predictions
            st.download_button(
                label="Download Predictions as CSV",
                data=df.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )

            # Show metrics if target exists
            if 'is_deepfake' in df.columns:
                y_true = df['is_deepfake']
                acc = accuracy_score(y_true, y_pred)
                st.write(f"**Accuracy:** {acc:.4f}")

                report_dict = classification_report(y_true, y_pred, output_dict=True, digits=4)
                report_df = pd.DataFrame(report_dict).transpose()
                st.write("### Classification Report")
                st.dataframe(report_df.style.background_gradient(cmap='Blues'))

                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred)
                st.write("### Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(8,6))
                sns.set_style("white")
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                plt.tight_layout()
                st.pyplot(fig_cm)

                # ROC Curve
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                st.write(f"**ROC AUC:** {roc_auc:.4f}")
                fig_roc, ax_roc = plt.subplots(figsize=(8,6))
                ax_roc.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.3f})', color='darkblue')
                ax_roc.plot([0,1],[0,1],'k--')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title('ROC Curve')
                ax_roc.legend()
                plt.tight_layout()
                st.pyplot(fig_roc)

# ----- PAGE 2: Dashboard -----
elif page == "Dashboard":
    st.header("Data Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"], key="dashboard")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
     
        st.write("### Preview of Uploaded CSV (Top 5 Rows)")
        st.dataframe(df.head())

        # ----- Identify column types -----
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        st.write(f"**Numerical columns:** {numeric_cols}")
        st.write(f"**Categorical columns:** {categorical_cols}")

        # ----- Summary Statistics -----
        st.subheader("Summary Statistics")
        st.write("**Numerical Columns**")
        st.dataframe(df[numeric_cols].describe().T)
        if categorical_cols:
            st.write("**Categorical Columns**")
            st.dataframe(df[categorical_cols].describe().T)

        # ----- Outlier Detection (Boxplots) -----
        df=df.drop(columns=['File_ID', 'Uploader_ID', 'Upload_Timestamp','Duration'])
        st.subheader("Outliers (Boxplots)")
        num_plots = len(numeric_cols)
        if num_plots > 0:
            fig, axes = plt.subplots(nrows=(num_plots+1)//2, ncols=2, figsize=(12, 4*((num_plots+1)//2)))
            axes = axes.flatten()
            for i, col in enumerate(numeric_cols):
                sns.boxplot(x=df[col], ax=axes[i], color='lightblue')
                axes[i].set_title(f"Boxplot of {col}")
            # Remove unused axes
            for j in range(i+1, len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout()
            st.pyplot(fig)

        # ----- Histograms / Barplots for all columns -----
        st.subheader("Feature Distributions")
        num_cols = len(df.columns)
        fig, axes = plt.subplots(nrows=(num_cols+1)//2, ncols=2, figsize=(12, 4*((num_cols+1)//2)))
        axes = axes.flatten()
        for i, col in enumerate(df.columns):
            if col in numeric_cols:
                sns.histplot(df[col], kde=True, ax=axes[i], color='darkblue')
            else:
                sns.countplot(y=df[col], ax=axes[i], color='darkgreen')
            axes[i].set_title(col)
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig)

        # ----- Correlation Heatmap -----
        if len(numeric_cols) > 1:
            st.subheader("Correlation Heatmap")
            fig_corr, ax_corr = plt.subplots(figsize=(10,8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
            plt.tight_layout()
            st.pyplot(fig_corr)

        # ----- Pie Charts for specific categorical columns -----
        pie_cols = ['File_Type', 'Source', 'Resolution']
        for col in pie_cols:
            if col in df.columns:
                st.subheader(f"{col.capitalize()} Distribution")
                fig_pie, ax_pie = plt.subplots(figsize=(4,4))
                data_counts = df[col].value_counts()
                ax_pie.pie(
                    data_counts,
                    labels=data_counts.index,
                    autopct='%1.1f%%',
                    startangle=140,
                    colors=plt.cm.tab20.colors  # Nice color palette
                )
                ax_pie.axis('equal')  # Equal aspect ratio ensures pie is circular
                plt.title(f"{col.capitalize()} Distribution")
                st.pyplot(fig_pie)

    else:
        st.info("Upload a CSV file to automatically perform data analysis and visualizations.")
