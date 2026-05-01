import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

plt.switch_backend('Agg')

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")
st.title("🏦 Loan Approval Prediction System")
st.markdown("---")

@st.cache_data
def create_loan_dataset():
    np.random.seed(42)
    n = 4269
    df = pd.DataFrame({
        'no_of_dependents': np.random.randint(0, 6, n),
        'education': np.random.choice(['Graduate', 'Not Graduate'], n, p=[0.78, 0.22]),
        'self_employed': np.random.choice(['No', 'Yes'], n, p=[0.86, 0.14]),
        'income_annum': np.random.randint(200000, 9900001, n),
        'loan_amount': np.random.randint(300000, 39500001, n),
        'loan_term': np.random.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], n),
        'cibil_score': np.random.randint(300, 901, n),
        'commercial_assets_value': np.random.randint(0, 19400001, n),
        'luxury_assets_value': np.random.randint(300000, 39200001, n),
        'bank_asset_value': np.random.randint(0, 14700001, n)
    })
    approval_score = (
        (df['cibil_score'] > 650).astype(float) * 0.40 +
        (df['income_annum'] > 5000000).astype(float) * 0.25 +
        (df['loan_amount'] / df['income_annum'] < 3).astype(float) * 0.20 +
        (df['bank_asset_value'] > 3000000).astype(float) * 0.15
    )
    noise = np.random.normal(0, 0.08, n)
    df['loan_status'] = np.where(approval_score + noise > 0.45, 'Approved', 'Rejected')
    return df

if 'df' not in st.session_state:
    with st.spinner('Loading Loan Dataset...'):
        st.session_state.df = create_loan_dataset()
        st.session_state.loaded = True

df = st.session_state.df

# ====== STEP 1: DATASET OVERVIEW ======
st.header("📊 Step 1: Dataset Overview")
col_info1, col_info2, col_info3, col_info4 = st.columns(4)
with col_info1: st.metric("Total Records", df.shape[0])
with col_info2: st.metric("Total Features", df.shape[1] - 1)
with col_info3: st.metric("Missing Values", df.isnull().sum().sum())
with col_info4:
    approved = (df['loan_status'] == 'Approved').sum()
    st.metric("Approval Rate", f"{approved/len(df)*100:.1f}%")

st.subheader("📋 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

with st.expander("📋 Dataset Information"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Column Names:**")
        for col in df.columns:
            st.write(f"- {col}")
    with col2:
        st.write("**Data Types:**")
        st.dataframe(df.dtypes, use_container_width=True)
    st.write("\n**Statistical Summary:**")
    st.dataframe(df.describe(), use_container_width=True)

st.markdown("---")

# ====== STEP 2: EDA ======
st.header("📈 Step 2: Exploratory Data Analysis (EDA)")
target_col = 'loan_status'

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Loan Status Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    status_counts = df[target_col].value_counts()
    colors = ['#2ecc71' if x == 'Approved' else '#e74c3c' for x in status_counts.index]
    status_counts.plot(kind='bar', color=colors, ax=ax)
    ax.set_title("Loan Approval Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Loan Status", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.xticks(rotation=0)
    for i, v in enumerate(status_counts):
        ax.text(i, v + 50, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("📊 Class Distribution")
    class_counts = df[target_col].value_counts()
    for cls, count in class_counts.items():
        st.metric(f"{cls} Loans", f"{count:,}", f"{count/len(df)*100:.1f}%")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
           colors=['#2ecc71', '#e74c3c'], startangle=90,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title("Approval vs Rejection", fontsize=14, fontweight='bold')
    st.pyplot(fig)
    plt.close()

st.subheader("📊 Numerical Features Distribution")
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
selected_features = st.multiselect("Select features to visualize:", num_cols,
                                    default=['cibil_score', 'income_annum', 'loan_amount'])
if selected_features:
    cols_per_row = 3
    num_rows = (len(selected_features) + cols_per_row - 1) // cols_per_row
    for row in range(num_rows):
        cols = st.columns(cols_per_row)
        for i in range(cols_per_row):
            idx = row * cols_per_row + i
            if idx < len(selected_features):
                col_name = selected_features[idx]
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    df[col_name].hist(bins=20, color='skyblue', edgecolor='black', ax=ax)
                    ax.set_title(f"{col_name}", fontsize=11, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

st.subheader("🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', center=0,
            linewidths=1, ax=ax, fmt='.2f', cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()
st.markdown("---")

# ====== STEP 3: MODEL TRAINING ======
st.header("🤖 Step 3: Model Training")
col1, col2 = st.columns(2)
with col1:
    st.info(f"🎯 **Target Column**: **{target_col}**")
    classes = df[target_col].unique()
    st.caption(f"Classes: {', '.join(map(str, classes))}")
    st.success(f"✅ Binary Classification ({len(classes)} classes)")
with col2:
    model_choice = st.selectbox("Select Model:",
        ["🥇 Logistic Regression", "🥈 Decision Tree", "🥉 Support Vector Machine (SVM)"])

with st.expander("⚙️ Advanced Settings"):
    test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
    random_state = st.number_input("Random State", 0, 100, 42)

if st.button("🚀 Train Model", type="primary", use_container_width=True):
    try:
        with st.spinner('Training model... Please wait'):
            X = df.drop(columns=[target_col]).copy()
            y = df[target_col].copy()
            st.info(f"📊 **Class Distribution**: {dict(y.value_counts())}")

            num_cols_X = X.select_dtypes(include=['int64', 'float64']).columns
            cat_cols_X = X.select_dtypes(include=['object']).columns

            if len(num_cols_X) > 0:
                X[num_cols_X] = X[num_cols_X].fillna(X[num_cols_X].mean())
            if len(cat_cols_X) > 0:
                for col in cat_cols_X:
                    X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')

            le_dict = {}
            for col in cat_cols_X:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le

            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y.astype(str))
            target_classes = le_target.classes_

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size/100, random_state=random_state, stratify=y_encoded)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)

            if "Logistic" in model_choice:
                model = LogisticRegression(max_iter=1000, random_state=random_state)
                model_name = "Logistic Regression"
            elif "Decision" in model_choice:
                model = DecisionTreeClassifier(random_state=random_state)
                model_name = "Decision Tree"
            else:
                model = SVC(random_state=random_state)
                model_name = "Support Vector Machine"

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')

            # Save for Step 5
            st.session_state.trained_model     = model
            st.session_state.trained_scaler    = scaler
            st.session_state.trained_le_dict   = le_dict
            st.session_state.trained_le_target = le_target
            st.session_state.model_name        = model_name
            st.session_state.model_trained     = True

            st.success("✅ Model trained successfully!")
            st.markdown("---")
            st.header("📊 Step 4: Model Evaluation Results")

            accuracy  = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall    = recall_score(y_test, y_pred, zero_division=0)
            f1        = f1_score(y_test, y_pred, zero_division=0)

            st.subheader("📈 Performance Metrics")
            mc = st.columns(5)
            with mc[0]: st.metric("Accuracy",  f"{accuracy:.2%}")
            with mc[1]: st.metric("Precision", f"{precision:.2%}")
            with mc[2]: st.metric("Recall",    f"{recall:.2%}")
            with mc[3]: st.metric("F1-Score",  f"{f1:.2%}")
            with mc[4]: st.metric("CV Score",  f"{cv_scores.mean():.2%}")

            if accuracy > 0.85:   st.success("🎉 Excellent Performance! Model is ready for deployment!")
            elif accuracy > 0.75: st.info("👍 Good Performance! Model works well.")
            elif accuracy > 0.65: st.warning("⚠️ Moderate Performance. Consider trying a different model.")
            else:                 st.error("❌ Poor Performance. Try a different model or check your data.")

            st.subheader("📋 Model Performance Summary")
            st.code(f"""
{model_name}
{'='*60}
Accuracy  : {accuracy:.4f}
Precision : {precision:.4f}
Recall    : {recall:.4f}
F1 Score  : {f1:.4f}
{'='*60}
Confusion Matrix:
{confusion_matrix(y_test, y_pred)}
""")

            st.subheader("🔍 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           linewidths=2, annot_kws={"fontsize": 14})
                ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
                ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
                ax.set_xticklabels(target_classes); ax.set_yticklabels(target_classes)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col2:
                tn, fp, fn, tp = cm.ravel()
                st.metric("True Negatives (TN)",  tn)
                st.metric("False Positives (FP)", fp)
                st.metric("False Negatives (FN)", fn)
                st.metric("True Positives (TP)",  tp)
                st.write(f"- ✅ Correct Predictions: {tn+tp} ({(tn+tp)/len(y_test)*100:.1f}%)")
                st.write(f"- ❌ Wrong Predictions:   {fp+fn} ({(fp+fn)/len(y_test)*100:.1f}%)")

            st.subheader("📋 Detailed Classification Report")
            report = classification_report(y_test, y_pred, target_names=target_classes,
                                           output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen').format("{:.3f}"),
                        use_container_width=True)

            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 Feature Importance Analysis")
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                top_features = importance_df.head(10)
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(top_features['Feature'], top_features['Importance'],
                               color='purple', alpha=0.7)
                ax.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
                ax.set_xlabel('Importance Score'); ax.invert_yaxis()
                for bar in bars:
                    w = bar.get_width()
                    ax.text(w, bar.get_y()+bar.get_height()/2, f'{w:.3f}',
                            ha='left', va='center', fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()
                with st.expander("📊 View All Feature Importances"):
                    st.dataframe(importance_df.style.background_gradient(cmap='Purples'),
                                use_container_width=True)

            st.success("✅ EDA + ML Pipeline completed successfully!")

    except Exception as e:
        st.error(f"❌ Error during training: {str(e)}")
        with st.expander("🔍 See Full Error Details"):
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")

# ====== STEP 5: LIVE PREDICTION ======
st.header("🎯 Step 5: Predict Your Loan Approval")

if not st.session_state.get('model_trained', False):
    st.warning("⚠️ Please train a model in Step 3 first before making predictions!")
else:
    st.success(f"✅ Using trained model: **{st.session_state.model_name}**")
    st.markdown("Fill in your details below to check your loan eligibility:")

    with st.form("prediction_form"):
        st.subheader("👤 Personal Information")
        col1, col2, col3 = st.columns(3)
        with col1: p_dependents    = st.selectbox("Number of Dependents", [0,1,2,3,4,5])
        with col2: p_education     = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with col3: p_self_employed = st.selectbox("Self Employed", ["No", "Yes"])

        st.subheader("💰 Financial Information")
        col4, col5, col6 = st.columns(3)
        with col4: p_income      = st.number_input("Annual Income (Rs.)", min_value=200000, max_value=9900000, value=500000, step=50000)
        with col5: p_loan_amount = st.number_input("Loan Amount (Rs.)", min_value=300000, max_value=39500000, value=5000000, step=100000)
        with col6: p_loan_term   = st.selectbox("Loan Term (years)", [2,4,6,8,10,12,14,16,18,20])

        st.subheader("📊 Credit & Assets")
        col7, col8, col9, col10 = st.columns(4)
        with col7:  p_cibil      = st.slider("CIBIL Score", 300, 900, 650)
        with col8:  p_commercial = st.number_input("Commercial Assets (Rs.)", min_value=0, max_value=19400000, value=0, step=100000)
        with col9:  p_luxury     = st.number_input("Luxury Assets (Rs.)", min_value=300000, max_value=39200000, value=1000000, step=100000)
        with col10: p_bank       = st.number_input("Bank Assets (Rs.)", min_value=0, max_value=14700000, value=500000, step=100000)

        submitted = st.form_submit_button("🔍 Predict My Loan Status", use_container_width=True, type="primary")

    if submitted:
        try:
            le_dict   = st.session_state.trained_le_dict
            scaler    = st.session_state.trained_scaler
            model     = st.session_state.trained_model
            le_target = st.session_state.trained_le_target

            edu_enc = le_dict['education'].transform([p_education])[0]        if 'education'     in le_dict else (1 if p_education=="Graduate" else 0)
            emp_enc = le_dict['self_employed'].transform([p_self_employed])[0] if 'self_employed' in le_dict else (1 if p_self_employed=="Yes" else 0)

            input_data   = np.array([[p_dependents, edu_enc, emp_enc, p_income, p_loan_amount, p_loan_term, p_cibil, p_commercial, p_luxury, p_bank]])
            input_scaled = scaler.transform(input_data)
            prediction   = model.predict(input_scaled)[0]
            result_label = le_target.inverse_transform([prediction])[0]

            st.markdown("---")
            st.subheader("🏆 Prediction Result")

            if result_label == "Approved":
                st.success("## ✅ Loan APPROVED!")
                st.balloons()
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Decision", "✅ Approved")
                with c2: st.metric("CIBIL Score", p_cibil)
                with c3: st.metric("Annual Income", f"Rs. {p_income:,}")
                st.info("🎉 Congratulations! Based on your profile, your loan is likely to be approved.")
            else:
                st.error("## ❌ Loan REJECTED")
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Decision", "❌ Rejected")
                with c2: st.metric("CIBIL Score", p_cibil)
                with c3: st.metric("Annual Income", f"Rs. {p_income:,}")
                st.warning("💡 Tips to improve your approval chances:")
                st.markdown("""
- 📈 **Improve your CIBIL Score** — aim for 700+
- 💰 **Increase your income** or reduce the loan amount
- 🏦 **Increase your bank assets**
- 📉 **Reduce your Loan-to-Income ratio** (loan should be less than 3x annual income)
""")

            with st.expander("📋 View Your Input Summary"):
                summary = {
                    "Dependents": p_dependents, "Education": p_education,
                    "Self Employed": p_self_employed, "Annual Income (Rs.)": f"{p_income:,}",
                    "Loan Amount (Rs.)": f"{p_loan_amount:,}", "Loan Term (years)": p_loan_term,
                    "CIBIL Score": p_cibil, "Commercial Assets (Rs.)": f"{p_commercial:,}",
                    "Luxury Assets (Rs.)": f"{p_luxury:,}", "Bank Assets (Rs.)": f"{p_bank:,}",
                }
                st.table(pd.DataFrame(summary.items(), columns=["Feature", "Value"]))

        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    🏦 <b>Loan Approval Prediction System</b><br>
    Built with Streamlit 🚀
</div>
""", unsafe_allow_html=True)
