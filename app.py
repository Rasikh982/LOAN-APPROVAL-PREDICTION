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

# ============== CREATE LOAN DATASET (BUILT-IN) ==============
@st.cache_data
def create_loan_dataset():
    """Create a realistic loan approval dataset"""
    np.random.seed(42)
    n = 4269  # Same as Kaggle dataset
    
    # Generate realistic loan data
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
    
    # Create loan approval logic (realistic)
    approval_score = (
        (df['cibil_score'] > 650).astype(float) * 0.40 +
        (df['income_annum'] > 5000000).astype(float) * 0.25 +
        (df['loan_amount'] / df['income_annum'] < 3).astype(float) * 0.20 +
        (df['bank_asset_value'] > 3000000).astype(float) * 0.15
    )
    
    noise = np.random.normal(0, 0.08, n)
    df['loan_status'] = np.where(approval_score + noise > 0.45, 'Approved', 'Rejected')
    
    return df

# ============== LOAD DATASET ==============
if 'df' not in st.session_state:
    with st.spinner('📊 Loading Loan Dataset...'):
        st.session_state.df = create_loan_dataset()
        st.session_state.loaded = True

df = st.session_state.df

# ============== DATASET INFO ==============
st.header("📊 Step 1: Dataset Overview")

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
with col_info1:
    st.metric("Total Records", df.shape[0])
with col_info2:
    st.metric("Total Features", df.shape[1] - 1)
with col_info3:
    st.metric("Missing Values", df.isnull().sum().sum())
with col_info4:
    approved = (df['loan_status'] == 'Approved').sum()
    st.metric("Approval Rate", f"{approved/len(df)*100:.1f}%")

st.subheader("📋 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# Show dataset info
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

# ============== EDA SECTION ==============
st.header("📈 Step 2: Exploratory Data Analysis (EDA)")

target_col = 'loan_status'

# Target Distribution
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
    
    # Add value labels on bars
    for i, v in enumerate(status_counts):
        ax.text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("📊 Class Distribution")
    class_counts = df[target_col].value_counts()
    for cls, count in class_counts.items():
        percentage = count/len(df)*100
        st.metric(f"{cls} Loans", f"{count:,}", f"{percentage:.1f}%")
    
    # Pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    colors_pie = ['#2ecc71', '#e74c3c']
    ax.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
           colors=colors_pie, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title("Approval vs Rejection", fontsize=14, fontweight='bold')
    st.pyplot(fig)
    plt.close()

# Numerical Features Distribution
st.subheader("📊 Numerical Features Distribution")
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

selected_features = st.multiselect(
    "Select features to visualize:",
    num_cols,
    default=['cibil_score', 'income_annum', 'loan_amount']
)

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
                    ax.set_xlabel(col_name, fontsize=9)
                    ax.set_ylabel("Frequency", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', center=0, 
            linewidths=1, ax=ax, fmt='.2f', cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ============== MODEL TRAINING SECTION ==============
st.header("🤖 Step 3: Model Training")

col1, col2 = st.columns(2)

with col1:
    st.info(f"🎯 **Target Column**: **{target_col}**")
    classes = df[target_col].unique()
    st.caption(f"Classes: {', '.join(map(str, classes))}")
    st.success(f"✅ Binary Classification ({len(classes)} classes)")

with col2:
    model_choice = st.selectbox(
        "Select Model:",
        ["🥇 Logistic Regression", "🥈 Decision Tree", "🥉 Support Vector Machine (SVM)"],
        help="Choose the machine learning model"
    )

with st.expander("⚙️ Advanced Settings"):
    test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
    random_state = st.number_input("Random State", 0, 100, 42)

if st.button("🚀 Train Model", type="primary", use_container_width=True):
    try:
        with st.spinner('🔄 Training model... Please wait'):
            # Prepare data
            X = df.drop(columns=[target_col]).copy()
            y = df[target_col].copy()
            
            class_counts = y.value_counts()
            st.info(f"📊 **Class Distribution**: {dict(class_counts)}")
            
            # Handle missing values
            num_cols_X = X.select_dtypes(include=['int64', 'float64']).columns
            cat_cols_X = X.select_dtypes(include=['object']).columns
            
            # Fill missing values
            if len(num_cols_X) > 0:
                X[num_cols_X] = X[num_cols_X].fillna(X[num_cols_X].mean())
            if len(cat_cols_X) > 0:
                for col in cat_cols_X:
                    X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
            
            # Encode categorical features
            le_dict = {}
            for col in cat_cols_X:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
            
            # Encode target
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y.astype(str))
            target_classes = le_target.classes_
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size/100, random_state=random_state, stratify=y_encoded
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select and train model
            if "Logistic" in model_choice:
                model = LogisticRegression(max_iter=1000, random_state=random_state)
                model_name = "Logistic Regression"
            elif "Decision" in model_choice:
                model = DecisionTreeClassifier(random_state=random_state)
                model_name = "Decision Tree"
            else:  # SVM
                model = SVC(random_state=random_state)
                model_name = "Support Vector Machine"
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            st.success("✅ Model trained successfully!")
            
            # ============== RESULTS ==============
            st.markdown("---")
            st.header("📊 Step 4: Model Evaluation Results")
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Display metrics
            st.subheader("📈 Performance Metrics")
            metric_cols = st.columns(5)
            with metric_cols[0]:
                st.metric("Accuracy", f"{accuracy:.2%}")
            with metric_cols[1]:
                st.metric("Precision", f"{precision:.2%}")
            with metric_cols[2]:
                st.metric("Recall", f"{recall:.2%}")
            with metric_cols[3]:
                st.metric("F1-Score", f"{f1:.2%}")
            with metric_cols[4]:
                st.metric("CV Score", f"{cv_scores.mean():.2%}")
            
            # Performance feedback
            if accuracy > 0.85:
                st.success("🎉 **EXCELLENT PERFORMANCE!** Model is ready for deployment!")
            elif accuracy > 0.75:
                st.info("👍 **GOOD PERFORMANCE!** Model works well.")
            elif accuracy > 0.65:
                st.warning("⚠️ **MODERATE PERFORMANCE.** Consider trying different model.")
            else:
                st.error("❌ **POOR PERFORMANCE.** Try a different model or check your data.")
            
            # Print metrics (like Kaggle code)
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
{'='*60}
""")
            
            # Confusion Matrix
            st.subheader("🔍 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                           cbar_kws={"shrink": 0.8}, linewidths=2, annot_kws={"fontsize": 14})
                ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
                ax.set_ylabel('Actual', fontsize=12)
                ax.set_xlabel('Predicted', fontsize=12)
                
                # Add labels
                ax.set_xticklabels(target_classes)
                ax.set_yticklabels(target_classes)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.write("**Confusion Matrix Breakdown:**")
                tn, fp, fn, tp = cm.ravel()
                
                st.metric("True Negatives (TN)", tn, help="Correctly predicted Rejected")
                st.metric("False Positives (FP)", fp, help="Incorrectly predicted Approved")
                st.metric("False Negatives (FN)", fn, help="Incorrectly predicted Rejected")
                st.metric("True Positives (TP)", tp, help="Correctly predicted Approved")
                
                st.write(f"\n**Interpretation:**")
                st.write(f"- ✅ Correct Predictions: {tn + tp} ({(tn+tp)/len(y_test)*100:.1f}%)")
                st.write(f"- ❌ Wrong Predictions: {fp + fn} ({(fp+fn)/len(y_test)*100:.1f}%)")
            
            # Classification Report
            st.subheader("📋 Detailed Classification Report")
            report = classification_report(y_test, y_pred, target_names=target_classes, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen').format("{:.3f}"), 
                        use_container_width=True)
            
            # Feature Importance (for Decision Tree only)
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 Feature Importance Analysis")
                feature_names = X.columns
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Plot top 10 features
                top_features = importance_df.head(10)
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(top_features['Feature'], top_features['Importance'], 
                       color='purple', alpha=0.7)
                ax.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
                ax.set_xlabel('Importance Score', fontsize=12)
                ax.invert_yaxis()
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                           f'{width:.3f}', ha='left', va='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                with st.expander("📊 View All Feature Importances"):
                    st.dataframe(importance_df.style.background_gradient(cmap='Purples'), 
                               use_container_width=True)
            
            st.success("✅ EDA + ML completed successfully!")
    
    except Exception as e:
        st.error(f"❌ **Error during training**: {str(e)}")
        st.error("**Possible Solutions:**")
        st.markdown("- Make sure the data is loaded correctly")
        st.markdown("- Check if there are any data issues")
        st.markdown("- Try a different model")
        
        with st.expander("🔍 See Full Error Details"):
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    🏦 <b>Loan Approval Prediction System</b><br>
    Built with Streamlit 🚀
</div>
""", unsafe_allow_html=True)
