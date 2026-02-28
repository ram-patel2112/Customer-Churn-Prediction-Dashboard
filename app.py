import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# GLOBAL STYLING
# ---------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3, h4 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Customer Churn Intelligence Dashboard")
st.markdown("Advanced churn insights powered by analytics and machine learning.")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("European_Bank.csv")

df = load_data()

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------
if "RelationshipScore" not in df.columns:
    df["RelationshipScore"] = (
        df["IsActiveMember"] +
        df["NumOfProducts"] +
        df["HasCrCard"]
    )

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.header("🔎 Filters")

active_filter = st.sidebar.selectbox(
    "Active Status",
    ["All", "Active Only", "Inactive Only"]
)

if active_filter == "Active Only":
    df = df[df["IsActiveMember"] == 1]
elif active_filter == "Inactive Only":
    df = df[df["IsActiveMember"] == 0]

st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard identifies churn drivers based on engagement, "
    "product usage, and relationship strength."
)

# ---------------------------------------------------
# KPI CALCULATIONS
# ---------------------------------------------------
overall_churn = df["Exited"].mean() * 100
active_churn = df[df["IsActiveMember"] == 1]["Exited"].mean() * 100
inactive_churn = df[df["IsActiveMember"] == 0]["Exited"].mean() * 100

premium_threshold = df["Balance"].quantile(0.75)
premium_df = df[df["Balance"] >= premium_threshold]
premium_churn = premium_df["Exited"].mean() * 100

high_balance_disengagement = (
    premium_df["IsActiveMember"].value_counts(normalize=True).get(0, 0) * 100
)

sticky_df = df[df["RelationshipScore"] >= 4]
non_sticky_df = df[df["RelationshipScore"] < 4]

sticky_churn = sticky_df["Exited"].mean() * 100
non_sticky_churn = non_sticky_df["Exited"].mean() * 100

# ---------------------------------------------------
# KPI CARDS
# ---------------------------------------------------
st.subheader("📌 Key Performance Indicators")

def kpi_card(title, value, color):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}, #111827);
            padding:25px;
            border-radius:15px;
            text-align:center;
            box-shadow: 0 6px 18px rgba(0,0,0,0.4);
            transition: 0.3s;
        ">
            <h4 style="margin-bottom:8px; color:white;">{title}</h4>
            <h2 style="margin-top:0; color:white;">{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

k1, k2, k3, k4 = st.columns(4)
with k1: kpi_card("Overall Churn", f"{overall_churn:.2f}%", "#1f77b4")
with k2: kpi_card("Active Churn", f"{active_churn:.2f}%", "#2ca02c")
with k3: kpi_card("Inactive Churn", f"{inactive_churn:.2f}%", "#d62728")
with k4: kpi_card("Premium Churn", f"{premium_churn:.2f}%", "#9467bd")

k5, k6, k7 = st.columns(3)
with k5: kpi_card("High-Balance Disengagement", f"{high_balance_disengagement:.2f}%", "#ff7f0e")
with k6: kpi_card("Sticky Customer Churn", f"{sticky_churn:.2f}%", "#17becf")
with k7: kpi_card("Non-Sticky Customer Churn", f"{non_sticky_churn:.2f}%", "#8c564b")

st.divider()

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Engagement", "📦 Product", "💎 Premium", "🤖 ML Prediction"]
)

# ---------------- TAB 1 ----------------
with tab1:
    churn_active = df.groupby("IsActiveMember")["Exited"].mean().reset_index()
    churn_active["Exited"] *= 100

    fig1 = px.bar(
        churn_active,
        x="IsActiveMember",
        y="Exited",
        text=churn_active["Exited"].round(2),
        color="Exited",
        template="plotly_dark"
    )
    fig1.update_layout(title="Churn by Active Status", transition_duration=500)
    st.plotly_chart(fig1, use_container_width=True)

    relationship_churn = df.groupby("RelationshipScore")["Exited"].mean().reset_index()
    relationship_churn["Exited"] *= 100

    fig2 = px.line(
        relationship_churn,
        x="RelationshipScore",
        y="Exited",
        markers=True,
        template="plotly_dark"
    )
    fig2.update_layout(title="Relationship Strength Trend", transition_duration=500)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- TAB 2 ----------------
with tab2:
    churn_products = df.groupby("NumOfProducts")["Exited"].mean().reset_index()
    churn_products["Exited"] *= 100

    fig3 = px.bar(
        churn_products,
        x="NumOfProducts",
        y="Exited",
        text=churn_products["Exited"].round(2),
        color="Exited",
        template="plotly_dark"
    )
    fig3.update_layout(title="Churn by Number of Products", transition_duration=500)
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- TAB 3 ----------------
with tab3:
    premium_active_split = premium_df.groupby("IsActiveMember")["Exited"].mean().reset_index()
    premium_active_split["Exited"] *= 100

    fig4 = px.bar(
        premium_active_split,
        x="IsActiveMember",
        y="Exited",
        text=premium_active_split["Exited"].round(2),
        color="Exited",
        template="plotly_dark"
    )
    fig4.update_layout(title="Silent Premium Risk Segment", transition_duration=500)
    st.plotly_chart(fig4, use_container_width=True)

# ---------------- TAB 4 (ML SECTION) ----------------
with tab4:
    st.subheader("Churn Prediction Model")

    features = [
        "CreditScore",
        "Age",
        "Balance",
        "NumOfProducts",
        "IsActiveMember",
        "HasCrCard"
    ]

    X = df[features]
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        template="plotly_dark"
    )

    fig_cm.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

# ---------------------------------------------------
# DOWNLOAD DATA
# ---------------------------------------------------
st.divider()
st.subheader("📥 Download Filtered Data")

st.download_button(
    label="Download Current View as CSV",
    data=df.to_csv(index=False),
    file_name="filtered_churn_data.csv",
    mime="text/csv"
)