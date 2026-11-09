
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Instamart AI 360",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"

GREEN = "#00D68F"
BG = "#0B0F14"
PANEL = "#151C25"
BORDER = "#263241"

# =========================================================
# DARK PREMIUM THEME
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0B0F14;
    --panel: #151C25;
    --border: #263241;
    --text: #F1F5F9;
    --muted: #94A3B8;
    --green: #00D68F;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(11,15,20,0.95);
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: #101720;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

section[data-testid="stSidebar"] .stRadio label {
    padding: 11px;
    border-radius: 9px;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: #202C39;
}

h1, h2, h3 {
    color: var(--text) !important;
    font-weight: 700 !important;
}

p, label {
    color: var(--muted);
}

div[data-testid="stMetric"] {
    background: var(--panel);
    padding: 22px;
    border: 1px solid var(--border);
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.15);
}

div[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}

div[data-testid="stMetricValue"] {
    color: var(--green) !important;
    font-weight: 800;
}

div[data-testid="stPlotlyChart"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 12px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

.stTextInput input,
.stTextArea textarea {
    background: #111923 !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
}

div[data-baseweb="select"] > div {
    background: #151C25 !important;
    border-color: var(--border) !important;
}

.stButton button,
.stFormSubmitButton button {
    background: var(--green) !important;
    color: #06130F !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    padding: 10px 18px;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #00B879 !important;
}

hr {
    border-color: var(--border);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_data():

    customers = pd.read_csv(DATA / "customers.csv")
    orders = pd.read_csv(DATA / "orders.csv")
    tickets = pd.read_csv(DATA / "support_tickets.csv")

    orders["order_date"] = pd.to_datetime(
        orders["order_date"],
        errors="coerce"
    )

    return customers, orders, tickets


@st.cache_resource
def load_ticket_model():

    model_path = MODELS / "ticket_classifier.pkl"

    if model_path.exists():
        return joblib.load(model_path)

    return None


try:
    customers, orders, tickets = load_data()
    ticket_model = load_ticket_model()

except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()


# =========================================================
# CHART DESIGN
# =========================================================

def chart_layout(fig, height=350):

    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=45, b=15),
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(
            family="Inter",
            color="#CBD5E1",
            size=12
        ),
        title_font=dict(
            size=15,
            color="#F1F5F9"
        ),
        legend_title_text="",
        colorway=[
            GREEN,
            "#6366F1",
            "#F59E0B",
            "#EC4899",
            "#38BDF8"
        ]
    )

    fig.update_xaxes(
        showgrid=False,
        color="#94A3B8",
        linecolor=BORDER
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=BORDER,
        zeroline=False,
        color="#94A3B8"
    )

    return fig


def show_metrics(items):

    cols = st.columns(len(items))

    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def page_header(title, subtitle):

    st.title(title)
    st.caption(subtitle)
    st.divider()


# =========================================================
# RFM CUSTOMER SEGMENTATION
# =========================================================

def build_segments(customers, orders):

    delivered = orders[
        orders["order_status"] == "Delivered"
    ].copy()

    if delivered.empty:
        rfm = customers[["customer_id"]].copy()
        rfm["last_purchase"] = pd.NaT
        rfm["frequency"] = 0
        rfm["monetary"] = 0.0
        rfm["recency"] = 999
        rfm["R_score"] = 1
        rfm["F_score"] = 1
        rfm["M_score"] = 1
        rfm["customer_segment"] = "No Delivered Purchase"
        return rfm

    analysis_date = (
        orders["order_date"].max().normalize()
        + pd.Timedelta(days=1)
    )

    rfm = delivered.groupby("customer_id").agg(
        last_purchase=("order_date", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_value", "sum")
    ).reset_index()

    rfm["recency"] = (
        analysis_date - rfm["last_purchase"]
    ).dt.days

    rfm = customers[["customer_id"]].merge(
        rfm,
        on="customer_id",
        how="left"
    )

    rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
    rfm["monetary"] = rfm["monetary"].fillna(0)
    rfm["recency"] = rfm["recency"].fillna(999)

    active = rfm["frequency"] > 0

    rfm["R_score"] = 1
    rfm["F_score"] = 1
    rfm["M_score"] = 1

    if active.sum() >= 5:

        rfm.loc[active, "R_score"] = pd.qcut(
            rfm.loc[active, "recency"].rank(method="first"),
            5,
            labels=[5, 4, 3, 2, 1]
        ).astype(int)

        rfm.loc[active, "F_score"] = pd.qcut(
            rfm.loc[active, "frequency"].rank(method="first"),
            5,
            labels=[1, 2, 3, 4, 5]
        ).astype(int)

        rfm.loc[active, "M_score"] = pd.qcut(
            rfm.loc[active, "monetary"].rank(method="first"),
            5,
            labels=[1, 2, 3, 4, 5]
        ).astype(int)

    conditions = [
        rfm["frequency"] == 0,
        (rfm["R_score"] >= 4) & (rfm["F_score"] >= 4) & (rfm["M_score"] >= 4),
        (rfm["F_score"] >= 4) & (rfm["R_score"] >= 3),
        (rfm["R_score"] >= 4) & rfm["F_score"].between(2, 3),
        (rfm["R_score"] >= 4) & (rfm["F_score"] == 1),
        (rfm["R_score"] <= 2) & (rfm["F_score"] >= 3),
        (rfm["R_score"] <= 2) & (rfm["F_score"] <= 2)
    ]

    choices = [
        "No Delivered Purchase",
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "New Customers",
        "At Risk",
        "Hibernating"
    ]

    rfm["customer_segment"] = np.select(
        conditions,
        choices,
        default="Regular Customers"
    )

    return rfm


rfm = build_segments(customers, orders)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🛒 INSTA MART")
    st.markdown("### AI 360")
    st.caption("CUSTOMER INTELLIGENCE PLATFORM")

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Overview",
            "Customer Intelligence",
            "Churn Analytics",
            "Ticket Intelligence"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### System Status")

    st.success("Synthetic data connected")

    if ticket_model is not None:
        st.success("Ticket model loaded")
    else:
        st.warning("Ticket model unavailable")

    st.divider()

    st.caption("Version 1.0 | Educational MVP")


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    page_header(
        "Dashboard Overview",
        "Welcome to your AI-powered customer intelligence platform."
    )

    delivered = orders[
        orders["order_status"] == "Delivered"
    ]

    total_revenue = delivered["order_value"].sum()

    delivered_count = len(delivered)

    average_order = (
        total_revenue / delivered_count
        if delivered_count > 0 else 0
    )

    show_metrics([
        ("Total Customers", f"{len(customers):,}"),
        ("Total Orders", f"{len(orders):,}"),
        ("Delivered Orders", f"{delivered_count:,}"),
        ("Support Tickets", f"{len(tickets):,}")
    ])

    st.markdown("### Revenue Performance")

    c1, c2 = st.columns(2)

    c1.metric(
        "Total Delivered Revenue",
        f"₹{total_revenue:,.0f}"
    )

    c2.metric(
        "Average Order Value",
        f"₹{average_order:,.2f}"
    )

    left, right = st.columns(2)

    with left:

        st.markdown("### Customer Segments")

        segment_counts = (
            rfm["customer_segment"]
            .value_counts()
            .reset_index()
        )

        segment_counts.columns = ["Segment", "Customers"]

        fig = px.pie(
            segment_counts,
            names="Segment",
            values="Customers",
            hole=0.60,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    with right:

        st.markdown("### Ticket Categories")

        category_counts = (
            tickets["category"]
            .value_counts()
            .reset_index()
        )

        category_counts.columns = ["Category", "Tickets"]

        fig = px.bar(
            category_counts,
            x="Category",
            y="Tickets",
            color="Category",
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    st.markdown("### Order Activity")

    trend = (
        orders.dropna(subset=["order_date"])
        .groupby(orders["order_date"].dt.date)
        .size()
        .reset_index(name="Orders")
    )

    trend.columns = ["Date", "Orders"]

    fig = px.line(
        trend,
        x="Date",
        y="Orders",
        markers=True
    )

    fig.update_traces(
        line=dict(color=GREEN, width=3)
    )

    st.plotly_chart(
        chart_layout(fig, 320),
        use_container_width=True
    )


# =========================================================
# CUSTOMER INTELLIGENCE PAGE
# =========================================================

elif page == "Customer Intelligence":

    page_header(
        "Customer Intelligence",
        "Understand customer behaviour through RFM segmentation."
    )

    show_metrics([
        ("Total Customers", f"{len(rfm):,}"),
        (
            "Champions",
            f"{(rfm['customer_segment'] == 'Champions').sum():,}"
        ),
        (
            "At Risk",
            f"{(rfm['customer_segment'] == 'At Risk').sum():,}"
        ),
        (
            "Hibernating",
            f"{(rfm['customer_segment'] == 'Hibernating').sum():,}"
        )
    ])

    left, right = st.columns(2)

    with left:

        st.markdown("### Customer Segment Distribution")

        segment_counts = (
            rfm["customer_segment"]
            .value_counts()
            .reset_index()
        )

        segment_counts.columns = ["Segment", "Customers"]

        fig = px.bar(
            segment_counts,
            x="Segment",
            y="Customers",
            color="Segment",
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_xaxes(tickangle=-25)

        st.plotly_chart(
            chart_layout(fig, 400),
            use_container_width=True
        )

    with right:

        st.markdown("### Customer Monetary Value")

        fig = px.histogram(
            rfm[rfm["monetary"] > 0],
            x="monetary",
            nbins=25,
            color_discrete_sequence=[GREEN]
        )

        fig.update_layout(
            xaxis_title="Customer Value (₹)",
            yaxis_title="Number of Customers"
        )

        st.plotly_chart(
            chart_layout(fig, 400),
            use_container_width=True
        )

    st.markdown("### Customer Explorer")

    selected_segment = st.selectbox(
        "Select Customer Segment",
        ["All Segments"] + sorted(
            rfm["customer_segment"].unique()
        )
    )

    customer_view = rfm.copy()

    if selected_segment != "All Segments":

        customer_view = customer_view[
            customer_view["customer_segment"] == selected_segment
        ]

    search = st.text_input(
        "Search Customer ID",
        placeholder="Search customer..."
    )

    if search:

        customer_view = customer_view[
            customer_view["customer_id"]
            .astype(str)
            .str.contains(search, case=False, na=False)
        ]

    st.caption(f"{len(customer_view)} customer records found")

    st.dataframe(
        customer_view[
            [
                "customer_id",
                "recency",
                "frequency",
                "monetary",
                "R_score",
                "F_score",
                "M_score",
                "customer_segment"
            ]
        ].sort_values(
            "monetary",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# CHURN ANALYTICS PAGE
# =========================================================

elif page == "Churn Analytics":

    page_header(
        "Churn Analytics",
        "Monitor customer churn probability and risk levels."
    )

    report_path = (
        PROCESSED / "customer_churn_risk_report.csv"
    )

    if report_path.exists():

        churn_report = pd.read_csv(report_path)

        if "risk_level" in churn_report.columns:

            risk_counts = churn_report["risk_level"].value_counts()

            high = int(risk_counts.get("High", 0))
            medium = int(risk_counts.get("Medium", 0))
            low = int(risk_counts.get("Low", 0))

            show_metrics([
                ("Customers Analysed", f"{len(churn_report):,}"),
                ("High Risk", f"{high:,}"),
                ("Medium Risk", f"{medium:,}"),
                ("Low Risk", f"{low:,}")
            ])

            left, right = st.columns(2)

            with left:

                st.markdown("### Risk Distribution")

                risk_df = pd.DataFrame({
                    "Risk": ["High", "Medium", "Low"],
                    "Customers": [high, medium, low]
                })

                fig = px.pie(
                    risk_df,
                    names="Risk",
                    values="Customers",
                    hole=0.60,
                    color="Risk",
                    color_discrete_map={
                        "High": "#EF4444",
                        "Medium": "#F59E0B",
                        "Low": GREEN
                    }
                )

                st.plotly_chart(
                    chart_layout(fig),
                    use_container_width=True
                )

            with right:

                st.markdown("### Churn Probability")

                if "churn_probability" in churn_report.columns:

                    fig = px.histogram(
                        churn_report,
                        x="churn_probability",
                        nbins=20,
                        color_discrete_sequence=["#6366F1"]
                    )

                    st.plotly_chart(
                        chart_layout(fig),
                        use_container_width=True
                    )

            st.markdown("### Customer Risk Explorer")

            risk_filter = st.selectbox(
                "Filter by Risk",
                ["All", "High", "Medium", "Low"]
            )

            filtered = churn_report.copy()

            if risk_filter != "All":

                filtered = filtered[
                    filtered["risk_level"] == risk_filter
                ]

            if "churn_probability" in filtered.columns:

                filtered = filtered.sort_values(
                    "churn_probability",
                    ascending=False
                )

            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Risk Report",
                data=churn_report.to_csv(index=False).encode("utf-8"),
                file_name="customer_churn_risk_report.csv",
                mime="text/csv"
            )

        else:

            st.warning("Risk level column is missing from the report.")

    else:

        st.warning(
            "Churn report not found. "
            "Export the report from your churn notebook first."
        )


# =========================================================
# TICKET INTELLIGENCE PAGE
# =========================================================

elif page == "Ticket Intelligence":

    page_header(
        "Ticket Intelligence",
        "AI-powered customer support analytics and classification."
    )

    high_priority = int(
        (tickets["priority"] == "High").sum()
    )

    negative = int(
        (tickets["sentiment"] == "Negative").sum()
    )

    open_tickets = int(
        (tickets["ticket_status"] == "Open").sum()
    )

    show_metrics([
        ("Total Tickets", f"{len(tickets):,}"),
        ("High Priority", f"{high_priority:,}"),
        ("Negative Sentiment", f"{negative:,}"),
        ("Open Tickets", f"{open_tickets:,}")
    ])

    left, right = st.columns(2)

    with left:

        st.markdown("### Ticket Categories")

        fig = px.pie(
            tickets,
            names="category",
            hole=0.60,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    with right:

        st.markdown("### Ticket Priority")

        priority_counts = (
            tickets["priority"]
            .value_counts()
            .reset_index()
        )

        priority_counts.columns = ["Priority", "Tickets"]

        fig = px.bar(
            priority_counts,
            x="Priority",
            y="Tickets",
            color="Priority",
            color_discrete_map={
                "High": "#EF4444",
                "Medium": "#F59E0B",
                "Low": GREEN
            }
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    st.markdown("### Customer Sentiment")

    sentiment_counts = (
        tickets["sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = ["Sentiment", "Tickets"]

    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Tickets",
        color="Sentiment",
        color_discrete_map={
            "Negative": "#EF4444",
            "Neutral": "#94A3B8",
            "Positive": GREEN
        }
    )

    st.plotly_chart(
        chart_layout(fig, 300),
        use_container_width=True
    )

    st.divider()

    st.markdown("### ✨ AI Ticket Classification")

    st.caption(
        "Enter a customer complaint to predict its ticket category."
    )

    with st.form("ticket_prediction_form"):

        ticket_text = st.text_area(
            "Customer Complaint",
            placeholder=(
                "Example: My order was delayed "
                "and I have not received my refund..."
            ),
            height=120
        )

        submitted = st.form_submit_button(
            "Predict Ticket Category"
        )

    if submitted:

        if not ticket_text.strip():

            st.warning("Please enter a ticket message.")

        elif ticket_model is None:

            st.error("Ticket classification model was not found.")

        else:

            prediction = ticket_model.predict(
                [ticket_text]
            )[0]

            st.success(
                f"Predicted Ticket Category: {prediction}"
            )

    st.markdown("### Recent Support Tickets")

    st.dataframe(
        tickets.head(20),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "INSTAMART AI 360 | DARK PREMIUM EDITION | "
    "Educational prototype using synthetic data"
)
