import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Olympic Medal Forecasting System",
    layout="wide"
)

st.title(" Olympic Medal Forecasting & Analytics Dashboard")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/country_year_aggregated.csv")

df = load_data()

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("models/medal_prediction_model.pkl")

model = load_model()

# Create Tabs
tab1, tab2, tab3 = st.tabs([
    " Overview",
    " Country Deep Dive",
    " Medal Prediction"

])

with tab1:

    st.subheader(" Global Medal Trend Over Time")

    medal_trend = df.groupby("Year")["Total_Medals"].sum().reset_index()

    fig = px.line(
        medal_trend,
        x="Year",
        y="Total_Medals",
        markers=True,
        title="Total Medals by Year"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader(" Top 10 Countries (All Time)")

    top_countries = (
        df.groupby("NOC")["Total_Medals"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )


    top_countries = (
        df.groupby("NOC")["Total_Medals"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_rank = px.bar(
        top_countries,
        x="Total_Medals",
        y="NOC",
        orientation="h",
        title="Top 10 Countries by Total Medals"
    )

    fig_rank.update_layout(
        yaxis=dict(autorange="reversed")  # THIS makes highest appear on top
    )

    st.plotly_chart(fig_rank)

    fig2 = px.bar(
        top_countries,
        x="NOC",
        y="Total_Medals",
        title="Top 10 Medal Winning Countries",
        text="Total_Medals"
    )

    st.plotly_chart(fig2, use_container_width=True)


with tab2:

    st.subheader(" Country Performance Analysis")

    countries = sorted(df["NOC"].unique())
    selected_country = st.selectbox("Select Country", countries)

    country_df = df[df["NOC"] == selected_country]

    # --- Metrics Row ---
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Medals (All Time)", int(country_df["Total_Medals"].sum()))
    col2.metric("Average Medals per Olympics", round(country_df["Total_Medals"].mean(), 2))
    col3.metric("Total Athletes Sent", int(country_df["Total_Athletes"].sum()))

    # --- Medal Trend ---
    st.subheader(" Medal Trend Over Time")

    fig1 = px.line(
        country_df,
        x="Year",
        y="Total_Medals",
        markers=True,
        title=f"{selected_country} - Medal Trend"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # --- Athlete Trend ---
    st.subheader(" Athlete Participation Trend")

    fig2 = px.line(
        country_df,
        x="Year",
        y="Total_Athletes",
        markers=True,
        title=f"{selected_country} - Athlete Count Over Time"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # --- Efficiency Trend ---
    st.subheader(" Medal Efficiency Trend")

    fig3 = px.line(
        country_df,
        x="Year",
        y="Medal_Efficiency",
        markers=True,
        title=f"{selected_country} - Medal Efficiency"
    )

    st.plotly_chart(fig3, use_container_width=True)

with tab3:

    st.subheader(" Predict Future Olympic Medals")

    countries = sorted(df["NOC"].unique())
    selected_country = st.selectbox("Select Country for Prediction", countries)

    # Get last available data for selected country
    country_history = df[df["NOC"] == selected_country].sort_values("Year")
    latest_data = country_history.iloc[-1]

    st.write("Using latest historical performance as base input.")

    # User Inputs
    col1, col2 = st.columns(2)

    with col1:
        total_athletes = st.number_input(
            "Expected Number of Athletes",
            value=int(latest_data["Total_Athletes"])
        )

        total_events = st.number_input(
            "Expected Number of Events",
            value=int(latest_data["Total_Events"])
        )

    with col2:
        host = st.selectbox("Is Host Country?", [0, 1])

        medal_efficiency = st.number_input(
            "Expected Medal Efficiency",
            value=float(latest_data["Medal_Efficiency"])
        )

    if st.button("Predict Medals"):

        input_data = pd.DataFrame([{
            "Previous_medal": latest_data["Total_Medals"],
            "Rolling_3_avg_medal": latest_data["Rolling_3_avg_medal"],
            "Medal_growth_rate": latest_data["Medal_growth_rate"],
            "Total_Athletes": total_athletes,
            "Total_Events": total_events,
            "Medal_Efficiency": medal_efficiency,
            "Host": host
        }])

        prediction = model.predict(input_data)[0]

        st.success(f" Predicted Medal Count: {round(prediction)}")

