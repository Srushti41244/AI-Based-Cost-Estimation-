import streamlit as st
import pandas as pd
import os
import numpy as np
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor

# Caching the dataset
@st.cache_resource
def load_and_prepare_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' not found.")
        return None
    return pd.read_csv(file_path)

# Caching the trained model
@st.cache_resource
def train_and_cache_model(df, cost_per_kwh):
    features = []
    targets = []
    season_mapping = {'Summer': 0, 'Winter': 1, 'Rainy': 2}

    for _, row in df.iterrows():
        for season in ['Summer', 'Winter', 'Rainy']:
            active_power = row[f'{season}_kWh']
            standby_power = row['Standby_kWh']
            season_numeric = season_mapping[season]

            for hours in range(1, 25):
                for quantity in range(1, 11):
                    daily_consumption = (active_power * hours * quantity) + (standby_power * (24 - hours) * quantity)
                    monthly_consumption = daily_consumption * 30
                    estimated_cost = monthly_consumption * cost_per_kwh

                    features.append([active_power, standby_power, hours, season_numeric, quantity])
                    targets.append(estimated_cost)

    features = np.array(features)
    targets = np.array(targets)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(features, targets)
    return model

# Predict electricity cost
def predict_cost_xgb(appliance_name, usage_hours_per_day, season, cost_per_kwh, df, appliance_count):
    appliance_data = df[df['Appliance'] == appliance_name].iloc[0]
    active_power = appliance_data[f'{season}_kWh']
    standby_power = appliance_data['Standby_kWh']

    daily_consumption = (active_power * usage_hours_per_day * appliance_count) + \
                        (standby_power * (24 - usage_hours_per_day) * appliance_count)
    monthly_consumption = daily_consumption * 30
    estimated_cost = monthly_consumption * cost_per_kwh
    return estimated_cost

# Optimize usage hours
def optimize_usage_hours(appliance_name, season, cost_per_kwh, df, appliance_count, model):
    appliance_data = df[df['Appliance'] == appliance_name].iloc[0]
    active_power = appliance_data[f'{season}_kWh']
    standby_power = appliance_data['Standby_kWh']
    season_mapping = {'Summer': 0, 'Winter': 1, 'Rainy': 2}
    season_numeric = season_mapping[season]

    optimal_hours = 1
    min_cost = float('inf')

    for hours in range(1, 25):
        estimated_cost = model.predict([[active_power, standby_power, hours, season_numeric, appliance_count]])[0]
        if estimated_cost < min_cost:
            min_cost = estimated_cost
            optimal_hours = hours

    return optimal_hours, min_cost

# Load dataset
file_path = 'Normalized_Energy_Consumption.csv'
df = load_and_prepare_data(file_path)

# Streamlit UI
st.title("AI-Based Energy Estimation and Optimization")

if df is not None:
    if 'Appliance' not in df.columns:
        st.error("The 'Appliance' column is missing from the CSV file.")
    else:
        appliance_list = df['Appliance'].unique().tolist()
        num_appliances = st.number_input("Enter the number of different appliances:", min_value=1, step=1)

        appliance_info = []

        st.write("### Enter details for each appliance")
        for i in range(1, num_appliances + 1):
            st.write(f"#### Appliance {i}")
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                search_term = st.text_input(f"Search appliance {i}:", key=f"search_{i}")
                filtered_appliances = [appliance for appliance in appliance_list if search_term.lower() in appliance.lower()]
                selected_appliance = st.selectbox(f"Select appliance {i}:", options=filtered_appliances, key=f"appliance_{i}")

            with col2:
                appliance_count = st.number_input(f"Quantity:", min_value=1, step=1, key=f"quantity_{i}")

            with col3:
                usage_hours = st.number_input(f"Usage hours:", min_value=1, max_value=24, key=f"hours_{i}")

            if selected_appliance:
                appliance_info.append({'name': selected_appliance, 'count': appliance_count, 'hours': usage_hours})

        season = st.selectbox("Select season:", ['Summer', 'Winter', 'Rainy'])
        cost_per_kwh = st.number_input("Enter cost per kWh in ₹:", min_value=0.0, step=0.01)

        model = train_and_cache_model(df, cost_per_kwh)

        if st.button('Estimate Total Monthly Bill'):
            total_cost = 0
            for appliance in appliance_info:
                cost = predict_cost_xgb(appliance['name'], appliance['hours'], season, cost_per_kwh, df, appliance['count'])
                total_cost += cost
                st.write(f"Estimated Monthly Bill for {appliance['count']} {appliance['name']}(s): ₹{cost:.2f}")
            st.write(f"### Total Estimated Monthly Bill: ₹{total_cost:.2f}")

        if st.button('Optimize Bill'):
            st.write("### Optimizing Usage Hours...")
            total_optimized_cost = 0
            for appliance in appliance_info:
                optimal_hours, optimized_cost = optimize_usage_hours(
                    appliance['name'], season, cost_per_kwh, df, appliance['count'], model
                )
                total_optimized_cost += optimized_cost
                st.write(f"Optimized Usage Hours for {appliance['name']}: {optimal_hours} hours")
                st.write(f"Optimized Estimated Cost: ₹{optimized_cost:.2f}")
            st.write(f"### Total Optimized Monthly Bill: ₹{total_optimized_cost:.2f}")