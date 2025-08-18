import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.title("Standard Deviation of Mean on Normal Distribution")
st.sidebar.header("1st Parameters")
pop_std_1 = st.sidebar.slider("Population Standard Deviation", value=1.0, min_value=0.1, max_value=10.0,key="pop_std_1")
pop_mean_1 = st.sidebar.slider("Population Mean", value=0.0, min_value=-10.0, max_value=10.0,key="pop_mean_1")
sample_size_1 = st.sidebar.slider("Sample Size", value=30, min_value=1, max_value=100,key="sample_size_1")
st.sidebar.header("2nd Parameters")
pop_std_2 = st.sidebar.slider("Population Standard Deviation", value=1.0, min_value=0.1, max_value=10.0,key="pop_std_2")
pop_mean_2 = st.sidebar.slider("Population Mean", value=0.0, min_value=-10.0, max_value=10.0,key="pop_mean_2")
sample_size_2 = st.sidebar.slider("Sample Size", value=30, min_value=1, max_value=100,key="sample_size_2")

pop_data_1 = np.random.normal(pop_mean_1, pop_std_1, 10000)
pop_data_2 = np.random.normal(pop_mean_2, pop_std_2, 10000)

fig = go.Figure()
fig.add_trace(go.Histogram(x=pop_data_1, name="Population Data", nbinsx=30))
fig.add_trace(go.Histogram(x=pop_data_2, name="Population Data", nbinsx=30))
fig.update_layout(barmode="overlay")
st.plotly_chart(fig)

