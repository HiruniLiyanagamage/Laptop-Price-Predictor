import streamlit as st
import pickle
import numpy as np

# Load the model and data
pipe = pickle.load(open('rd_model.pkl', 'rb'))
df = pickle.load(open('lapdf.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Mapping for categorical features
company_map = {
    0: 'Acer ', 1: 'Apple ', 2: 'Asus ', 3: 'Chuwi ', 4: 'Dell ',
    5: 'Fujitsu ', 6: 'Google ', 7: 'HP ', 8: 'Huawei ', 9: 'LG ',
    10: 'Lenovo ', 11: 'MSI ', 12: 'Mediacom ', 13: 'Microsoft ',
    14: 'Razer ', 15: 'Samsung ', 16: 'Toshiba ', 17: 'Vero ', 18: 'Xiaomi '
}


typename_map = {
    0: '2 in 1 Convertible ', 1: 'Gaming ', 2: 'Netbook ',
    3: 'Notebook ', 4: 'Ultrabook ', 5: 'Workstation '
}

opsys_map = {
    0: 'Android ', 1: 'Chrome OS ', 2: 'Linux ',
    3: 'Mac OS X ', 4: 'No OS ', 5: 'Windows 10 ',
    6: 'Windows 10 S ', 7: 'Windows 7 ', 8: 'macOS '
}

gpu_brand_map = {
    0: 'AMD', 1: 'ARM', 2: 'Intel',
    3: 'Nvidia'
}

cpu_brand_map = {
    0: 'AMD ',
    1: 'Intel ',
    2: 'Samsung '
}

# Collect user inputs

col1, col2, col3 = st.columns(3)

with col1:
    # Brand
    company = st.selectbox('Company', list(company_map.keys()), format_func=lambda x: company_map[x])

    # Type of laptop
    typename = st.selectbox('Type Name', list(typename_map.keys()), format_func=lambda x: typename_map[x])

    # Inches
    inches = st.number_input('Inches', min_value=10.0, max_value=25.0, value=15.0, step=0.1)

    # RAM
    ram = st.number_input('RAM (in GB)', min_value=2, max_value=64, value=8, step=1)

    # OS
    opsys = st.selectbox('Operating System', list(opsys_map.keys()), format_func=lambda x: opsys_map[x])

with col2:
    # Weight
    weight = st.number_input('Weight (in kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1)

    # GPU
    gpu = st.selectbox('GPU', list(gpu_brand_map.keys()), format_func=lambda x: gpu_brand_map[x])

    # Screen size
    screen_width = st.number_input('Screen Width (in pixels)', min_value=800, max_value=4000, value=1920, step=1)
    screen_height = st.number_input('Screen Height (in pixels)', min_value=600, max_value=3000, value=1080, step=1)
    
    # CPU
    cpu_brand = st.selectbox('CPU Brand', list(cpu_brand_map.keys()), format_func=lambda x: cpu_brand_map[x])

with col3:
    cpu_freq = st.number_input('CPU Frequency (in GHz)', min_value=1.0, max_value=5.0, value=2.5, step=0.1)
    # Storage
    ssd = st.number_input('SSD (in GB)', min_value=0, max_value=2048, value=256, step=128)
    flash = st.number_input('Flash (in GB)', min_value=0, max_value=512, value=0, step=128)
    hdd = st.number_input('HDD (in GB)', min_value=0, max_value=2048, value=0, step=128)
    hybrid = st.number_input('Hybrid (in GB)', min_value=0, max_value=2048, value=0, step=128)


# Predict button
if st.button('Predict Price'):
    # Prepare query
    query = np.array([company, typename, inches, ram, opsys, weight, gpu, screen_width, screen_height, cpu_brand, cpu_freq, ssd, flash, hdd, hybrid])
    query = query.reshape(1, -1)

    # Predict
    predicted_price = pipe.predict(query)[0]

    st.title(f"The predicted price is ${predicted_price:.2f}")
