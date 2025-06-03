import pickle

import numpy as np
import streamlit as st

input_Gender = 'Male'
input_length = 1.0
input_diameter = 1.0
input_height = 0.5
input_weight = 40.0
input_shucked_weight = 20.0
input_viscera_weight = 10.0
input_shell_weight = 10.0


def main():
    global input_Gender, input_length, input_diameter, input_height, input_weight, input_shucked_weight, input_viscera_weight, input_shell_weight

    st.set_page_config(
        page_title='Crab Age Prediction',
        page_icon='🦀',
        layout='wide'
    )

    # st.title('🦀 Crab Age Prediction')

    with open('README.md','r', encoding='utf-8') as f:
        st.write(f.read())

    st.divider()

    st.header("Model")

    cols = st.columns([0.45, 0.1, 0.45])

    with cols[2]:
        grid = [st.columns(2) for x in range(4)]

        input_Gender = grid[0][0].selectbox("Gender", options=['Male', 'Female', 'Indeterminate'])
        input_length = grid[0][1].number_input("Length (feets)", min_value=0.0, max_value=2.5, value=input_length)
        input_diameter = grid[1][0].number_input("Diameter (feets)", min_value=0.0, max_value=2.0, value=input_diameter)
        input_height = grid[1][1].number_input("Height (feets)", min_value=0.0, max_value=1.0, value=input_height)
        input_weight = grid[2][0].number_input("Weight (ounces)", min_value=0.0, max_value=80.0, step=1.0,
                                               value=input_weight)
        input_shucked_weight = grid[2][1].number_input("Shucked Weight (ounces)", min_value=0.0, max_value=40.0,
                                                       step=1.0, value=input_shucked_weight)
        input_viscera_weight = grid[3][0].number_input("Viscera Weight (ounces)", min_value=0.0, max_value=20.0,
                                                       step=1.0, value=input_viscera_weight)
        input_shell_weight = grid[3][1].number_input("Shell Weight (ounces)", min_value=0.0, max_value=20.0, step=1.0,
                                                     value=input_shell_weight)

        st.container(height=10, border=False)

        if st.button('Predict', type='primary', use_container_width=True, icon='❔'):
            output_crab_age = predictCrabAge()

            with st.container(border=True):
                st.subheader("Crab Age (months):")
                st.subheader(f"{output_crab_age:0.2f}")

    with cols[0]:
        st.image(r'readme_assets/crab_features.jpg')

        st.write("""
***Length*** of the Crab (in Feet; 1 foot = 30.48 cms)

***Diameter*** of the Crab (in Feet; 1 foot = 30.48 cms)

***Height*** of the Crab (in Feet; 1 foot = 30.48 cms)

***Weight*** of the Crab (in ounces; 1 Pound = 16 ounces)

***Shucked Weight***: Weight without the shell (in ounces; 1 Pound = 16 ounces)

***Viscera Weight*** is weight that wraps around your abdominal organs deep inside body (in ounces; 1 Pound = 16 ounces)

***Shell Weight***: Weight of the Shell (in ounces; 1 Pound = 16 ounces)""".strip())


def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_resources():
    MIN_MAX_SCALER_PATH = r'encoders_scalers/min-max-scaler.pickle'
    MODEL_PATH = r'models_cache/linear regression.pickle'

    min_max_scaler = load_obj(MIN_MAX_SCALER_PATH)
    model = load_obj(MODEL_PATH)

    return min_max_scaler, model


def predictCrabAge():
    global input_Gender, input_length, input_diameter, input_height, input_weight, input_shucked_weight, input_viscera_weight, input_shell_weight
    PI = (22.0 / 7.0)

    min_max_scaler, model = load_resources()

    sqrt_weight = input_weight ** 0.5
    sqrt_shucked_weight = input_shucked_weight ** 0.5
    sqrt_viscera_weight = input_viscera_weight ** 0.5
    sqrt_sell_weight = input_shell_weight ** 0.5
    height_power_2 = input_height ** 2
    crab_area = 4 * PI * input_diameter * input_diameter
    crab_clynder_size = crab_area * input_height
    crab_sphere_size = (4.0 / 3.0) * PI * (input_diameter ** 3)

    X = np.array([
        input_length,
        input_diameter,
        input_height,
        input_weight,
        input_shucked_weight,
        input_viscera_weight,
        input_shell_weight,
        sqrt_weight,
        sqrt_shucked_weight,
        sqrt_viscera_weight,
        sqrt_sell_weight,
        height_power_2,
        crab_area,
        crab_clynder_size,
        crab_sphere_size,
        int(input_Gender[0] == 'F'),
        int(input_Gender[0] == 'I'),
        int(input_Gender[0] == 'M'),
    ]).reshape((1, -1))

    X[:, 0:-3] = min_max_scaler.transform(X[:, 0:-3])

    Y = model.predict(X)[0]

    return Y


if __name__ == "__main__":
    main()
