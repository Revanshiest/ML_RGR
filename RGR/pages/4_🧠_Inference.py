import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from catboost import CatBoostRegressor

st.set_page_config(page_title="Inference", page_icon="🧠", layout="wide")
st.title("🧠 Инференс моделей ML")

st.markdown("Выберите модель и введите данные (через загрузку CSV или вручную), чтобы получить предсказание стоимости.")

# Функция загрузки модели
@st.cache_resource
def load_model(model_name):
    path = f"models/{model_name}"
    if not os.path.exists(path):
        return None
        
    if model_name.endswith('.cbm'):
        model = CatBoostRegressor()
        model.load_model(path)
        return model
    else:
        with open(path, "rb") as f:
            return pickle.load(f)

# Выбор модели
models_dict = {
    "ML1 (XGBoost)": "ml1_xgboost.pkl",
    "ML2 (Gradient Boosting)": "ml2_gradient_boosting.pkl",
    "ML3 (CatBoost)": "ml3_catboost.cbm",
    "ML4 (Random Forest)": "ml4_random_forest.pkl",
    "ML5 (Stacking)": "ml5_stacking.pkl",
    "ML6 (Bagging)": "ml6_bagging.pkl"
}

selected_model_name = st.selectbox("Выберите модель для предсказания:", list(models_dict.keys()), index=2)
model_file = models_dict[selected_model_name]
model_obj = load_model(model_file)

if model_obj is None:
    st.error(f"Модель {model_file} не найдена! Сначала запустите скрипт `train_models.py` для обучения и сохранения моделей.")
    st.stop()

# Чтение колонок из тренировочного датасета
@st.cache_data
def get_features_list():
    df = pd.read_csv("../data/moldova_train.csv", sep=",", encoding="utf-8", nrows=1)
    cols_to_drop = ['Price(euro)', 'Transmission_Manual', 'Fuel_type_Diesel', 'Style_Pickup', 'Make', 'Model']
    return [c for c in df.columns if c not in cols_to_drop]

feature_columns = get_features_list()

tab1, tab2 = st.tabs(["Ввод вручную", "Загрузка CSV"])

with tab1:
    st.subheader("Ручной ввод характеристик")
    with st.form("manual_input_form"):
        col1, col2, col3 = st.columns(3)
        
        # Основные признаки
        year = col1.number_input("Year (Год выпуска)", min_value=1950, max_value=2024, value=2015, step=1)
        distance = col2.number_input("Distance (Пробег, км)", min_value=0.0, value=100000.0, step=1000.0)
        engine_cap = col3.number_input("Engine_capacity(cm3) (Объем, см3)", min_value=0.0, value=1500.0, step=100.0)
        
        # Target Encoding признаки
        make_te = col1.number_input("Make_te (Target Encoded Марка)", value=10000.0)
        model_te = col2.number_input("Model_te (Target Encoded Модель)", value=10000.0)
        
        # Бинарные признаки
        transmission_auto = col3.selectbox("Transmission_Automatic (Автомат = 1, Механика = 0)", [1.0, 0.0])
        
        st.markdown("**Тип топлива и Стиль кузова:**")
        col_fuel, col_style = st.columns(2)
        selected_fuel = col_fuel.selectbox("Тип топлива", ["Diesel", "Electric", "Hybrid", "Metan/Propan", "Petrol", "Plug-in Hybrid"])
        selected_style = col_style.selectbox("Стиль кузова", ["Pickup", "Cabriolet", "Combi", "Coupe", "Crossover", "Hatchback", "Microvan", "Minivan", "Roadster", "SUV", "Sedan", "Universal"])

        fuel_electric = 1.0 if selected_fuel == "Electric" else 0.0
        fuel_hybrid = 1.0 if selected_fuel == "Hybrid" else 0.0
        fuel_metan = 1.0 if selected_fuel == "Metan/Propan" else 0.0
        fuel_petrol = 1.0 if selected_fuel == "Petrol" else 0.0
        fuel_plugin = 1.0 if selected_fuel == "Plug-in Hybrid" else 0.0
        
        style_cabriolet = 1.0 if selected_style == "Cabriolet" else 0.0
        style_combi = 1.0 if selected_style == "Combi" else 0.0
        style_coupe = 1.0 if selected_style == "Coupe" else 0.0
        style_crossover = 1.0 if selected_style == "Crossover" else 0.0
        style_hatchback = 1.0 if selected_style == "Hatchback" else 0.0
        style_microvan = 1.0 if selected_style == "Microvan" else 0.0
        style_minivan = 1.0 if selected_style == "Minivan" else 0.0
        style_roadster = 1.0 if selected_style == "Roadster" else 0.0
        style_suv = 1.0 if selected_style == "SUV" else 0.0
        style_sedan = 1.0 if selected_style == "Sedan" else 0.0
        style_universal = 1.0 if selected_style == "Universal" else 0.0

        submit_btn = st.form_submit_button("Получить предсказание 💶")
        
    if submit_btn:
        # Собираем данные в правильном порядке
        input_data = {
            'Year': year,
            'Distance': distance,
            'Engine_capacity(cm3)': engine_cap,
            'Transmission_Automatic': transmission_auto,
            'Fuel_type_Electric': fuel_electric,
            'Fuel_type_Hybrid': fuel_hybrid,
            'Fuel_type_Metan/Propan': fuel_metan,
            'Fuel_type_Petrol': fuel_petrol,
            'Fuel_type_Plug-in Hybrid': fuel_plugin,
            'Style_Cabriolet': style_cabriolet,
            'Style_Combi': style_combi,
            'Style_Coupe': style_coupe,
            'Style_Crossover': style_crossover,
            'Style_Hatchback': style_hatchback,
            'Style_Microvan': style_microvan,
            'Style_Minivan': style_minivan,
            'Style_Roadster': style_roadster,
            'Style_SUV': style_suv,
            'Style_Sedan': style_sedan,
            'Style_Universal': style_universal,
            'Make_te': make_te,
            'Model_te': model_te
        }
        
        # Формируем numpy массив
        row = np.array([input_data[col] for col in feature_columns]).reshape(1, -1)
        
        # Предсказание
        pred = model_obj.predict(row)[0]
            
        st.success(f"### Оценочная стоимость автомобиля: **{pred:,.2f} €**")

with tab2:
    st.subheader("Пакетное предсказание (CSV)")
    uploaded_file = st.file_uploader("Загрузите файл в формате *.csv (с теми же колонками, что и тестовый датасет)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file, sep=",", encoding="utf-8")
            st.write("Превью загруженных данных:")
            st.dataframe(df_upload.head())
            
            if st.button("Предсказать для всех строк 🚀"):
                # Предобработка
                cols_to_drop = ['Price(euro)', 'Transmission_Manual', 'Fuel_type_Diesel', 'Style_Pickup', 'Make', 'Model']
                df_inference = df_upload.drop(columns=[c for c in cols_to_drop if c in df_upload.columns], errors='ignore')
                
                # Проверка наличия нужных колонок
                missing_cols = [c for c in feature_columns if c not in df_inference.columns]
                if missing_cols:
                    st.error(f"В файле отсутствуют необходимые колонки: {missing_cols}")
                else:
                    X_infer = df_inference[feature_columns].values
                    
                    preds = model_obj.predict(X_infer)
                        
                    df_upload['Predicted_Price(euro)'] = preds
                    st.success("Предсказания успешно получены!")
                    
                    # Форматирование цены в €
                    df_upload['Predicted_Price(euro)'] = df_upload['Predicted_Price(euro)'].apply(lambda x: f"€ {x:,.2f}")
                    st.dataframe(df_upload[['Predicted_Price(euro)'] + [c for c in df_upload.columns if c != 'Predicted_Price(euro)']])
                    
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")
