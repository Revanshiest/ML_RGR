import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dataset Info", page_icon="📊")

st.title("📊 Информация о наборе данных")

st.header("Предметная область")
st.write("""
Данный датасет содержит информацию об автомобилях, выставленных на продажу в Молдове.
Целью работы является решение задачи регрессии: предсказание цены автомобиля (Price in euro) на основе его технических и эксплуатационных характеристик.
""")

st.header("Описание признаков")
st.markdown("""
* **Year**: Год выпуска автомобиля.
* **Distance**: Пробег автомобиля (в километрах).
* **Engine_capacity(cm3)**: Объем двигателя.
* **Price(euro)**: Цена автомобиля в евро (Целевая переменная).
* **Трансмиссия**: (Transmission_Automatic, Transmission_Manual) - тип коробки передач.
* **Тип топлива**: Категориальные признаки (Electric, Hybrid, Metan/Propan, Petrol, Plug-in Hybrid, Diesel).
* **Стиль кузова**: Cabriolet, Combi, Coupe, Crossover, Hatchback, Microvan, Minivan, Pickup, Roadster, SUV, Sedan, Universal.
* **Марка и Модель**: Представлены в виде Target Encoding (`Make_te`, `Model_te`).
""")

st.header("Особенности предобработки данных (EDA)")
st.write("""
Перед обучением моделей были выполнены следующие шаги:
1. Исходные признаки `Make` и `Model` были закодированы средним значением целевой переменной (Target Encoding).
2. Для устранения мультиколлинеарности были удалены некоторые dummy-переменные:
   `Transmission_Manual`, `Fuel_type_Diesel`, `Style_Pickup`.
3. Также из финального набора для обучения были исключены сырые текстовые признаки `Make` и `Model`.
""")

st.header("Пример данных (Train)")
try:
    df = pd.read_csv("../data/moldova_train.csv", sep=",", encoding="utf-8")
    st.dataframe(df.head(10))
except Exception as e:
    st.warning(f"Не удалось загрузить данные для предпросмотра. Проверьте путь ../data/moldova_train.csv. Ошибка: {e}")
