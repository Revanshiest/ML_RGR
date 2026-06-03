import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Visualizations", page_icon="📈", layout="wide")
st.title("📈 Визуализация данных")

@st.cache_data
def load_data():
    df = pd.read_csv("../data/moldova_train.csv", sep=",", encoding="utf-8")
    return df

try:
    df = load_data()
    
    st.write("В данном разделе представлены графики, демонстрирующие ключевые зависимости в исходном наборе данных.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Визуализация 1: Распределение цены
        st.subheader("1. Распределение цен")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        # Ограничим цену для наглядности (без экстремальных выбросов)
        sns.histplot(df[df['Price(euro)'] < 50000]['Price(euro)'], bins=50, kde=True, ax=ax1, color='royalblue')
        ax1.set_title("Распределение целевой переменной (Price < 50k)")
        ax1.set_xlabel("Цена (Евро)")
        ax1.set_ylabel("Количество автомобилей")
        st.pyplot(fig1)

    with col2:
        # Визуализация 2: Корреляционная матрица
        st.subheader("2. Корреляция признаков")
        cols = ['Price(euro)', 'Year', 'Distance', 'Engine_capacity(cm3)', 'Make_te', 'Model_te']
        corr = df[cols].corr()
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
        ax2.set_title("Корреляционная матрица (выборочно)")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    
    with col3:
        # Визуализация 3: Цена от Года выпуска
        st.subheader("3. Зависимость цены от года")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df[df['Price(euro)'] < 50000], x='Year', y='Price(euro)', alpha=0.3, ax=ax3, color='seagreen')
        ax3.set_title("Цена vs Год выпуска")
        ax3.set_xlabel("Год выпуска")
        ax3.set_ylabel("Цена (Евро)")
        st.pyplot(fig3)

    with col4:
        # Визуализация 4: Цена от Пробега
        st.subheader("4. Зависимость цены от пробега")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        # Фильтруем экстремальные значения пробега
        mask = (df['Distance'] < 500000) & (df['Price(euro)'] < 50000)
        sns.scatterplot(data=df[mask], x='Distance', y='Price(euro)', alpha=0.3, ax=ax4, color='crimson')
        ax4.set_title("Цена vs Пробег (до 500к км)")
        ax4.set_xlabel("Пробег (км)")
        ax4.set_ylabel("Цена (Евро)")
        st.pyplot(fig4)

except Exception as e:
    st.error(f"Не удалось построить графики. Убедитесь, что данные доступны по пути `../data/moldova_train.csv`. Ошибка: {e}")
