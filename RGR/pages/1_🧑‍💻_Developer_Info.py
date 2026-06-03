import streamlit as st
import os

st.set_page_config(page_title="Developer Info", page_icon="🧑‍💻")

st.title("🧑‍💻 Информация о разработчике")

col1, col2 = st.columns([1, 2])

with col1:
    # Заглушка для фото. Пользователь может заменить "photo.jpg" на свое фото.
    photo_path = "photo.jpg"
    if os.path.exists(photo_path):
        st.image(photo_path, width=250, caption="Разработчик")
    else:
        st.info("📷 Место для вашей цветной фотографии. Сохраните фото под именем 'photo.jpg' в этой же папке.")

with col2:
    st.subheader("Личные данные")
    st.markdown("""
    * **ФИО:** Меженов Егор Ильич
    * **Учебная группа:** ФИТ-242
    * **Тема РГР:** Разработка Web-приложения (дашборда)
для инференса (вывода) моделей ML и анализа данных
    """)

st.markdown("---")
st.markdown("Разработано в рамках лабораторных работ по машинному обучению.")
