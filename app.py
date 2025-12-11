import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title('Classificador de Números (MNIST)')
st.write('Envie uma imagem de um número desenhado à mão (0-9).')

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])
    return model

@st.cache_resource
def load_model_weights():
    model = create_model()
    model.load_weights('final_CNN_model.h5')
    return model

try:
    model = load_model_weights()
    st.success("Modelo carregado com sucesso!")
except Exception as e:
    st.error(f"Erro ao carregar modelo: {e}")

file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file).convert('L')
    st.image(image, caption='Imagem enviada', width=150)

    img_array = np.array(image.resize((28, 28)))
    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    if st.button('Classificar'):
        prediction = model.predict(img_array)
        label = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        st.markdown(f"### Resultado: **{label}**")
        st.info(f"Certeza da IA: {confidence:.2f}%")
