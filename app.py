import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

# --- Configuração da Página ---
st.set_page_config(page_title="MNIST Classifier", layout="wide")
st.title('Classificador de Números (MNIST)')
st.markdown('Envie uma imagem em preto e branco de um número desenhado à mão (0-9).')
st.caption(f"Usando TensorFlow versão: {tf.__version__} | Python versão: {sys.version.split(' ')[0]}")

# --- Definição da Arquitetura do Modelo ---
# A arquitetura deve ser EXATAMENTE a mesma usada para treinar o arquivo .h5.
def create_model():
    model = tf.keras.Sequential([
        # Estrutura CNN idêntica ao treinamento
        tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])
    return model

# --- Carregamento e Cache do Modelo ---
@st.cache_resource
def load_model_weights():
    # 1. Cria a arquitetura
    model = create_model()
    model_path = 'final_CNN_model.h5'
    
    # 2. Verifica se o arquivo existe
    if not os.path.exists(model_path):
        st.error(f"Arquivo de pesos não encontrado: {model_path}")
        st.warning("Verifique se o arquivo está no mesmo diretório do 'app.py'.")
        # Levanta um erro específico que pode ser capturado
        raise FileNotFoundError(f"Arquivo de pesos {model_path} ausente. Verifique o caminho.")
        
    # 3. Carrega os pesos
    try:
        # Nota: load_weights é usado quando se tem apenas o .h5 dos pesos, e a arquitetura é recriada.
        model.load_weights(model_path)
    except Exception as e:
        st.error("Erro ao carregar os pesos do arquivo .h5.")
        st.warning("Pode ser incompatibilidade de versão Keras/TensorFlow. Tente recriar o .h5 na sua versão atual do TF.")
        # Retorna o erro detalhado
        raise e
        
    return model

# Tenta carregar o modelo
model = None
try:
    with st.spinner("Carregando modelo e pesos..."):
        model = load_model_weights()
    st.success("✅ Modelo CNN carregado com sucesso!")
except Exception as e:
    st.error("🚨 O aplicativo não pôde inicializar devido a um erro no carregamento do modelo.")
    st.exception(e) # Mostra o erro detalhado para debug

# --- Interface de Upload e Classificação ---
if model is not None:
    file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

    if file is not None:
        try:
            # 1. Abre a imagem e converte para escala de cinza (L)
            image = Image.open(file).convert('L') 
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption='Imagem enviada', width=150)

            # 2. Pré-processamento: Redimensiona e Normaliza
            img_resized = image.resize((28, 28)) 
            img_array = np.array(img_resized)
            img_array = img_array.astype('float32') / 255.0

            # 3. INVERSÃO DE COR (Crucial para MNIST: Fundo preto, número branco)
            if np.mean(img_array) > 0.5: 
                img_array = 1.0 - img_array
                st.caption("Cores da imagem invertidas (fundo preto/número branco).")

            # 4. Ajusta as dimensões para o formato do modelo (1, 28, 28, 1)
            img_array = img_array.reshape(1, 28, 28, 1) 
            
            with col2:
                if st.button('Classificar Imagem'):
                    with st.spinner('Realizando a predição...'):
                        prediction = model.predict(img_array)
                        label = np.argmax(prediction)
                        confidence = np.max(prediction) * 100
                        
                        st.markdown(f"### 🤖 Resultado: **{label}**")
                        st.info(f"Certeza da IA: **{confidence:.2f}%**")
                        
                        st.subheader("Probabilidades")
                        # Exibe o gráfico das probabilidades para visualização
                        st.bar_chart(prediction.flatten())
                        
        except Exception as e:
            st.error(f"Erro no processamento da imagem ou predição: {e}")
