import streamlit as st
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import io
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, cohen_kappa_score, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Configuração da página
st.set_page_config(page_title="MNIST Model App", layout="wide")
# Carregar o dataset MNIST
@st.cache_data
def load_data():
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = datasets.mnist.load_data()
    return x_train_raw, y_train_raw, x_test_raw, y_test_raw
x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_data()
# Pré-processamento
num_classes = 10
y_train = keras.utils.to_categorical(y_train_raw, num_classes)
y_test = keras.utils.to_categorical(y_test_raw, num_classes)
x_train = x_train_raw.reshape(60000, 784).astype('float32') / 255
x_test = x_test_raw.reshape(10000, 784).astype('float32') / 255
# Função para treinar e salvar modelos
def train_dnn():
