import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

# 1. Função para criar o modelo (igual à do seu app)
def create_model():
    model = tf.keras.Sequential([
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

# 2. Preparar os dados MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pré-processamento e formatação
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# 3. Criar, compilar e treinar o modelo
cnn_model = create_model()
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

print("Iniciando treinamento do modelo...")
cnn_model.fit(x_train, y_train,
              batch_size=128,
              epochs=5, # Geralmente 5-15 épocas são suficientes para MNIST
              validation_data=(x_test, y_test))

# 4. Salvar os pesos do modelo (Este é o arquivo que seu app precisa!)
file_path = 'final_CNN_model.h5'
cnn_model.save_weights(file_path)

print(f"\nTreinamento concluído. Pesos salvos em: {file_path}")
