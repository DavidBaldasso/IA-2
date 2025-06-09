import numpy as np
import tensorflow as tf

class NeuralNetwork:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.shapes = [6, 16, 16, 16, 16, 3]    # 6 entradas
        self.weights = []
        self.biases = []
        
        for i in range(len(self.shapes) - 1):
            weight = tf.Variable(tf.random.normal([self.shapes[i + 1], self.shapes[i]]), dtype=tf.float32)
            bias = tf.Variable(tf.random.normal([self.shapes[i + 1]]), dtype=tf.float32)
            self.weights.append(weight)
            self.biases.append(bias)

    def think(self, obstacle, game_speed, dino_x, dino_y):
        # Calcular características más informativas
        distancia = obstacle.rect.x - dino_x
        altura_obstaculo = obstacle.rect.y
        ancho_obstaculo = obstacle.rect.width
        altura_obstaculo_normalizada = obstacle.rect.height
        tipo = 1.0 if "Bird" in str(type(obstacle)) else 0.0
        altura_dino = dino_y
        
        # Vector de entrada con más información
        X = tf.convert_to_tensor([
            distancia, 
            altura_obstaculo, 
            tipo, 
            game_speed, 
            altura_dino,
            ancho_obstaculo
        ], dtype=tf.float32)
        
        # Normalización mejorada
        X = X / tf.convert_to_tensor([
            1000.0,  # distancia máxima
            400.0,   # altura máxima
            1.0,     # tipo (ya normalizado)
            100.0,    # velocidad máxima esperada
            400.0,   # altura máxima del dino
            100.0    # ancho máximo del obstáculo
        ], dtype=tf.float32)

        # Forward pass por la red
        output = X
        for i in range(len(self.weights) - 1):
            output = tf.matmul(self.weights[i], tf.expand_dims(output, axis=-1)) + tf.expand_dims(self.biases[i], axis=-1)
            output = tf.nn.tanh(output)  # Función de activación tanh
            output = tf.squeeze(output)

        # Capa final con softmax
        output = tf.matmul(self.weights[-1], tf.expand_dims(output, axis=-1)) + tf.expand_dims(self.biases[-1], axis=-1)
        output = tf.nn.softmax(tf.squeeze(output))

        return self.act(output)

    def act(self, output):
        action = tf.argmax(output).numpy()
        if action == 0:
            return "JUMP"
        elif action == 1:
            return "DUCK"
        else:
            return "RIGHT"