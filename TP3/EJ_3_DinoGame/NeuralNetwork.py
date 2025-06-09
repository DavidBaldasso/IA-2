import numpy as np
import tensorflow as tf

class NeuralNetwork:
    """
    Clase que implementa una red neuronal feedforward para controlar al dinosaurio en el juego. 
    """

    def __init__(self):
        self.initialize()

    def initialize(self):
        """
        Configura la arquitectura de la red neuronal y inicializa pesos y sesgos.
        """
        # Definir la forma de cada capa [entrada, oculta1, oculta2, oculta3, oculta4, salida]
        self.shapes = [6, 16, 16, 16, 16, 3]
        
        # Listas para almacenar pesos y sesgos de cada capa
        self.weights = []
        self.biases = []
        
        # Inicializar pesos y sesgos para cada conexión entre capas
        for i in range(len(self.shapes) - 1):
            # Matriz de pesos: [neuronas_capa_siguiente, neuronas_capa_actual]
            weight = tf.Variable(
                tf.random.normal([self.shapes[i + 1], self.shapes[i]]), 
                dtype=tf.float32
            )
            
            # Vector de sesgos: [neuronas_capa_siguiente]
            bias = tf.Variable(
                tf.random.normal([self.shapes[i + 1]]), 
                dtype=tf.float32
            )
            
            self.weights.append(weight)
            self.biases.append(bias)

    def think(self, obstacle, game_speed, dino_x, dino_y):
        """
        Procesa la información del entorno y decide qué acción tomar.
        """
        # === EXTRACCIÓN DE CARACTERÍSTICAS ===
        # Calcular distancia horizontal al obstáculo
        distancia = obstacle.rect.x - dino_x
        
        # Altura del obstáculo
        altura_obstaculo = obstacle.rect.y
        
        # Ancho del obstáculo
        ancho_obstaculo = obstacle.rect.width
        
        # Altura del obstáculo normalizada
        altura_obstaculo_normalizada = obstacle.rect.height
        
        # Tipo de obstáculo
        tipo = 1.0 if "Bird" in str(type(obstacle)) else 0.0
        
        # Altura actual del dinosaurio
        altura_dino = dino_y
        
        # === CREAR VECTOR DE ENTRADA ===
        # Convertir características a tensor de TensorFlow
        X = tf.convert_to_tensor([
            distancia,                     # Distancia al obstáculo
            altura_obstaculo,              # Altura del obstáculo
            tipo,                          # Tipo de obstáculo (pájaro/cactus)
            game_speed,                    # Velocidad del juego
            altura_dino,                   # Altura del dinosaurio
            ancho_obstaculo                # Ancho del obstáculo
        ], dtype=tf.float32)
        
        # === NORMALIZACIÓN ===
        # Normalizar cada característica para mejorar el entrenamiento
        # Los valores grandes pueden dominar el aprendizaje
        X = X / tf.convert_to_tensor([
            1000.0,  # Distancia máxima esperada
            400.0,   # Altura máxima de la pantalla
            1.0,     # Tipo ya está normalizado (0 o 1)
            100.0,   # Velocidad máxima esperada del juego
            400.0,   # Altura máxima del dinosaurio
            100.0    # Ancho máximo esperado del obstáculo
        ], dtype=tf.float32)

        # === PROPAGACIÓN HACIA ADELANTE (FORWARD PASS) ===
        output = X
        
        # Procesar a través de capas ocultas (todas excepto la última)
        for i in range(len(self.weights) - 1):
            # Multiplicación matricial: W * x + b
            output = tf.matmul(self.weights[i], tf.expand_dims(output, axis=-1)) + tf.expand_dims(self.biases[i], axis=-1)
            
            # Función de activación tanh: mantiene salidas entre -1 y 1
            # Mejor que sigmoid para evitar el problema del gradiente desvaneciente
            output = tf.nn.tanh(output)
            
            # Aplanar tensor para la siguiente capa
            output = tf.squeeze(output)

        # === CAPA DE SALIDA ===
        # Aplicar softmax. Cada neurona representa la probabilidad de una acción
        output = tf.matmul(self.weights[-1], tf.expand_dims(output, axis=-1)) + tf.expand_dims(self.biases[-1], axis=-1)
        output = tf.nn.softmax(tf.squeeze(output))

        # Convertir probabilidades a acción concreta
        return self.act(output)

    def act(self, output):
        """
        Convierte la salida de la red neuronal en una acción específica.
        """
        # Encontrar la neurona con mayor activación (mayor probabilidad)
        action = tf.argmax(output).numpy()
        
        # Mapear índice de neurona a acción
        if action == 0:
            return "JUMP"    
        elif action == 1:
            return "DUCK"     
        else:
            return "RIGHT" 
        