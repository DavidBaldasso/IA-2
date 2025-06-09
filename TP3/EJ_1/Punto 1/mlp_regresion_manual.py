import os
import random
import math
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
#                                  CONFIGURACIÓN
#
# 1) Nombre de tu archivo de datos (un punto por línea, formato "(x, y)"):
ARCHIVO_DATOS = "datos_ejercicio1.txt"

# 2) Porcentaje de datos que usarás para entrenar (por ejemplo, 0.10 → 10%,
#    0.15 → 15%, 0.05 → 5%, etc.):
PORCENTAJE_TRAIN = 0.80

# 3) Arquitectura de la red: lista con la cantidad de neuronas en cada capa oculta.
#    Ejemplo:
#       []        → sin capas ocultas (equivale a regresión lineal W*x + b)
#       [5]       → una capa oculta con 5 neuronas
#       [5, 5]    → dos capas ocultas de 5 neuronas cada una (como antes)
#       [10, 8, 4]→ tres capas ocultas: 10 neuronas, luego 8, luego 4.
#
#    La capa de entrada siempre es de dimensión 1 (x escalar), y la capa de salida
#    también será de dimensión 1 (ŷ escalar), ya que es regresión.
#
CAPAS_OCULTAS = [4]

# 4) Hiperparámetros de entrenamiento:
EPOCHS = 1000         # Número de épocas totales
LEARNING_RATE = 0.001  # Tasa de aprendizaje (η)
SEED = 42             # Semilla para reproducibilidad (opcional). Usa None para aleatorio.

# ─────────────────────────────────────────────────────────────────────────────


def leer_puntos_desde_archivo(nombre_archivo):
    """
    Lee un archivo de texto con líneas "(x, y)" y devuelve lista de tuplas [(x1, y1), ...].
    """
    puntos = []
    if not os.path.isfile(nombre_archivo):
        raise FileNotFoundError(f"No se encontró el archivo: {nombre_archivo}")

    with open(nombre_archivo, 'r', encoding='utf-8') as f:
        for linea in f:
            linea = linea.strip()
            if not (linea.startswith("(") and linea.endswith(")")):
                continue
            try:
                cuerpo = linea[1:-1]
                x_str, y_str = cuerpo.split(",")
                x = float(x_str.strip())
                y = float(y_str.strip())
                puntos.append((x, y))
            except Exception as e:
                print(f"Atención: línea con formato incorrecto: '{linea}' → {e}")
    return puntos


def dividir_dataset(puntos, porcentaje_entrenamiento=0.10, semilla=None):
    """
    Divide aleatoriamente la lista 'puntos' en (train_set, test_set),
    donde train_set tiene (porcentaje_entrenamiento * len(puntos)) elementos.
    Si semilla no es None, se fija random.seed(semilla) para reproducibilidad.
    """
    if semilla is not None:
        random.seed(semilla)

    total = len(puntos)
    num_entrenamiento = max(1, int(total * porcentaje_entrenamiento))
    copia = puntos[:]
    random.shuffle(copia)

    train_set = copia[:num_entrenamiento]
    test_set = copia[num_entrenamiento:]
    return train_set, test_set


def tanh(x):
    return math.tanh(x)


def derivada_tanh(x):
    t = math.tanh(x)
    return 1.0 - t * t


def identidad(x):
    return x


def derivada_identidad(x):
    return 1.0


def inicializar_red(layer_sizes):
    """
    Dado un vector layer_sizes = [n_entrada, n_oculta1, n_oculta2, ..., n_salida],
    inicializa pesos y bias aleatorios en [-0.5, 0.5].
    Devuelve:
        W: lista de matrices de pesos, donde W[l][i][j] es el peso que une
           la neurona i de la capa l con la neurona j de la capa (l+1).
        b: lista de vectores de bias, donde b[l][j] es el bias de la neurona j
           en la capa (l+1). (No hay biases para la capa de entrada).
    Ejemplo:
        layer_sizes = [1, 5, 5, 1]
          → W[0] es  (1×5), W[1] es (5×5), W[2] es (5×1).
          → b[0] es (5,),    b[1] es (5,),    b[2] es (1,).
    """
    W = []
    b = []
    for l in range(len(layer_sizes) - 1):
        n_in = layer_sizes[l]
        n_out = layer_sizes[l + 1]
        # Matriz de pesos n_in × n_out
        W.append([[random.uniform(-0.5, 0.5) for _ in range(n_out)] for _ in range(n_in)])
        # Bias para cada neurona de la capa de salida de este bloque
        b.append([random.uniform(-0.5, 0.5) for _ in range(n_out)])
    return W, b


def forward(red, x):
    """
    Dada la red (W, b) y la entrada escalar x, calcula:
      - z[l]: vector de valores z antes de la activación en capa l (l=1..L)
      - a[l]: vector de activaciones en capa l (l=0..L), donde a[0] = [x]
      - y_pred = a[L] (en nuestro caso L = número de capas - 1)
    Devuelve un diccionario con listas:
      {
        "zs": [z1, z2, ..., zL],
        "activs": [a0, a1, a2, ..., aL],
        "y_pred": valor escalar de la salida
      }
    """
    W, b = red["W"], red["b"]
    layer_sizes = red["layer_sizes"]

    # a[0] es la capa de entrada, solo contiene [x].
    activs = [[x]]
    zs = []  # lista para guardar z en cada capa oculta/salida

    # Recorrer capas 1..L (ocultas y salida) 
    for l in range(len(layer_sizes) - 1):
        entrada = activs[-1]             # a[l], lista de tamaño n_in
        n_in = layer_sizes[l]
        n_out = layer_sizes[l + 1]

        z_l = [0.0] * n_out
        a_l = [0.0] * n_out

        for j in range(n_out):
            suma = 0.0
            for i in range(n_in):
                suma += W[l][i][j] * entrada[i]
            suma += b[l][j]
            z_l[j] = suma

            # Activación:
            #   - si no es la última capa (l < L-1) → tanh
            #   - si es la capa final (l == L-1) → identidad
            if l < len(layer_sizes) - 2:
                a_l[j] = tanh(suma)
            else:
                a_l[j] = identidad(suma)

        zs.append(z_l)
        activs.append(a_l)

    y_pred = activs[-1][0]  # la salida es un escalar (única neurona en capa final)
    return {"zs": zs, "activs": activs, "y_pred": y_pred}


def calcular_error(y_pred, y_true):
    """
    Para un solo par (y_pred, y_true), devuelve (loss, dL_dy):
      loss = 1/2 * (y_pred - y_true)^2
      dL_dy = (y_pred - y_true)
    """
    diff = y_pred - y_true
    loss = 0.5 * diff * diff
    dL_dy = diff
    return loss, dL_dy


def backpropagar(red, valores_forward, y_true, lr):
    """
    Dada la red, los valores calculados en forward, el y_true y la tasa lr:
    1) Calcula todos los gradientes parciales de la pérdida respecto a W y b.
    2) Actualiza W y b en su dirección opuesta al gradiente.

    red: diccionario con keys:
         - "W"           : lista de matrices de pesos
         - "b"           : lista de vectores de bias
         - "layer_sizes" : lista [n_entrada, n_oculta1, ..., n_salida]
    valores_forward: resultado de forward(red, x), con:
         - "zs": [z1, z2, ..., zL]
         - "activs": [a0, a1, ..., aL]
         - "y_pred": escal ar
    y_true: valor real para este ejemplo
    lr: learning rate
    """
    W, b = red["W"], red["b"]
    layer_sizes = red["layer_sizes"]
    zs = valores_forward["zs"]         # lista de vectores z por capa
    activs = valores_forward["activs"] # lista de vectores a por capa
    y_pred = valores_forward["y_pred"]

    L = len(layer_sizes) - 1  # número de “bloques” (ocultas + salida)
    # Cada l abre un bloque desde capa l → capa l+1

    # 1) Calculamos la derivada en la capa de salida (l = L-1)
    loss, dL_dy = calcular_error(y_pred, y_true)


    # Derivada de identidad en capa final: 1
    # Entonces dL/dz^{(L)} = dL/dy_pred * d y_pred/d z^{(L)} = dL_dy * 1
    delta_prev = [dL_dy]  # vector de tamaño 1, delta en capa final

    # Gradientes acumulados para cada capa:
    #   dW[l][i][j] corresponde a ∂L/∂W[l][i][j]
    #   db[l][j]         corresponde a ∂L/∂b[l][j]
    grad_W = [ [ [0.0]*layer_sizes[l+1] for _ in range(layer_sizes[l]) ]
               for l in range(L) ]
    grad_b = [ [0.0]*layer_sizes[l+1] for l in range(L) ]

    # 2) Retropropagar a través de cada bloque l = L-1, L-2, ..., 0
    for l in reversed(range(L)):
        # En la iteración l:
        #   - la capa “de salida” de este bloque es la capa index l+1 (dim = layer_sizes[l+1]),
        #   - la capa “de entrada” al bloque es layer_sizes[l].
        n_in = layer_sizes[l]
        n_out = layer_sizes[l + 1]

        # Vector de activaciones en la capa de entrada: a^{(l)}  (o activs[l])
        a_l = activs[l]       # tamaño n_in
        # Vector de valores z en la capa de salida: z^{(l+1)}  (o zs[l])
        z_next = zs[l]        # tamaño n_out

        # 2.1) Calcular gradientes ∂L/∂W[l][i][j]  y  ∂L/∂b[l][j]
        #     Recordar: para la capa final usamos derivada identidad, ya aplicada.
        #     Para capas ocultas (l < L-1) multiplicamos por derivada de tanh.
        delta_current = [0.0]*n_out  # será delta en esta capa de salida

        for j in range(n_out):
            if l == L - 1:
                # Capa de salida: delta_prev[j] = dL/dz^{(L)} (tamaño 1)
                delta_current[j] = delta_prev[j]
            else:
                # Capa oculta: delta_prev[j] = (sum_k W[l+1][j][k] * delta_prev_next[k]) * tanh'(z_next[j])
                delta_current[j] = delta_prev[j] * derivada_tanh(z_next[j])

            # gradiente respecto a b[l][j]:
            grad_b[l][j] = delta_current[j]

            # gradiente respecto a W[l][i][j] para cada i ∈ [0..n_in-1]:
            for i in range(n_in):
                grad_W[l][i][j] = delta_current[j] * a_l[i]

        # 2.2) Calcular delta_prev para la siguiente iteración (la capa anterior al bloque)
        if l > 0:
            # Queremos ∂L/∂a^{(l)} = sum_j (W[l][i][j] * delta_current[j]),
            # y luego ∂L/∂z^{(l)} = (∂L/∂a^{(l)}) * tanh'(z^{(l)}) para cada neurona i.
            z_prev = zs[l - 1]       # z en capa l (tamaño layer_sizes[l])
            delta_prev = [0.0] * layer_sizes[l]
            for i in range(layer_sizes[l]):
                suma = 0.0
                for j in range(layer_sizes[l + 1]):
                    suma += W[l][i][j] * delta_current[j]
                # Multiplicar por derivada de tanh(z_prev[i])
                delta_prev[i] = suma * derivada_tanh(z_prev[i])

        # 2.3) Actualizar pesos y bias del bloque l:
        for i in range(n_in):
            for j in range(n_out):
                W[l][i][j] -= LEARNING_RATE * grad_W[l][i][j]
        for j in range(n_out):
            b[l][j] -= LEARNING_RATE * grad_b[l][j]

        # Preparar para la siguiente iteración (subir un bloque más)
        delta_prev = delta_prev

    return loss  # para sumar al loss total de la época


def entrenar_red(red, train_set, epochs=1000, lr=0.01, lambda_reg=0.01):
    """
    Entrena la red 'red' usando train_set [(x, y), ...] durante 'epochs' épocas.
    Retorna la lista historial_perdidas donde cada elemento es el loss promedio
    de esa época.
    """
    historial_perdidas = []

    for epoca in range(epochs):
        random.shuffle(train_set)
        suma_loss = 0.0

        for x, y_true in train_set:
            vals = forward(red, x)
            loss = backpropagar(red, vals, y_true, lr)
            suma_loss += loss

        loss_promedio = suma_loss / len(train_set)
        historial_perdidas.append(loss_promedio)

        if (epoca + 1) % max(1, (epochs // 10)) == 0 or epoca == 0:
            print(f"Época {epoca+1}/{epochs} — Loss promedio: {loss_promedio:.6f}")

    return historial_perdidas


def evaluar_red(red, test_set):
    """
    Calcula el MSE promedio sobre test_set y devuelve (mse_promedio, lista_predicciones).
    """
    suma_error2 = 0.0
    lista_predicciones = []

    for x, y_true in test_set:
        y_pred = forward(red, x)["y_pred"]
        diff = y_pred - y_true
        suma_error2 += diff * diff
        lista_predicciones.append((x, y_pred))

    mse_promedio = suma_error2 / len(test_set) if test_set else 0.0
    return mse_promedio, lista_predicciones


def graficar_tendencia(red, todos_los_puntos, rango_x=(-3, 4), paso=0.05):
    """
    Dibuja los puntos originales y superpone la curva de predicción de la red.
    """
    xs_orig = [p[0] for p in todos_los_puntos]
    ys_orig = [p[1] for p in todos_los_puntos]

    plt.figure(figsize=(10, 6))
    plt.scatter(xs_orig, ys_orig, color='blue', s=20, alpha=0.5, label="Datos originales")

    x_min, x_max = rango_x
    xs_curva = []
    ys_curva = []
    x_val = x_min
    while x_val <= x_max:
        y_val = forward(red, x_val)["y_pred"]
        xs_curva.append(x_val)
        ys_curva.append(y_val)
        x_val += paso

    plt.plot(xs_curva, ys_curva, color='red', linewidth=2, label="Predicción MLP")
    plt.xlim(x_min, x_max)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Regresión multicapa (MLP) — Curva de tendencia")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Si quieres reproducir exactamente el mismo comportamiento, fijamos semilla:
    if SEED is not None:
        random.seed(SEED)

    # 1) Leer datos
    todos_los_puntos = leer_puntos_desde_archivo(ARCHIVO_DATOS)
    print(f"Total de puntos leídos: {len(todos_los_puntos)}")

    # 2) Dividir en entrenamiento / prueba
    train_set, test_set = dividir_dataset(
        todos_los_puntos,
        porcentaje_entrenamiento=PORCENTAJE_TRAIN,
        semilla=SEED
    )
    print(f"Puntos para entrenamiento: {len(train_set)} — Puntos para prueba: {len(test_set)}")

    # 3) Construir la arquitectura completa: [n_entrada] + CAPAS_OCULTAS + [n_salida]
    layer_sizes = [1] + CAPAS_OCULTAS + [1]
    print(f"Arquitectura (capa a capa): {layer_sizes}")
    W, b = inicializar_red(layer_sizes)
    red = {"W": W, "b": b, "layer_sizes": layer_sizes}

    # 4) Entrenar
    print("Comenzando entrenamiento...")
    historial = entrenar_red(red, train_set, epochs=EPOCHS, lr=LEARNING_RATE)

    # 5) Evaluar en test
    mse_test, _ = evaluar_red(red, test_set)
    print(f"Error MSE en conjunto de prueba: {mse_test:.6f}")

    # 6) Graficar pérdida durante entrenamiento
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), historial, color='green')
    plt.xlabel("Época")
    plt.ylabel("Loss promedio")
    plt.title("Convergencia de la pérdida durante entrenamiento")
    plt.grid(True)
    plt.show()

    # 7) Graficar la tendencia aprendida sobre todo el dataset
    graficar_tendencia(red, todos_los_puntos, rango_x=(-3, 4), paso=0.05)
