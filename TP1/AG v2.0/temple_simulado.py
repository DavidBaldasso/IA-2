import heapq, random, math, time, numpy as np
from functools import lru_cache

# Calcula la distancia Manhattan entre dos puntos (suma de diferencias absolutas en x e y)
def distancia_manhattan(p1, p2): 
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

# Convierte el mapa del almacén a una matriz binaria (0 = espacio libre, 1 = obstáculo)
def convertir_a_mapa_binario(almacen):
    binario = np.zeros((len(almacen), len(almacen[0])), dtype=np.int8)
    for y in range(len(almacen)):
        for x in range(len(almacen[0])):
            binario[y, x] = 0 if almacen[y][x] == '.' else 1
    return binario

# Implementación del algoritmo A* para encontrar el camino más corto entre dos puntos
def a_estrella(mapa_binario, inicio, fin):
    filas, columnas = mapa_binario.shape
    movimientos = [(-1,0), (1,0), (0,-1), (0,1)]  # Movimientos posibles: arriba, abajo, izquierda, derecha
    conjunto_abierto = [(0, inicio)]  # Cola de prioridad para explorar nodos
    vino_de, puntaje_g, puntaje_f = {}, {inicio: 0}, {inicio: distancia_manhattan(inicio, fin)}
    
    while conjunto_abierto:
        _, actual = heapq.heappop(conjunto_abierto)  # Obtiene nodo con menor puntaje_f
        if actual == fin:  # Si llegamos al destino, reconstruimos el camino
            camino = [actual]
            while actual in vino_de:
                actual = vino_de[actual]
                camino.append(actual)
            return camino[::-1]  # Devuelve el camino invertido (del inicio al final)
        
        # Explora vecinos
        for dy, dx in movimientos:
            vecino = (actual[0]+dy, actual[1]+dx)
            if (0 <= vecino[0] < filas and 0 <= vecino[1] < columnas and 
                mapa_binario[vecino[0], vecino[1]] == 0):  # Si es válido y transitable
                tentativo_g = puntaje_g[actual] + 1
                if vecino not in puntaje_g or tentativo_g < puntaje_g[vecino]:
                    vino_de[vecino] = actual
                    puntaje_g[vecino] = tentativo_g
                    puntaje_f[vecino] = tentativo_g + distancia_manhattan(vecino, fin)
                    heapq.heappush(conjunto_abierto, (puntaje_f[vecino], vecino))
    return []  # No se encontró camino

# Precalcula todos los caminos posibles entre posiciones importantes. Esto evita tener que calcular la misma ruta múltiples veces.
def precalcular_caminos(mapa_binario, posiciones):
    todos_los_caminos = {}
    puntos = list(posiciones.values())
    for i, inicio in enumerate(puntos):
        for j, fin in enumerate(puntos):
            if i != j: todos_los_caminos[(inicio, fin)] = a_estrella(mapa_binario, inicio, fin)
    return todos_los_caminos

# Genera matriz de distancias entre todos los puntos
def calcular_matriz_distancias(todos_los_caminos):
    puntos = set()
    for inicio, fin in todos_los_caminos.keys():
        puntos.add(inicio)
        puntos.add(fin)
    
    puntos = list(puntos)
    n = len(puntos)
    matriz_distancias = np.full((n, n), np.inf)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                camino = todos_los_caminos.get((puntos[i], puntos[j]), [])
                if camino: matriz_distancias[i, j] = len(camino) - 1
    
    return matriz_distancias, {punto: i for i, punto in enumerate(puntos)}

# Calcula el costo total de una ruta
def calcular_costo(ruta, posiciones, matriz_distancias, indices_puntos):
    costo = 0
    indice_actual = indices_puntos[posiciones['C']]  # Comienza desde el punto C (inicio)
    for item in ruta:
        indice_siguiente = indices_puntos[posiciones[item]]
        if matriz_distancias[indice_actual, indice_siguiente] == np.inf: return float('inf')
        costo += matriz_distancias[indice_actual, indice_siguiente]
        indice_actual = indice_siguiente
    return costo

# Genera variaciones de una ruta para explorar soluciones alternativas
def generar_vecinos(ruta, num=5):
    vecinos = []
    for _ in range(num):
        nueva_ruta = ruta.copy()
        operacion = random.randint(0, 2)
        
        if operacion == 0:  # Operación: Intercambiar dos elementos
            i, j = random.sample(range(len(nueva_ruta)), 2)
            nueva_ruta[i], nueva_ruta[j] = nueva_ruta[j], nueva_ruta[i]
        elif operacion == 1 and len(nueva_ruta) > 1:  # Operación: Mover un elemento a otra posición
            i, j = random.sample(range(len(nueva_ruta)), 2)
            item = nueva_ruta.pop(i)
            nueva_ruta.insert(j, item)
        elif operacion == 2 and len(nueva_ruta) > 2:  # Operación: Invertir una sección
            i, j = sorted(random.sample(range(len(nueva_ruta)), 2))
            nueva_ruta[i:j+1] = reversed(nueva_ruta[i:j+1])
        vecinos.append(nueva_ruta)
    return vecinos

# Algoritmo de recocido simulado para encontrar una ruta óptima
def recocido_simulado(ruta_inicial, posiciones, matriz_distancias, indices_puntos, 
                      temperatura=1000, enfriamiento=0.6, iteraciones=100, max_estancamiento=20):
    """Args:
    - ruta_inicial: Ruta inicial
    - posiciones: Diccionario de posiciones
    - matriz_distancias: Matriz de distancias
    - indices_puntos: Mapeo de puntos a índices
    - temperatura: Temperatura inicial
    - enfriamiento: Factor de enfriamiento
    - iteraciones: Iteraciones por temperatura
    - max_estancamiento: Iteraciones máximas sin mejora
    """
    mejor_ruta = ruta_actual = ruta_inicial.copy()
    mejor_costo = costo_actual = calcular_costo(ruta_actual, posiciones, matriz_distancias, indices_puntos)
    estancamiento = 0
    
    while temperatura > 0.1 and estancamiento < max_estancamiento:
        mejora = False
        
        for _ in range(iteraciones):
            vecinos = generar_vecinos(ruta_actual, num=5)
            
            for vecino in vecinos:
                nuevo_costo = calcular_costo(vecino, posiciones, matriz_distancias, indices_puntos)
                if nuevo_costo == float('inf'): continue
                delta = nuevo_costo - costo_actual
                
                # Acepta solución si es mejor o con probabilidad basada en temperatura
                if delta < 0 or random.random() < math.exp(-delta / temperatura):
                    ruta_actual, costo_actual = vecino, nuevo_costo
                    if nuevo_costo < mejor_costo:
                        mejor_ruta, mejor_costo = vecino.copy(), nuevo_costo
                        mejora = True
                    if delta < 0: break
        
        estancamiento = 0 if mejora else estancamiento + 1
        temperatura *= enfriamiento  # Enfría la temperatura gradualmente
    
    return mejor_ruta

# Encuentra punto de acceso para un producto (ubicación adyacente al estante)
def encontrar_punto_acceso(almacen, producto):
    filas, columnas = len(almacen), len(almacen[0])
    puntos_producto = [(y, x) for y in range(filas) for x in range(columnas) 
                     if almacen[y][x] == producto]
    
    for y, x in puntos_producto:
        for dx in [-1, 1]:  # Busca espacio libre a la izquierda o derecha
            nx = x + dx
            if 0 <= nx < columnas and almacen[y][nx] == '.':
                return (y, nx)
    return None

# Procesa una lista de productos y devuelve los resultados
def procesar_lista_productos(args):
    indice_lista, productos, almacen, mapa_binario = args
    
    resultados = {
        "indice_lista": indice_lista,
        "productos": productos,
        "mejor_ruta": None,
        "costo_real": 0,
        "tiempo_calculo": 0,
        "productos_accesibles": []
    }
    
    posiciones = {'C': (5, 0)}  # Posición inicial del recolector
    
    # Encuentra puntos de acceso
    for producto in productos[:]:
        punto_acceso = encontrar_punto_acceso(almacen, producto)
        if punto_acceso:
            posiciones[producto] = punto_acceso
            resultados["productos_accesibles"].append(producto)
    
    if not resultados["productos_accesibles"]:
        resultados["error"] = "No se seleccionaron productos accesibles."
        return resultados
    
    # Calcula ruta óptima
    tiempo_inicio = time.time()
    
    todos_los_caminos = precalcular_caminos(mapa_binario, posiciones)
    matriz_distancias, indices_puntos = calcular_matriz_distancias(todos_los_caminos)
    mejor_ruta = recocido_simulado(resultados["productos_accesibles"], posiciones, matriz_distancias, indices_puntos)
    
    tiempo_fin = time.time()
    resultados["tiempo_calculo"] = tiempo_fin - tiempo_inicio
    resultados["mejor_ruta"] = mejor_ruta
    
    # Usar la función calcular_costo que ya existe
    resultados["costo_real"] = calcular_costo(mejor_ruta, posiciones, matriz_distancias, indices_puntos)
    
    return resultados

# Función principal que procesa todas las listas de productos en un almacén
def procesar(almacen, listas_productos=None):
    mapa_binario = convertir_a_mapa_binario(almacen)
    tiempo_calculo_total = 0
    costo_total_rutas = 0
    
    for i, productos in enumerate(listas_productos):
        resultado = procesar_lista_productos((i+1, productos, almacen, mapa_binario))
        
        if "error" in resultado:
            print(f"ERROR: {resultado['error']}")
            continue
            
        tiempo_calculo_total += resultado['tiempo_calculo']
        costo_total_rutas += resultado['costo_real']

    return costo_total_rutas
