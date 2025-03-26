import heapq, random, math, pygame, time, numpy as np
from functools import lru_cache

# Calcula la distancia Manhattan entre dos puntos (suma de diferencias absolutas en x e y)
def distancia_manhattan(p1, p2): return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

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
    vino_de, puntuacion_g, puntuacion_f = {}, {inicio: 0}, {inicio: distancia_manhattan(inicio, fin)}
    
    while conjunto_abierto:
        _, actual = heapq.heappop(conjunto_abierto)  # Obtiene nodo con menor puntuacion_f
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
                tentativa_g = puntuacion_g[actual] + 1
                if vecino not in puntuacion_g or tentativa_g < puntuacion_g[vecino]:
                    vino_de[vecino] = actual
                    puntuacion_g[vecino] = tentativa_g
                    puntuacion_f[vecino] = tentativa_g + distancia_manhattan(vecino, fin)
                    heapq.heappush(conjunto_abierto, (puntuacion_f[vecino], vecino))
    return []  # No se encontró camino

# Precalcula todos los caminos posibles entre posiciones importantes
def precalcular_caminos(mapa_binario, posiciones):
    todos_los_caminos = {}
    puntos = list(posiciones.values())
    for i, inicio in enumerate(puntos):
        for j, fin in enumerate(puntos):
            if i != j: todos_los_caminos[(inicio, fin)] = a_estrella(mapa_binario, inicio, fin)
    return todos_los_caminos

# Genera matriz de distancias entre todos los puntos
def calcular_matriz_distancia(todos_los_caminos):
    puntos = set()
    for inicio, fin in todos_los_caminos.keys():
        puntos.add(inicio)
        puntos.add(fin)
    
    puntos = list(puntos)
    n = len(puntos)
    matriz_dist = np.full((n, n), np.inf)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                camino = todos_los_caminos.get((puntos[i], puntos[j]), [])
                if camino: matriz_dist[i, j] = len(camino) - 1
    
    return matriz_dist, {punto: i for i, punto in enumerate(puntos)}

# Calcula el costo total de una ruta
def calcular_costo(ruta, posiciones, matriz_dist, indices_puntos):
    costo = 0
    indice_actual = indices_puntos[posiciones['C']]  # Comienza desde el punto C (inicio)
    for item in ruta:
        indice_siguiente = indices_puntos[posiciones[item]]
        if matriz_dist[indice_actual, indice_siguiente] == np.inf: return float('inf')
        costo += matriz_dist[indice_actual, indice_siguiente]
        indice_actual = indice_siguiente
    return costo

# Genera variaciones de una ruta para explorar soluciones alternativas
def generar_vecinos(ruta, num=5):
    vecinos = []
    for _ in range(num):
        nueva_ruta = ruta.copy()
        op = random.randint(0, 2)
        
        if op == 0:  # Operación: Intercambiar dos elementos
            i, j = random.sample(range(len(nueva_ruta)), 2)
            nueva_ruta[i], nueva_ruta[j] = nueva_ruta[j], nueva_ruta[i]
        elif op == 1 and len(nueva_ruta) > 1:  # Operación: Mover un elemento a otra posición
            i, j = random.sample(range(len(nueva_ruta)), 2)
            item = nueva_ruta.pop(i)
            nueva_ruta.insert(j, item)
        elif op == 2 and len(nueva_ruta) > 2:  # Operación: Invertir una sección
            i, j = sorted(random.sample(range(len(nueva_ruta)), 2))
            nueva_ruta[i:j+1] = reversed(nueva_ruta[i:j+1])
        vecinos.append(nueva_ruta)
    return vecinos

# Algoritmo de recocido simulado para encontrar una ruta óptima
def recocido_simulado(ruta_inicial, posiciones, matriz_dist, indices_puntos, 
                      temp=1000, enfriamiento=0.6, iteraciones=100, max_estancamiento=20):
    mejor_ruta = ruta_actual = ruta_inicial.copy()
    mejor_costo = costo_actual = calcular_costo(ruta_actual, posiciones, matriz_dist, indices_puntos)
    estancamiento = 0
    
    while temp > 0.1 and estancamiento < max_estancamiento:
        mejorado = False
        
        for _ in range(iteraciones):
            vecinos = generar_vecinos(ruta_actual, num=5)
            
            for vecino in vecinos:
                nuevo_costo = calcular_costo(vecino, posiciones, matriz_dist, indices_puntos)
                if nuevo_costo == float('inf'): continue
                delta = nuevo_costo - costo_actual
                
                # Acepta solución si es mejor o con probabilidad basada en temperatura
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    ruta_actual, costo_actual = vecino, nuevo_costo
                    if nuevo_costo < mejor_costo:
                        mejor_ruta, mejor_costo = vecino.copy(), nuevo_costo
                        mejorado = True
                    if delta < 0: break
        
        estancamiento = 0 if mejorado else estancamiento + 1
        temp *= enfriamiento  # Enfría la temperatura gradualmente
    
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

# Interfaz gráfica para seleccionar productos usando Pygame
def seleccionar_productos(almacen):
    pygame.init()
    print("\nHaz clic en los productos, presiona ENTER cuando hayas terminado.\n")
    tamano_celda = 40
    filas, columnas = len(almacen), len(almacen[0])
    pantalla = pygame.display.set_mode((columnas * tamano_celda, filas * tamano_celda))
    reloj = pygame.time.Clock()

    productos = []
    posiciones_seleccionadas = []
    ejecutando = True

    while ejecutando:
        pantalla.fill((255, 255, 255))
        
        # Dibuja almacén
        for y in range(filas):
            for x in range(columnas):
                color = (200, 200, 200) if almacen[y][x] == '.' else (100, 100, 100)
                pygame.draw.rect(pantalla, color, (x*tamano_celda, y*tamano_celda, tamano_celda, tamano_celda))
                
                if almacen[y][x].isdigit() and almacen[y][x] != 'C':
                    if (x, y) in posiciones_seleccionadas:
                        pygame.draw.rect(pantalla, (0, 255, 0), (x*tamano_celda, y*tamano_celda, tamano_celda, tamano_celda))
                    
                    fuente = pygame.font.Font(None, 24)
                    texto = fuente.render(almacen[y][x], True, (0, 0, 0))
                    pantalla.blit(texto, (x*tamano_celda+10, y*tamano_celda+10))

        # Procesa eventos
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                ejecutando = False
            if evento.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                x, y = x//tamano_celda, y//tamano_celda
                if almacen[y][x].isdigit() and almacen[y][x] != 'C':
                    estante = almacen[y][x]
                    if estante not in productos:
                        productos.append(estante)
                        posiciones_seleccionadas.append((x, y))
            if evento.type == pygame.KEYDOWN and evento.key == pygame.K_RETURN:
                ejecutando = False
                print("Productos seleccionados:", productos)
                print("\nPor favor espera. Buscando la mejor ruta...\n")
                
        pygame.display.flip()
        reloj.tick(30)
    
    pygame.quit()
    return productos, posiciones_seleccionadas

# Anima visualmente la ruta calculada
def animar_ruta(ruta, posiciones, almacen, posiciones_seleccionadas):
    pygame.init()
    tamano_celda = 40
    filas, columnas = len(almacen), len(almacen[0])
    pantalla = pygame.display.set_mode((columnas * tamano_celda, filas * tamano_celda))
    reloj = pygame.time.Clock()

    segmentos_ruta = []
    colores = []
    
    # Precalcula segmentos de ruta
    mapa_binario = convertir_a_mapa_binario(almacen)
    ruta_completa = ['C'] + ruta
    
    for i in range(len(ruta_completa) - 1):
        camino = a_estrella(mapa_binario, posiciones[ruta_completa[i]], posiciones[ruta_completa[i+1]])
        if not camino:
            print(f"No se encontró camino entre {ruta_completa[i]} y {ruta_completa[i+1]}")
            return
        segmentos_ruta.append(camino)
        colores.append((random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))

    ejecutando = True

    # Muestra cada segmento de la ruta
    for i, camino in enumerate(segmentos_ruta):
        pantalla.fill((255, 255, 255))

        # Redibuja almacén
        for y in range(filas):
            for x in range(columnas):
                color = (200, 200, 200) if almacen[y][x] == '.' else (100, 100, 100)
                pygame.draw.rect(pantalla, color, (x*tamano_celda, y*tamano_celda, tamano_celda, tamano_celda))

                if (x, y) in posiciones_seleccionadas:
                    pygame.draw.rect(pantalla, (0, 255, 0), (x*tamano_celda, y*tamano_celda, tamano_celda, tamano_celda))

                if almacen[y][x].isdigit():
                    fuente = pygame.font.Font(None, 24)
                    texto = fuente.render(almacen[y][x], True, (0, 0, 0))
                    pantalla.blit(texto, (x*tamano_celda+10, y*tamano_celda+10))
        
        # Dibuja segmentos anteriores
        for j in range(i):
            for v in segmentos_ruta[j]:
                pygame.draw.rect(pantalla, colores[j], (v[1]*tamano_celda, v[0]*tamano_celda, tamano_celda, tamano_celda))

        # Dibuja segmento actual
        for j in range(len(camino)):
            pygame.draw.rect(pantalla, colores[i], (camino[j][1]*tamano_celda, camino[j][0]*tamano_celda, tamano_celda, tamano_celda))
            
            # Dibuja agente en último punto
            if j == len(camino) - 1:
                pygame.draw.circle(pantalla, (0, 0, 0), 
                                  (camino[j][1]*tamano_celda + tamano_celda//2, camino[j][0]*tamano_celda + tamano_celda//2), 
                                  tamano_celda//4)
                
            # Muestra posición inicial
            inicio_x, inicio_y = posiciones['C']
            fuente = pygame.font.Font(None, 36)
            texto = fuente.render('C', True, (0, 0, 0))
            pygame.draw.rect(pantalla, (255, 255, 0), (inicio_y*tamano_celda, inicio_x*tamano_celda, tamano_celda, tamano_celda))
            pantalla.blit(texto, (inicio_y*tamano_celda+10, inicio_x*tamano_celda+10))

            pygame.display.flip()
            reloj.tick(8)
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    ejecutando = False
                    break
        
        time.sleep(0.5)

    # Bucle final para cerrar ventana
    while ejecutando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                ejecutando = False

    pygame.quit()

def main():
    # Inicializa el almacén
    almacen = [
        list("............."),
        list("..##..##..##."),
        list("..##..##..##."),
        list("..##..##..##."),
        list("..##..##..##."),
        list("............."),
        list("..##..##..##."),
        list("..##..##..##."),
        list("..##..##..##."),
        list("..##..##..##."),
        list(".............")
    ]

    # Numera los estantes
    contador = 1
    ancho_bloque, alto_bloque = 2, 5
    filas, columnas = len(almacen), len(almacen[0])
    estantes = {(y, x) for y in range(filas) for x in range(columnas) if almacen[y][x] == '#'}
    
    visitados = set()
    for y in range(0, filas, alto_bloque):
        for x in range(0, columnas, ancho_bloque):
            bloque = [(ny, nx) for dy in range(alto_bloque) for dx in range(ancho_bloque)
                    if (ny := y + dy) < filas and (nx := x + dx) < columnas 
                    and (ny, nx) in estantes and (ny, nx) not in visitados]
            if bloque:
                for ny, nx in bloque:
                    almacen[ny][nx] = str(contador)
                    contador += 1
                    visitados.add((ny, nx))

    # Selecciona productos y prepara
    productos, posiciones_seleccionadas = seleccionar_productos(almacen)
    posiciones = {'C': (5, 0)}  # Posición inicial del recolector
    
    # Encuentra puntos de acceso
    print("Buscando puntos de acceso horizontales...")
    for producto in productos[:]:
        punto_acceso = encontrar_punto_acceso(almacen, producto)
        if punto_acceso:
            posiciones[producto] = punto_acceso
        else:
            print(f"ADVERTENCIA: No hay acceso horizontal al producto {producto}. Omitiendo.")
            productos.remove(producto)
    
    if not productos:
        print("ERROR: No se seleccionaron productos accesibles.")
        return
    
    # Calcula ruta óptima
    print("Iniciando cálculo de ruta óptima...")
    tiempo_inicio = time.time()
    
    mapa_binario = convertir_a_mapa_binario(almacen)
    print("Precalculando todos los caminos...")
    todos_los_caminos = precalcular_caminos(mapa_binario, posiciones)
    
    print("Generando matriz de distancias...")
    matriz_dist, indices_puntos = calcular_matriz_distancia(todos_los_caminos)
    
    print("Ejecutando recocido simulado...")
    mejor_ruta = recocido_simulado(productos, posiciones, matriz_dist, indices_puntos)
    
    tiempo_fin = time.time()
    print(f"Tiempo de cálculo: {tiempo_fin - tiempo_inicio:.2f} segundos")
    
    # Calcula costo real
    costo_real = 0
    actual = posiciones['C']
    for item in mejor_ruta:
        camino = a_estrella(mapa_binario, actual, posiciones[item])
        if camino:
            costo_real += len(camino) - 1
            actual = posiciones[item]
        else:
            print(f"ADVERTENCIA: No se encontró camino hacia {item}")
    
    print("Mejor ruta:", mejor_ruta)
    print("Costo total de la ruta:", costo_real)
    
    # Anima la ruta
    animar_ruta(mejor_ruta, posiciones, almacen, posiciones_seleccionadas)

if __name__ == "__main__":
    main()