import pygame, time, numpy as np
import heapq

# Calcula la distancia Manhattan entre dos puntos
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

# Anima visualmente la ruta calculada con A*
def animar_camino(camino, almacen, num_producto, pos_inicio=(5, 0)):
    pygame.init()
    tamano_celda = 40
    filas, columnas = len(almacen), len(almacen[0])
    pantalla = pygame.display.set_mode((columnas * tamano_celda, filas * tamano_celda))
    pygame.display.set_caption("Simulador de Ruta en Almacén")
    reloj = pygame.time.Clock()
    
    # Dibuja la animación inicialmente
    paso_actual = 0
    
    ejecutando = True
    pausado = False
    animacion_completa = False
    
    while ejecutando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                ejecutando = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_r:  # Reiniciar animación
                    paso_actual = 0
                    animacion_completa = False
        
        # Actualiza solo si no está pausado y la animación no está completa
        if not pausado and not animacion_completa:
            pantalla.fill((255, 255, 255))
            
            # Dibuja el almacén
            for y in range(filas):
                for x in range(columnas):
                    celda = almacen[y][x]
                    # Color base para celdas
                    if celda == '.':
                        color = (200, 200, 200)  # Espacios libres
                    elif celda == num_producto:
                        color = (0, 255, 0)      # Producto seleccionado
                    elif celda.isdigit():
                        color = (150, 150, 150)  # Otros productos
                    else:
                        color = (100, 100, 100)  # Obstáculos
                    
                    pygame.draw.rect(pantalla, color, (x*tamano_celda, y*tamano_celda, tamano_celda, tamano_celda))
                    
                    # Etiquetas para productos
                    if celda.isdigit():
                        fuente = pygame.font.Font(None, 24)
                        texto = fuente.render(celda, True, (0, 0, 0))
                        pantalla.blit(texto, (x*tamano_celda+10, y*tamano_celda+10))
            
            # Dibuja la ruta hasta el punto actual
            for i in range(min(paso_actual + 1, len(camino))):
                y, x = camino[i]
                pygame.draw.rect(pantalla, (100, 100, 255), (x*tamano_celda, y*tamano_celda, tamano_celda, tamano_celda))
                
                # Marca posición actual con círculo
                if i == paso_actual:
                    pygame.draw.circle(pantalla, (0, 0, 0), 
                                      (x*tamano_celda + tamano_celda//2, y*tamano_celda + tamano_celda//2), 
                                      tamano_celda//4)
            
            # Marca posición inicial (C)
            fuente = pygame.font.Font(None, 36)
            texto = fuente.render('C', True, (0, 0, 0))
            pygame.draw.rect(pantalla, (255, 255, 0), (pos_inicio[1]*tamano_celda, pos_inicio[0]*tamano_celda, tamano_celda, tamano_celda))
            pantalla.blit(texto, (pos_inicio[1]*tamano_celda+10, pos_inicio[0]*tamano_celda+10))
            
            # Avanza un paso en la animación
            if paso_actual < len(camino) - 1:
                paso_actual += 1
            else:
                animacion_completa = True
                
            pygame.display.flip()
            reloj.tick(10)  # Velocidad de animación
        else:
            # Si está pausado, solo actualiza eventos para seguir respondiendo
            reloj.tick(30)
            
    pygame.quit()

def principal():
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
    
    # Obtiene entrada del usuario para un único producto
    pos_inicio = (5, 0)  # Posición inicial del recolector
    
    while True:
        num_producto = input("Ingrese número de producto (1-48): ").strip()
        
        if not num_producto.isdigit():
            print("Entrada inválida. Por favor ingrese un número entre 1 y 48.")
            continue  # Repite la solicitud de entrada

        if not (1 <= int(num_producto) <= 48):  # Verifica si está en el rango
            print("Número de producto inválido. Por favor ingrese un número entre 1 y 48.")
            continue

        break  # Sale del bucle si la entrada es válida

    # Encuentra punto de acceso para el producto
    mapa_binario = convertir_a_mapa_binario(almacen)
    punto_acceso = encontrar_punto_acceso(almacen, num_producto)

    if not punto_acceso:
        print(f"Error: No hay punto de acceso para el producto {num_producto}")
        return 

    # Calcula ruta con A*
    camino = a_estrella(mapa_binario, pos_inicio, punto_acceso)

    if not camino:
        print(f"No se encontró camino hacia el producto {num_producto}")
        return

    print(f"¡Camino encontrado! Longitud: {len(camino)-1} pasos")
    print("Presione R para ver la animación")

    # Anima la ruta
    animar_camino(camino, almacen, num_producto, pos_inicio)

if __name__ == "__main__":
    principal()