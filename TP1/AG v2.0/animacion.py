import pygame
import csv
from collections import Counter
import re

# Inicializar pygame
pygame.init()

# Dimensiones de la pantalla
ANCHO, ALTO = 800, 600
pantalla = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Mapa de Calor del Almacén")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (200, 200, 200)

# Definir la estructura del diseño del almacén
plantilla_almacen = [
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

def analizar_diseño(cadena_diseño):
    """Analizar la cadena de diseño en una cuadrícula 2D con números de artículos,
    dividiendo números unidos que excedan el rango válido (1-48)"""
    # Crear una cuadrícula basada en la plantilla del almacén
    cuadricula = []
    for fila in plantilla_almacen:
        cuadricula.append([0 if celda == '.' else None for celda in fila])
    
    # Encontrar todas las posiciones donde se pueden colocar artículos (donde hay un '#' en la plantilla)
    posiciones_articulos = []
    for y, fila in enumerate(plantilla_almacen):
        for x, celda in enumerate(fila):
            if celda == '#':
                posiciones_articulos.append((y, x))
    
    # Extraer y procesar números de la cadena de diseño
    numeros_brutos = re.findall(r'\d+', cadena_diseño)
    numeros_procesados = []
    
    for cadena_num in numeros_brutos:
        num = int(cadena_num)
        
        # Número inválido, necesita ser dividido
        digitos = [int(d) for d in cadena_num]
                
        # Intentar combinar dígitos para formar números válidos
        i = 0
        while i < len(digitos):
            if i + 1 < len(digitos) and 1 <= digitos[i] * 10 + digitos[i+1] <= 48:
                # Combinar dos dígitos si el resultado está entre 1-48
                numeros_procesados.append(digitos[i] * 10 + digitos[i+1])
                i += 2
            else:
                # Usar dígito individual si la combinación no funciona
                if 1 <= digitos[i] <= 9:  # Solo agregar si es un número válido de un solo dígito
                    numeros_procesados.append(digitos[i])
                i += 1
    
    # Asignar números procesados a posiciones de artículos
    for i, (y, x) in enumerate(posiciones_articulos):
        if i < len(numeros_procesados):
            cuadricula[y][x] = numeros_procesados[i]
        else:
            cuadricula[y][x] = 0  # Valor predeterminado si nos quedamos sin números
    
    return cuadricula

def cargar_datos():
    """Cargar datos desde archivos CSV"""
    diseños = []
    with open('resultados.csv', 'r') as archivo:
        lector_csv = csv.reader(archivo)
        next(lector_csv)  # Saltar encabezado
        for fila in lector_csv:
            generacion = int(fila[0])
            costo = float(fila[1])
            cadena_diseño = fila[2]
            diseños.append((generacion, costo, cadena_diseño))
    
    pedidos = []
    with open('ordenes.csv', 'r') as archivo:
        lector_csv = csv.reader(archivo)
        for fila in lector_csv:
            pedido = [int(articulo) for articulo in fila]
            pedidos.extend(pedido)
    
    return diseños, pedidos

def obtener_frecuencia_articulos(pedidos):
    """Calcular la frecuencia de cada artículo en los pedidos"""
    return Counter(pedidos)

def crear_mapa_colores(frecuencia_articulos):
    """Crear un mapa de colores basado en la frecuencia de artículos"""
    frecuencia_max = max(frecuencia_articulos.values()) if frecuencia_articulos else 1
    mapa_colores = {}
    
    for articulo, frecuencia in frecuencia_articulos.items():
        # Crear un color de calor desde azul (frío) hasta rojo (caliente)
        intensidad = int(255 * (frecuencia / frecuencia_max))
        mapa_colores[articulo] = (intensidad, 0, 255 - intensidad)
    
    return mapa_colores

def dibujar_almacen(diseño, mapa_colores, frecuencia_articulos, generacion, costo):
    """Dibujar el diseño del almacén con mapa de calor"""
    pantalla.fill(BLANCO)
    
    # Dibujar título e información de generación
    fuente = pygame.font.SysFont('Arial', 24)
    titulo = fuente.render(f"Mapa de Calor del Almacén - Generación {generacion} (Costo: {costo})", True, NEGRO)
    pantalla.blit(titulo, (20, 20))
    
    # Calcular tamaño de celda y desplazamiento
    tamaño_celda = min(ANCHO // 15, ALTO // 15)
    desplazamiento_x = (ANCHO - 13 * tamaño_celda) // 2
    desplazamiento_y = 80
    
    # Dibujar la cuadrícula del almacén
    for y, fila in enumerate(diseño):
        for x, articulo in enumerate(fila):
            rect = pygame.Rect(desplazamiento_x + x * tamaño_celda, desplazamiento_y + y * tamaño_celda, tamaño_celda, tamaño_celda)
            
            if articulo == 0:  # Espacio vacío
                pygame.draw.rect(pantalla, BLANCO, rect)
                pygame.draw.rect(pantalla, GRIS, rect, 1)
            else:
                color = mapa_colores.get(articulo, (100, 100, 100))
                pygame.draw.rect(pantalla, color, rect)
                pygame.draw.rect(pantalla, NEGRO, rect, 1)
                
                fuente_pequeña = pygame.font.SysFont('Arial', tamaño_celda // 2)
                texto_articulo = fuente_pequeña.render(str(articulo), True, BLANCO)
                rect_texto = texto_articulo.get_rect(center=rect.center)
                pantalla.blit(texto_articulo, rect_texto)
    
    # Dibujar leyenda
    fuente_leyenda = pygame.font.SysFont('Arial', 18)
    texto_leyenda = fuente_leyenda.render("La intensidad del color indica la frecuencia en los pedidos", True, NEGRO)
    pantalla.blit(texto_leyenda, (20, ALTO - 40))
    
    pygame.display.flip()

def mostrar_comparacion(primer_diseño, ultimo_diseño, frecuencia_articulos, costo1, costo2):
    """Mostrar comparación entre el primer y último diseño"""
    pantalla.fill(BLANCO)

    # Dibujar título
    fuente = pygame.font.SysFont('Arial', 24)
    
    # Calcular tamaño de celda y desplazamiento
    tamaño_celda = min(ANCHO // 30, ALTO // 15)
    desplazamiento_x1 = 20
    desplazamiento_x2 = ANCHO // 2 + 20
    desplazamiento_y = 80
    
    mapa_colores = crear_mapa_colores(frecuencia_articulos)
    
    # Dibujar diseño original
    subtitulo = fuente.render("Diseño Original", True, NEGRO)
    pantalla.blit(subtitulo, (desplazamiento_x1, 50))

    fuente_costo1 = pygame.font.SysFont('Arial', 18)
    texto_costo1 = fuente_costo1.render(f"Costo: {costo1} ", True, NEGRO)
    pantalla.blit(texto_costo1, (desplazamiento_x1, 380))
    
    for y, fila in enumerate(primer_diseño):
        for x, articulo in enumerate(fila):
            rect = pygame.Rect(desplazamiento_x1 + x * tamaño_celda, desplazamiento_y + y * tamaño_celda, tamaño_celda, tamaño_celda)
            
            if articulo == 0:  # Espacio vacío
                pygame.draw.rect(pantalla, BLANCO, rect)
                pygame.draw.rect(pantalla, GRIS, rect, 1)
            else:
                color = mapa_colores.get(articulo, (100, 100, 100))
                pygame.draw.rect(pantalla, color, rect)
                pygame.draw.rect(pantalla, NEGRO, rect, 1)
                
                fuente_pequeña = pygame.font.SysFont('Arial', tamaño_celda // 2)
                texto_articulo = fuente_pequeña.render(str(articulo), True, BLANCO)
                rect_texto = texto_articulo.get_rect(center=rect.center)
                pantalla.blit(texto_articulo, rect_texto)
    
    # Dibujar diseño final
    subtitulo = fuente.render("Diseño Final", True, NEGRO)
    pantalla.blit(subtitulo, (desplazamiento_x2, 50))
    
    fuente_costo = pygame.font.SysFont('Arial', 18)
    texto_costo = fuente_costo.render(f"Costo: {costo2}", True, NEGRO)
    pantalla.blit(texto_costo, (desplazamiento_x2, 380))
    
    for y, fila in enumerate(ultimo_diseño):
        for x, articulo in enumerate(fila):
            rect = pygame.Rect(desplazamiento_x2 + x * tamaño_celda, desplazamiento_y + y * tamaño_celda, tamaño_celda, tamaño_celda)
            
            if articulo == 0:  # Espacio vacío
                pygame.draw.rect(pantalla, BLANCO, rect)
                pygame.draw.rect(pantalla, GRIS, rect, 1)
            else:
                color = mapa_colores.get(articulo, (100, 100, 100))
                pygame.draw.rect(pantalla, color, rect)
                pygame.draw.rect(pantalla, NEGRO, rect, 1)
                
                fuente_pequeña = pygame.font.SysFont('Arial', tamaño_celda // 2)
                texto_articulo = fuente_pequeña.render(str(articulo), True, BLANCO)
                rect_texto = texto_articulo.get_rect(center=rect.center)
                pantalla.blit(texto_articulo, rect_texto)
    
    fuente_leyenda = pygame.font.SysFont('Arial', 18)
    texto_leyenda = fuente_leyenda.render("La intensidad del color indica la frecuencia en los pedidos", True, NEGRO)
    pantalla.blit(texto_leyenda, (20, ALTO - 40))
    
    pygame.display.flip()

def main():
    diseños, pedidos = cargar_datos()
    frecuencia_articulos = obtener_frecuencia_articulos(pedidos)
    mapa_colores = crear_mapa_colores(frecuencia_articulos)
    
    primer_diseño = analizar_diseño(diseños[0][2])
    ultimo_diseño = analizar_diseño(diseños[-1][2])
    
    reloj = pygame.time.Clock()
    ejecutando = True
    mostrar_vista_comparacion = False
    indice_diseño_actual = 0
    
    while ejecutando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                ejecutando = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_RIGHT:
                    # Mover al siguiente diseño
                    indice_diseño_actual = min(indice_diseño_actual + 1, len(diseños) - 1)
                    mostrar_vista_comparacion = False
                elif evento.key == pygame.K_LEFT:
                    # Mover al diseño anterior
                    indice_diseño_actual = max(indice_diseño_actual - 1, 0)
                    mostrar_vista_comparacion = False
                elif evento.key == pygame.K_c:
                    # Alternar vista de comparación
                    mostrar_vista_comparacion = not mostrar_vista_comparacion
                elif evento.key == pygame.K_ESCAPE:
                    ejecutando = False
        
        if mostrar_vista_comparacion:
            _, costo1,_ = diseños[0]
            _, costo2,_ = diseños[-1]
            mostrar_comparacion(primer_diseño, ultimo_diseño, frecuencia_articulos, costo1, costo2)
        else:
            generacion, costo, cadena_diseño = diseños[indice_diseño_actual]
            diseño_actual = analizar_diseño(cadena_diseño)
            dibujar_almacen(diseño_actual, mapa_colores, frecuencia_articulos, generacion, costo)
        
        reloj.tick(30)
    
    pygame.quit()

if __name__ == "__main__":
    main()
