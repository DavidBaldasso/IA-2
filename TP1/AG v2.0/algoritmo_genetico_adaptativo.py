import random, copy, time, csv
import numpy as np, multiprocessing as mp
from temple_simulado import procesar

class OptimizadorLayoutAlmacenAG:
    def __init__(self, plantilla_almacen, tamaño_poblacion=10, generaciones=20, 
                 tasa_mutacion=0.2, tasa_cruce=0.7, tamaño_elite=1, num_procesos=None, listas_productos=None,
                 tasas_adaptativas=True, tamaño_torneo=2, incluir_voraz=False):
        """Args:
            plantilla_almacen: Diseño original del almacén con '.' para caminos y '#' para posiciones de estantes
            tamaño_poblacion: Número de individuos en la población
            generaciones: Número máximo de generaciones
            tasa_mutacion: Probabilidad de mutación
            tasa_cruce: Probabilidad de cruce
            tamaño_elite: Número de mejores individuos a preservar en cada generación
            num_procesos: Número de procesos a usar para evaluación de aptitud en paralelo (default: None = número de CPU)
            listas_productos: Lista de órdenes de productos para calcular aptitud
            tasas_adaptativas: Si se ajustan las tasas de mutación y cruce basadas en la diversidad de la población
            tamaño_torneo: Tamaño del torneo para selección
            incluir_voraz: Agrega un individuo artificial con una función de fitness mejorada
        """
        self.plantilla_almacen = copy.deepcopy(plantilla_almacen)
        self.tamaño_poblacion = tamaño_poblacion
        self.generaciones = generaciones
        self.tasa_mutacion_inicial = tasa_mutacion
        self.tasa_mutacion = tasa_mutacion
        self.tasa_cruce_inicial = tasa_cruce
        self.tasa_cruce = tasa_cruce
        self.tamaño_elite = tamaño_elite
        self.listas_productos = listas_productos if listas_productos else []
        self.num_procesos = num_procesos if num_procesos else mp.cpu_count()
        self.tasas_adaptativas = tasas_adaptativas
        self.tamaño_torneo = tamaño_torneo
        self.contador_estancamiento = 0
        self.umbral_estancamiento = 3  # Número de generaciones sin mejora para activar diversificación
        self.incluir_voraz = incluir_voraz
        
        # Extraer posiciones de estantes de la plantilla
        self.posiciones_estantes = []
        for y in range(len(plantilla_almacen)):
            for x in range(len(plantilla_almacen[0])):
                if plantilla_almacen[y][x] == '#':
                    self.posiciones_estantes.append((y, x))
        
        # Calcular el número total de productos (posiciones de estantes)
        self.num_productos = len(self.posiciones_estantes)

        # Conjunto esperado de productos - calcular una vez y reutilizar
        self.productos_esperados = {f"{i:02}" for i in range(1, self.num_productos + 1)}
        
        # Para seguimiento de la mejor solución
        self.mejor_individuo = None
        self.mejor_aptitud = float('-inf')
        
        # Pre-calcular distancias euclidianas entre posiciones de estantes
        self.matriz_distancias = self._calcular_matriz_distancias()
        
        # Frecuencia de productos en órdenes - usada para optimizar mutaciones
        self.frecuencias_productos = self._calcular_frecuencias_productos()

    def _calcular_matriz_distancias(self):
        """Pre-calcular distancias entre todas las posiciones de estantes."""
        matriz_distancias = {}
        for i, pos1 in enumerate(self.posiciones_estantes):
            for j, pos2 in enumerate(self.posiciones_estantes):
                if i != j:
                    y1, x1 = pos1
                    y2, x2 = pos2
                    distancia = abs(y1 - y2) + abs(x1 - x2)  # Distancia Manhattan
                    matriz_distancias[(pos1, pos2)] = distancia
        return matriz_distancias
    
    def _calcular_frecuencias_productos(self):
        """Calcular frecuencia de cada producto en las órdenes."""
        frecuencias = {f"{i:02}":0 for i in range(1, self.num_productos + 1)}
        
        for orden in self.listas_productos:
            for producto in orden:
                id_producto = f"{int(producto):02}"
                if id_producto in frecuencias:
                    frecuencias[id_producto] += 1
        
        return frecuencias

    def _crear_individuo_aleatorio(self):
        """Crear una configuración aleatoria de colocación de productos."""
        # Generar una permutación aleatoria de números de productos (1 a num_productos)
        numeros_productos = list(range(1, self.num_productos + 1))
        random.shuffle(numeros_productos)
        
        # Crear un almacén con la nueva colocación de productos
        almacen = copy.deepcopy(self.plantilla_almacen)
        
        # Colocar productos según los números mezclados
        for i, (y, x) in enumerate(self.posiciones_estantes):
            almacen[y][x] = f"{numeros_productos[i]:02}"  # Asegurar formato de 2 dígitos
        
        return almacen
    
    def _crear_individuo_voraz(self):
        """Crear una solución inicial voraz basada en frecuencias de productos."""
        # Ordenar productos por frecuencia (más frecuentes primero)
        ids_productos = sorted(self.frecuencias_productos.keys(), 
                            key=lambda x: self.frecuencias_productos[x], reverse=True)
        
        # Crear un almacén vacío
        almacen = copy.deepcopy(self.plantilla_almacen)
        
        # Posición delantera, izquierda (aproximada)
        central_y = len(almacen) // 2
        central_x = len(almacen[0]) // 5
        
        # Ordenar posiciones de estantes por distancia
        posiciones_ordenadas = sorted(self.posiciones_estantes, 
                                 key=lambda pos: abs(pos[0] - central_y) + abs(pos[1] - central_x))
        
        # Colocar productos más frecuentes más a la izquierda
        for i, id_producto in enumerate(ids_productos):
            if i < len(posiciones_ordenadas):
                y, x = posiciones_ordenadas[i]
                almacen[y][x] = id_producto
        
        return almacen
    
    def _corregir_individuo(self, individuo):
        """Corrige un individuo asegurando que tenga exactamente los productos del 01 al 48 sin repetir."""
        # Obtener productos actuales
        productos_actuales = [individuo[y][x] for y, x in self.posiciones_estantes]
        
        # Identificar productos faltantes y repetidos rápidamente
        conjunto_actual = set()
        repetidos = []
        
        for i, prod in enumerate(productos_actuales):
            if prod in conjunto_actual:
                # Marcar posición con producto repetido
                y, x = self.posiciones_estantes[i]
                repetidos.append((y, x))
            else:
                conjunto_actual.add(prod)
        
        # Productos que faltan
        faltantes = list(self.productos_esperados - conjunto_actual)
        
        # Reemplazar repetidos con faltantes
        for idx, (y, x) in enumerate(repetidos):
            if idx < len(faltantes):
                individuo[y][x] = faltantes[idx]
        
        return individuo
    
    def _calcular_aptitud_trabajador(self, individuo):
        """Función trabajadora para calcular la aptitud de un individuo."""
        try:
            # Crear una copia del individuo para evitar modificarlo
            copia_almacen = copy.deepcopy(individuo)
            
            # Llamar directamente a la función de procesamiento para obtener el costo total
            costo_total = procesar(copia_almacen, self.listas_productos)
            
            # Devolver costo negativo como aptitud (menor costo = mayor aptitud)
            return -costo_total
        except Exception as e:
            print(f"Error calculando aptitud: {e}")
            return float('-inf')  # La menor aptitud posible
    
    def _calcular_aptitud_paralelo(self, poblacion):
        """Calcular aptitud para todos los individuos en paralelo."""
        with mp.Pool(processes=self.num_procesos) as pool:
            aptitudes = pool.map(self._calcular_aptitud_trabajador, poblacion)
        return aptitudes
    
    def _calcular_diversidad(self, poblacion):
        """Calcular diversidad de la población."""
        if len(poblacion) <= 1:
            return 0
            
        # Muestrear para evitar calcular todas las comparaciones cuando la población es grande
        tamaño_muestra = min(10, len(poblacion))
        poblacion_muestreada = random.sample(poblacion, tamaño_muestra)
        
        # Extraer colocaciones de productos una vez y almacenar en arrays numpy
        colocaciones_extraidas = np.array([
            [ind[y][x] for y, x in self.posiciones_estantes]
            for ind in poblacion_muestreada
        ])
        
        # Usar operaciones vectoriales para calcular diferencias
        diferencias_totales = 0
        comparaciones = 0
        
        for i in range(tamaño_muestra):
            for j in range(i+1, tamaño_muestra):
                # Comparación vectorizada
                diferencias = np.sum(colocaciones_extraidas[i] != colocaciones_extraidas[j])
                diferencias_totales += diferencias
                comparaciones += 1
        
        if comparaciones > 0:
            return diferencias_totales / (comparaciones * self.num_productos)
        return 0
    
    def _ajustar_parametros(self, diversidad):
        """Ajustar tasas de mutación y cruce basadas en la diversidad de la población."""
        if not self.tasas_adaptativas:
            return
            
        if diversidad < 0.3:
            self.tasa_mutacion = min(0.9, self.tasa_mutacion_inicial * 2) 
            self.tasa_cruce = max(0.5, self.tasa_cruce_inicial * 0.8)
        elif diversidad > 0.7:
            self.tasa_mutacion = max(0.1, self.tasa_mutacion_inicial * 0.5)
            self.tasa_cruce = min(0.95, self.tasa_cruce_inicial * 1.2)
        else:
            self.tasa_mutacion = self.tasa_mutacion_inicial
            self.tasa_cruce = self.tasa_cruce_inicial
    
    def _seleccion(self, poblacion, aptitudes):
        """Seleccionar individuos para reproducción usando selección por torneo."""
        seleccionados = []
        
        # Primero añadir los individuos elite
        indices_elite = np.argsort(aptitudes)[-self.tamaño_elite:]
        for idx in indices_elite:
            individuo = poblacion[idx]
            seleccionados.append(copy.deepcopy(individuo))
        
        # Luego usar selección por torneo para el resto
        while len(seleccionados) < self.tamaño_poblacion:
            torneo = random.sample(range(len(poblacion)), self.tamaño_torneo)
            aptitudes_torneo = [aptitudes[i] for i in torneo]
            idx_ganador = torneo[np.argmax(aptitudes_torneo)]
            
            individuo = poblacion[idx_ganador]
            seleccionados.append(copy.deepcopy(individuo))

        return seleccionados
    
    def _cruce(self, padre1, padre2):
        """Realizar Cruce de Orden (OX) entre dos padres con punto de corte fijo en la mitad."""
        if random.random() > self.tasa_cruce:
            return copy.deepcopy(padre1)

        # Crear un hijo con la plantilla
        hijo = copy.deepcopy(self.plantilla_almacen)

        # Extraer colocaciones de productos de los padres
        productos_padre1 = [padre1[y][x] for y, x in self.posiciones_estantes]
        productos_padre2 = [padre2[y][x] for y, x in self.posiciones_estantes]

        # Calcular el punto medio
        longitud = len(self.posiciones_estantes)
        punto_medio = longitud // 2

        # Inicializar lista de productos del hijo
        lista_productos_hijo = [None] * longitud

        # Copiar primera mitad del padre1
        lista_productos_hijo[:punto_medio] = productos_padre1[:punto_medio]

        # Rellenar la segunda mitad con elementos del padre2 que no estén en la primera mitad
        productos_primera_mitad = set(lista_productos_hijo[:punto_medio])
        restantes_padre2 = [item for item in productos_padre2 if item not in productos_primera_mitad]

        lista_productos_hijo[punto_medio:] = restantes_padre2[:longitud - punto_medio]

        # Colocar productos en el hijo
        for i, (y, x) in enumerate(self.posiciones_estantes):
            hijo[y][x] = lista_productos_hijo[i]

        # Corregir posibles valores repetidos o faltantes
        hijo = self._corregir_individuo(hijo)

        return hijo
    
    def _mutacion(self, individuo):
        """Aplicar mutación de intercambio a un individuo."""
        if random.random() > self.tasa_mutacion:
            return individuo
        
        # Crear una copia mutable
        mutado = copy.deepcopy(individuo)
        
        # Aplicar múltiples intercambios con probabilidad decreciente
        for _ in range(5):  # Aplicar hasta 5 intercambios
            if random.random() < self.tasa_mutacion * (0.5 ** _):
                # Elegir dos posiciones aleatorias de estantes e intercambiar sus productos
                pos1, pos2 = random.sample(self.posiciones_estantes, 2)
                y1, x1 = pos1
                y2, x2 = pos2
                mutado[y1][x1], mutado[y2][x2] = mutado[y2][x2], mutado[y1][x1]

        # Corregir posibles valores repetidos o faltantes   
        mutado = self._corregir_individuo(mutado)
        
        return mutado
    
    def _busqueda_local(self, individuo):
        """Aplicar búsqueda local para mejorar un individuo."""
        mejorado = copy.deepcopy(individuo)
        
        # Obtener aptitud inicial
        aptitud = self._calcular_aptitud_trabajador(mejorado)
        
        # Probar algunos intercambios aleatorios
        for _ in range(3):  # Limitar a 3 iteraciones para evitar cálculos excesivos
            # Elegir dos posiciones aleatorias
            pos1, pos2 = random.sample(self.posiciones_estantes, 2)
            y1, x1 = pos1
            y2, x2 = pos2
            
            # Intercambiar productos
            mejorado[y1][x1], mejorado[y2][x2] = mejorado[y2][x2], mejorado[y1][x1]
            
            # Calcular nueva aptitud
            nueva_aptitud = self._calcular_aptitud_trabajador(mejorado)
            
            # Si no es mejor, revertir el intercambio
            if nueva_aptitud <= aptitud:
                mejorado[y1][x1], mejorado[y2][x2] = mejorado[y2][x2], mejorado[y1][x1]
            else:
                aptitud = nueva_aptitud
        
        return mejorado
    
    def _diversificar_poblacion(self, poblacion):
        """Diversificar la población cuando se estanca en óptimos locales."""
        nueva_poblacion = []
        
        # Mantener el mejor individuo
        mejor_individuo = poblacion[0]
        nueva_poblacion.append(mejor_individuo)
        
        # Añadir algunos individuos aleatorios
        for _ in range(self.tamaño_poblacion // 4):
            nueva_poblacion.append(self._crear_individuo_aleatorio())
        
        # Añadir algunas versiones mutadas del mejor individuo
        while len(nueva_poblacion) < self.tamaño_poblacion:
            individuo = copy.deepcopy(mejor_individuo)
            # Aplicar mutación fuerte
            for _ in range(5):
                individuo = self._mutacion(individuo)
            nueva_poblacion.append(individuo)
        
        return nueva_poblacion
    
    def ejecutar(self):
        """Ejecutar el algoritmo genético con mejoras."""
        # Crear población inicial con métodos diversos
        poblacion = []
        
        # Añadir individuo voraz
        if self.incluir_voraz:
            poblacion.append(self._crear_individuo_voraz())

        # Añadir individuos aleatorios
        for _ in range(self.tamaño_poblacion // 2):
            poblacion.append(self._crear_individuo_aleatorio())
        
        # Añadir variaciones del diseño original
        almacen_original = crear_almacen_original()
        poblacion.append(almacen_original)
        
        # Añadir versiones mutadas del diseño original
        while len(poblacion) < self.tamaño_poblacion:
            mutado = copy.deepcopy(almacen_original)
            for _ in range(3):
                mutado = self._mutacion(mutado)
            poblacion.append(mutado)
        
        self.mejor_aptitud = float('-inf')
        self.mejor_individuo = None
        
        # Seguimiento de los mejores individuos para cada generación
        mejores_individuos_generacion = []
        mejores_aptitudes_generacion = []
        
        # Empezar con el diseño original
        costo_original = procesar(almacen_original, self.listas_productos)
        mejores_individuos_generacion.append(almacen_original)
        mejores_aptitudes_generacion.append(-costo_original)  # Almacenar como aptitud (costo negativo)
        
        tiempo_inicio = time.time()
        
        for generacion in range(self.generaciones):
            tiempo_inicio_gen = time.time()
            print(f"Generación {generacion+1}/{self.generaciones}")
            
            # Calcular aptitud en paralelo
            aptitudes = self._calcular_aptitud_paralelo(poblacion)
            
            # Calcular diversidad de la población
            diversidad = self._calcular_diversidad(poblacion)
            
            # Ajustar parámetros basados en la diversidad
            self._ajustar_parametros(diversidad)
            
            # Ordenar población por aptitud
            indices_ordenados = np.argsort(aptitudes)[::-1]
            poblacion = [poblacion[i] for i in indices_ordenados]
            aptitudes = [aptitudes[i] for i in indices_ordenados]
            
            # Encontrar mejor individuo
            mejor_generacion = copy.deepcopy(poblacion[0])
            mejor_aptitud_generacion = aptitudes[0]

            # Asegurar que el mejor individuo es válido
            mejor_generacion = self._corregir_individuo(mejor_generacion)
            mejor_aptitud_generacion = self._calcular_aptitud_trabajador(mejor_generacion)

            mejores_individuos_generacion.append(mejor_generacion)
            mejores_aptitudes_generacion.append(mejor_aptitud_generacion)
            
            if mejor_aptitud_generacion > self.mejor_aptitud:
                self.mejor_aptitud = mejor_aptitud_generacion
                self.mejor_individuo = copy.deepcopy(mejor_generacion)
                self.contador_estancamiento = 0
                print(f"Nueva mejor aptitud: {-self.mejor_aptitud}")
            else:
                self.contador_estancamiento += 1
                print(f"Sin mejora. La mejor aptitud sigue siendo: {-self.mejor_aptitud}")
                
            # Comprobar estancamiento
            if self.contador_estancamiento >= self.umbral_estancamiento:
                print("Estancamiento detectado. Diversificando población...")
                poblacion = self._diversificar_poblacion(poblacion)
                self.contador_estancamiento = 0
            else:
                # Selección
                seleccionados = self._seleccion(poblacion, aptitudes)
                
                # Crear nueva población
                nueva_poblacion = seleccionados[:self.tamaño_elite]
                
                # Aplicar búsqueda local a individuos elite
                for i in range(min(self.tamaño_elite, len(nueva_poblacion))):
                    nueva_poblacion[i] = self._busqueda_local(nueva_poblacion[i])
                
                # Generar nuevos individuos
                while len(nueva_poblacion) < self.tamaño_poblacion:
                    padre1, padre2 = random.sample(seleccionados, 2)
                    hijo = self._cruce(padre1, padre2)
                    hijo = self._mutacion(hijo)
                    
                    hijo = self._corregir_individuo(hijo)
                    
                    nueva_poblacion.append(hijo)
                
                poblacion = nueva_poblacion
            
            # Imprimir estadísticas de diversidad y parámetros
            print(f"Diversidad de la población: {diversidad:.4f}")
            print(f"Tasa de mutación: {self.tasa_mutacion:.2f}, Tasa de cruce: {self.tasa_cruce:.2f}")
            
            tiempo_gen = time.time() - tiempo_inicio_gen
            print(f"Generación completada en {tiempo_gen:.2f} segundos")
            print("-" * 50)
        
        # Guardar resultados en CSV
        self._guardar_resultados_en_csv(mejores_individuos_generacion, mejores_aptitudes_generacion)
        
        tiempo_total = time.time() - tiempo_inicio
        print("\nAlgoritmo Genético Completado")
        print(f"Costo del mejor diseño de almacén: {-self.mejor_aptitud}")
        print(f"Tiempo total de ejecución: {tiempo_total:.2f} segundos")
        
        return self.mejor_individuo, -self.mejor_aptitud

    def _guardar_resultados_en_csv(self, diseños, aptitudes):
        """Guardar los mejores diseños y sus valores de aptitud en un archivo CSV."""
        try:
            with open('resultados.csv', 'w', newline='') as archivo:
                escritor = csv.writer(archivo)
                
                # Escribir encabezado
                escritor.writerow(['Generacion', 'Costo', 'Diseño'])
                
                # Escribir diseño original (generación 0)
                escritor.writerow([0, -aptitudes[0], self._diseño_a_cadena(diseños[0])])
                
                # Escribir mejores diseños para cada generación
                for i in range(1, len(diseños)):
                    escritor.writerow([i, -aptitudes[i], self._diseño_a_cadena(diseños[i])])
                    
            print(f"Resultados guardados exitosamente en resultados.csv")
        except Exception as e:
            print(f"Error al guardar resultados en CSV: {e}")

    def _diseño_a_cadena(self, diseño):
        """Convertir un diseño de almacén a una representación de cadena."""
        return '|'.join([''.join(celda.rjust(2, '0') if celda.isdigit() else celda for celda in fila) for fila in diseño])

def inicializar_plantilla_almacen_vacia():
    """Inicializar una plantilla de almacén vacía con '.' para caminos y '#' para posiciones de estantes."""
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
    return almacen

def crear_almacen_original():
    """Crear el diseño de almacén original con productos numerados del 1 al 48."""
    almacen = inicializar_plantilla_almacen_vacia()
    
    # Numerar los estantes
    contador = 1
    ancho_bloque, alto_bloque = 2, 5
    filas, cols = len(almacen), len(almacen[0])
    estantes = {(y, x) for y in range(filas) for x in range(cols) if almacen[y][x] == '#'}
   
    visitados = set()
    for y in range(0, filas, alto_bloque):
        for x in range(0, cols, ancho_bloque):
            bloque = [(ny, nx) for dy in range(alto_bloque) for dx in range(ancho_bloque)
                    if (ny := y + dy) < filas and (nx := x + dx) < cols
                    and (ny, nx) in estantes and (ny, nx) not in visitados]
            if bloque:
                for ny, nx in bloque:
                    almacen[ny][nx] = str(contador)
                    contador += 1
                    visitados.add((ny, nx))
    
    return almacen

if __name__ == "__main__":
    # Obtener número de núcleos CPU disponibles
    num_nucleos = mp.cpu_count()
    print(f"Número de núcleos CPU disponibles: {num_nucleos}")

    # Leer listas de productos del archivo CSV
    listas_productos = []
    try:
        with open('ordenes.csv', 'r') as archivo:
            lector_csv = csv.reader(archivo)
            for fila in lector_csv:
                # Convertir números de cadena a cadenas reales
                lista_productos = [str(int(producto)) for producto in fila]
                listas_productos.append(lista_productos)
        print(f"Cargadas exitosamente {len(listas_productos)} listas de productos desde ordenes.csv")
    except Exception as e:
        print(f"Error al leer archivo CSV: {e}")
        listas_productos = []
    
    # Crear la plantilla de almacén
    plantilla_almacen = inicializar_plantilla_almacen_vacia()

    # Crear diseño de almacén original
    almacen_original = crear_almacen_original()
    # Calcular costo del diseño original
    costo_original = procesar(almacen_original, listas_productos)
    print(f"Aptitud del diseño original: {costo_original}")

    # Crear y ejecutar el algoritmo genético con parámetros mejorados
    ag = OptimizadorLayoutAlmacenAG(
        plantilla_almacen,  # La matriz que representa el diseño físico del almacén con '#' para estantes
        tamaño_poblacion=7,  # Número de soluciones simultáneas - balance entre diversidad y eficiencia computacional
        generaciones=5,  # Cuántas iteraciones evolutivas ejecutará el algoritmo para mejorar soluciones
        tasa_mutacion=0.25,  # Probabilidad de que ocurra mutación - controla exploración de nuevas áreas
        tasa_cruce=0.9,  # Probabilidad de que ocurra cruce entre soluciones - favorece combinación de buenas características
        tamaño_elite=1,  # Número de mejores soluciones que pasan directamente a la siguiente generación sin modificaciones
        num_procesos=num_nucleos - 1,  # Distribución del trabajo en paralelo, dejando un núcleo libre para el SO y otros procesos
        listas_productos=listas_productos,  # Datos de órdenes reales para evaluar la calidad de las soluciones
        tasas_adaptativas=True,  # Permite que las tasas de mutación/cruce se ajusten automáticamente según la diversidad
        tamaño_torneo=2,  # En selección por torneo, cuántas soluciones compiten - valores más altos aumentan la presión selectiva
        incluir_voraz=True # Activar o desactivar un individuo voraz - mejora la función de fitness en la primera generación
    )
    
    # Ejecutar AG
    mejor_almacen, mejor_costo = ag.ejecutar()
    
    # Mostrar porcentaje de mejora
    mejora = ((costo_original - mejor_costo) / costo_original) * 100

    print("\nResultados:")
    print(f"Costo del Diseño Original: {costo_original}")
    print(f"Costo del Diseño Optimizado: {mejor_costo}")
    print(f"Mejora: {mejora:.2f}%\n")
    