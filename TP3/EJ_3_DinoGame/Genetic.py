import random
import numpy as np
import tensorflow as tf
from Dinosaur import Dinosaur
import json
import os

class GeneticAlgorithm:
    """
    Clase que implementa un algoritmo genético para evolucionar dinosaurios.
    """

    def __init__(self):
        # Contador de generaciones
        self.generation = 0
        
        # Historial de las mejores puntuaciones por generación
        self.best_scores_history = []
        
        # Pool de individuos élite de todas las generaciones
        self.elite_pool = []
        
        # Umbrales para clasificar individuos según su rendimiento, entendido como puntaje máximo alcanzado.
        self.elite_threshold = 500      
        self.super_elite_threshold = 800
        
        # Hall of fame: los mejores dinosaurios de todos los tiempos
        self.hall_of_fame = []
        
        # Parámetros para mutación adaptativa
        self.base_mutation_rate = 0.15        # Tasa base de mutación
        self.current_mutation_rate = self.base_mutation_rate
        self.stagnation_counter = 0           # Contador de generaciones sin mejora
        self.best_score_ever = 0              # Mejor puntuación histórica
        
        # Cargar dinosaurios guardados del hall of fame
        self.load_hall_of_fame()

    def save_hall_of_fame(self):
        """
        Guarda los mejores dinosaurios en un archivo JSON para persistencia.
        Serializa los pesos y sesgos de las redes neuronales.
        """
        try:
            hall_data = []
            for dino in self.hall_of_fame:
                # Convertir tensores de TensorFlow a listas de Python
                weights_data = [w.numpy().tolist() for w in dino.weights]
                biases_data = [b.numpy().tolist() for b in dino.biases]
                
                hall_data.append({
                    'score': dino.score,
                    'weights': weights_data,
                    'biases': biases_data
                })
            
            with open('hall_of_fame.json', 'w') as f:
                json.dump(hall_data, f)
        except Exception as e:
            print(f"Error guardando hall of fame: {e}")

    def load_hall_of_fame(self):
        """
        Carga los mejores dinosaurios desde un archivo JSON.
        Reconstruye las redes neuronales con los pesos guardados.
        """
        try:
            if os.path.exists('hall_of_fame.json'):
                with open('hall_of_fame.json', 'r') as f:
                    hall_data = json.load(f)
                
                for data in hall_data:
                    # Crear nuevo dinosaurio con red neuronal
                    dino = Dinosaur(0, None, True)  
                    
                    # Restaurar pesos y sesgos desde los datos guardados
                    for i, (w_data, b_data) in enumerate(zip(data['weights'], data['biases'])):
                        dino.weights[i].assign(tf.convert_to_tensor(w_data, dtype=tf.float32))
                        dino.biases[i].assign(tf.convert_to_tensor(b_data, dtype=tf.float32))
                    
                    dino.score = data['score']
                    self.hall_of_fame.append(dino)
                    self.best_score_ever = max(self.best_score_ever, dino.score)
                    
                print(f"\nCargados {len(self.hall_of_fame)} individuos del hall of fame")
        except Exception as e:
            print(f"Error cargando hall of fame: {e}")

    def update_hall_of_fame(self, population):
        """
        Actualiza el hall of fame con los mejores dinosaurios de la generación actual.
        """
        if not population:  # Protección contra población vacía
            return
            
        current_best = max(population, key=lambda x: x.score)
        
        # Agregar individuos al pool élite
        for dino in population:
            if dino.score >= self.elite_threshold:  # Más de 500 puntos
                elite_copy = self.create_elite_copy(dino)
                self.elite_pool.append(elite_copy)
        
        # Actualizar récord absoluto si hay mejora
        if current_best.score > self.best_score_ever:
            self.best_score_ever = current_best.score
            elite_copy = self.create_elite_copy(current_best)
            self.hall_of_fame.append(elite_copy)
            
            print(f"\n¡NUEVO RÉCORD! Puntuación: {current_best.score}")
            print(f"Individuo guardado en Hall of Fame")
            
        # Limpiar y mantener solo los mejores (evitar duplicados)
        all_elites = self.hall_of_fame + self.elite_pool
        if all_elites:
            unique_elites = {}
            for elite in all_elites:
                key = elite.score
                if key not in unique_elites or elite.score > unique_elites[key].score:
                    unique_elites[key] = elite
            
            # Ordenar por puntuación y mantener los mejores
            sorted_elites = sorted(unique_elites.values(), key=lambda x: x.score, reverse=True)
            self.hall_of_fame = sorted_elites[:10]  # Top 10 absolutos
            self.elite_pool = sorted_elites[10:50] if len(sorted_elites) > 10 else []
            
            self.save_hall_of_fame()

    def create_elite_copy(self, original):
        """
        Crea una copia de un dinosaurio élite para preservarlo.
        """
        elite = Dinosaur(original.id, None, True) 
        
        # Copiar pesos y sesgos de la red neuronal
        for i in range(len(original.weights)):
            elite.weights[i].assign(tf.identity(original.weights[i]))
            elite.biases[i].assign(tf.identity(original.biases[i]))
        
        elite.score = original.score
        return elite

    def adaptive_mutation_rate(self, population):
        """
        Ajusta dinámicamente la tasa de mutación según el progreso de la evolución.
        Si no hay mejora, aumenta la mutación para explorar más.
        """
        current_best = max(population, key=lambda x: x.score).score
        self.best_scores_history.append(current_best)
        
        # Si es la primera generación, establecer como mejor puntuación
        if len(self.best_scores_history) == 1:
            self.current_mutation_rate = self.base_mutation_rate
            return
        
        # Comparar con la generación anterior
        previous_best = self.best_scores_history[-2]
        
        # Si no hay mejora en la última generacion, aumentar mutación
        if current_best <= previous_best:
            self.stagnation_counter += 1
            # Aumentar mutación gradualmente (máximo 50%)
            self.current_mutation_rate = min(0.5, self.base_mutation_rate * (1 + self.stagnation_counter * 0.15))

        else:
            # Hay mejora, resetear contador y volver a tasa base
            self.stagnation_counter = 0
            self.current_mutation_rate = self.base_mutation_rate

    def classify_individuals(self, population):
        """
        Clasifica los dinosaurios según su rendimiento en diferentes categorías.
        """
        prometedores = []
        elites = []
        super_elites = []
        regular = []
        
        for dino in population:
            if dino.score >= self.super_elite_threshold:        # 800+ puntos
                super_elites.append(dino)
            elif dino.score >= self.elite_threshold:           # 500-799 puntos
                elites.append(dino)
            elif dino.score >= 200:                             # 200-499 puntos
                prometedores.append(dino)
            else:                                               # < 200 puntos
                regular.append(dino)
        
        return super_elites, elites, prometedores, regular

    def tournament_selection(self, candidates, tournament_size=3):
        """
        Selección por torneo: elige aleatoriamente un grupo y retorna el mejor.
        """
        if len(candidates) < tournament_size:
            return random.choice(candidates)
        
        tournament = random.sample(candidates, tournament_size)
        return max(tournament, key=lambda x: x.score)

    def intelligent_mutation(self, individual, mutation_strength="adaptive"):
        """
        Mutación inteligente que preserva características de individuos exitosos.
        """
        if mutation_strength == "adaptive":
            # Mutación más suave para individuos exitosos
            if individual.score > self.elite_threshold:
                rate = self.current_mutation_rate * 0.2  # Menos mutación para élites
            else:
                rate = self.current_mutation_rate
        else:
            rate = self.current_mutation_rate
            
        # Mutar pesos y sesgos de cada capa
        for i in range(len(individual.weights)):
            # Mutación de pesos
            if np.random.rand() < rate:
                # Ruido gaussiano con intensidad variable según el rendimiento
                noise_scale = 0.1 if individual.score > self.elite_threshold else 0.2
                noise = np.random.normal(0, noise_scale, individual.weights[i].shape)
                new_weights = individual.weights[i].numpy() + noise
                individual.weights[i].assign(tf.convert_to_tensor(new_weights, dtype=tf.float32))
            
            # Mutación de sesgos
            if np.random.rand() < rate:
                noise_scale = 0.1 if individual.score > self.elite_threshold else 0.2
                noise = np.random.normal(0, noise_scale, individual.biases[i].shape)
                new_biases = individual.biases[i].numpy() + noise
                individual.biases[i].assign(tf.convert_to_tensor(new_biases, dtype=tf.float32))

    def evolve(self, parent1, parent2):
        """
        Implementa diferentes estrategias de cruzamiento para crear descendencia.
        """
        child = Dinosaur(parent1.id, parent1.color, True)
        
        # Procesar cada capa de la red neuronal
        for i in range(len(parent1.weights)):
            # Cruzamiento uniforme: cada peso se toma aleatoriamente de un padre
            w1 = parent1.weights[i].numpy()
            w2 = parent2.weights[i].numpy()
            mask = np.random.rand(*w1.shape) < 0.5  # Máscara aleatoria
            child_w = np.where(mask, w1, w2)
            
            child.weights[i].assign(tf.convert_to_tensor(child_w, dtype=tf.float32))
            
            # Mismo proceso para sesgos (bias)
            b1, b2 = parent1.biases[i].numpy(), parent2.biases[i].numpy()
            mask_b = np.random.rand(*b1.shape) < 0.5
            child_b = np.where(mask_b, b1, b2)
            
            child.biases[i].assign(tf.convert_to_tensor(child_b, dtype=tf.float32))
        
        return child

def select_fittest(population, top_k=10):
    """
    Selecciona los k mejores individuos de la población.
    """
    sorted_population = sorted(population, key=lambda dino: dino.score, reverse=True)
    return sorted_population[:top_k]

def updateNetwork(population):
    """
    Esta función es llamada desde main.py cuando toda la población muere.
    """
    global genetic_algorithm
    
    # Inicializar algoritmo genético si no existe
    if 'genetic_algorithm' not in globals():
        genetic_algorithm = GeneticAlgorithm()
    
    genetic_algorithm.generation += 1
    
    # Guardar élites antes de cualquier modificación
    genetic_algorithm.update_hall_of_fame(population)
    genetic_algorithm.adaptive_mutation_rate(population)
    
    # Mostrar información
    current_best = max(population, key=lambda x: x.score).score if population else 0
    print(f"\n=== PRESERVACIÓN DE ÉLITES ===")
    print(f"Mejor de esta generación: {current_best}")
    print(f"Mejor histórico: {genetic_algorithm.best_score_ever}")
    print("===============================")
    
    # Clasificar individuos según su rendimiento
    super_elites, elites, prometedores, regular = genetic_algorithm.classify_individuals(population)
    
    print(f"Generación {genetic_algorithm.generation}:")
    print(f"  Súper élites: {len(super_elites)}")
    print(f"  Élites: {len(elites)}")
    print(f"  Prometedores: {len(prometedores)}")
    print(f"  Regulares: {len(regular)}")
    print(f"  Tasa de mutación actual: {genetic_algorithm.current_mutation_rate:.3f}")
    
    new_population = []
    population_size = len(population)
    
    # 1. PRESERVAR SÚPER ÉLITES (15% de la población)
    # Los mejores individuos se mantienen sin cambios
    elite_count = min(len(super_elites + elites), int(population_size * 0.15))
    if elite_count > 0:
        best_individuals = sorted(super_elites + elites, 
                                key=lambda x: x.score, reverse=True)[:elite_count]
        for elite in best_individuals:
            elite_copy = genetic_algorithm.create_elite_copy(elite)
            new_population.append(elite_copy)
    
    # 2. INCLUIR DEL HALL OF FAME (10% de la población)
    # Siempre disponibles, incluso si la generación actual es mala
    hall_count = min(len(genetic_algorithm.hall_of_fame), int(population_size * 0.1))
    if hall_count > 0:
        hall_sample = random.sample(genetic_algorithm.hall_of_fame, hall_count)
        for hall_member in hall_sample:
            hall_copy = genetic_algorithm.create_elite_copy(hall_member)
            new_population.append(hall_copy)
        print(f"Incluidos {hall_count} individuos del Hall of Fame")
    
    # 3. INCLUIR DEL POOL ÉLITE ADICIONAL (5% de la población)
    pool_count = min(len(genetic_algorithm.elite_pool), int(population_size * 0.05))
    if pool_count > 0:
        pool_sample = random.sample(genetic_algorithm.elite_pool, pool_count)
        for pool_member in pool_sample:
            pool_copy = genetic_algorithm.create_elite_copy(pool_member)
            new_population.append(pool_copy)
        print(f"Incluidos {pool_count} individuos del pool élite")
    
    # 4. GENERAR EL RESTO MEDIANTE CRUZAMIENTO ESTRATÉGICO (70% restante)
    # Crear pool de padres potenciales priorizando mejores individuos
    breeding_pool = super_elites + elites + prometedores
    if not breeding_pool:  # Si no hay buenos individuos, usar todos
        breeding_pool = population
    
    while len(new_population) < population_size:
        # Selección de padres con estrategia mixta
        if len(breeding_pool) >= 2:
            if np.random.rand() < 0.6:  # 60% selección por torneo (favorece mejores)
                parent1 = genetic_algorithm.tournament_selection(breeding_pool)
                parent2 = genetic_algorithm.tournament_selection(breeding_pool)
            else:  # 40% selección aleatoria (mantiene diversidad)
                parent1, parent2 = random.sample(breeding_pool, 2)
        else:
            parent1 = parent2 = breeding_pool[0]
        
        # Cruzamiento
        child = genetic_algorithm.evolve(parent1, parent2)
        
        # Mutación inteligente
        genetic_algorithm.intelligent_mutation(child)
        
        new_population.append(child)
    
    # Actualizar la población original con la nueva generación
    for i in range(len(population)):
        if i < len(new_population):
            population[i].weights = new_population[i].weights
            population[i].biases = new_population[i].biases
            population[i].color = new_population[i].color
