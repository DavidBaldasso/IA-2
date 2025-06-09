import random
import numpy as np
import tensorflow as tf
from Dinosaur import Dinosaur
import json
import os

class GeneticAlgorithm:
    def __init__(self):
        self.generation = 0
        self.best_scores_history = []
        self.elite_pool = []  # Pool de los mejores individuos de todas las generaciones
        self.greedy_threshold = 500  # Umbral para individuos voraces
        self.super_elite_threshold = 800  # Umbral para súper élites
        self.hall_of_fame = []  # Los mejores de todos los tiempos
        
        # Parámetros adaptativos
        self.base_mutation_rate = 0.25
        self.current_mutation_rate = self.base_mutation_rate
        self.stagnation_counter = 0
        self.best_score_ever = 0
        
        # Cargar hall of fame si existe
        self.load_hall_of_fame()

    def save_hall_of_fame(self):
        """Guarda los mejores individuos en un archivo"""
        try:
            hall_data = []
            for dino in self.hall_of_fame:
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
        """Carga los mejores individuos desde archivo"""
        try:
            if os.path.exists('hall_of_fame.json'):
                with open('hall_of_fame.json', 'r') as f:
                    hall_data = json.load(f)
                
                for data in hall_data:
                    dino = Dinosaur(0, None, True)  
                    
                    # Restaurar pesos y sesgos
                    for i, (w_data, b_data) in enumerate(zip(data['weights'], data['biases'])):
                        dino.weights[i].assign(tf.convert_to_tensor(w_data, dtype=tf.float32))
                        dino.biases[i].assign(tf.convert_to_tensor(b_data, dtype=tf.float32))
                    
                    dino.score = data['score']
                    self.hall_of_fame.append(dino)
                    self.best_score_ever = max(self.best_score_ever, dino.score)
                    
                print(f"Cargados {len(self.hall_of_fame)} individuos del hall of fame")
        except Exception as e:
            print(f"Error cargando hall of fame: {e}")

    def update_hall_of_fame(self, population):
        """Actualiza el hall of fame con los mejores individuos"""
        if not population:  # Protección contra población vacía
            return
            
        current_best = max(population, key=lambda x: x.score)
        
        # Siempre agregar individuos prometedores al pool élite
        for dino in population:
            if dino.score >= self.greedy_threshold:  # 500+ puntos
                elite_copy = self.create_elite_copy(dino)
                self.elite_pool.append(elite_copy)
        
        # Actualizar récord absoluto
        if current_best.score > self.best_score_ever:
            self.best_score_ever = current_best.score
            elite_copy = self.create_elite_copy(current_best)
            self.hall_of_fame.append(elite_copy)
            
            print(f"\n¡NUEVO RÉCORD! Puntuación: {current_best.score}")
            print(f"Individuo guardado en Hall of Fame")
            
        # Limpiar y mantener los mejores
        all_elites = self.hall_of_fame + self.elite_pool
        if all_elites:
            # Ordenar por puntuación y mantener únicos
            unique_elites = {}
            for elite in all_elites:
                key = elite.score
                if key not in unique_elites or elite.score > unique_elites[key].score:
                    unique_elites[key] = elite
            
            sorted_elites = sorted(unique_elites.values(), key=lambda x: x.score, reverse=True)
            self.hall_of_fame = sorted_elites[:10]  # Top 10 absolutos
            self.elite_pool = sorted_elites[10:50] if len(sorted_elites) > 10 else []  # Pool adicional
            
            self.save_hall_of_fame()

    def create_elite_copy(self, original):
        """Crea una copia profunda de un dinosaurio élite"""
        elite = Dinosaur(original.id, None, True) 
        
        for i in range(len(original.weights)):
            elite.weights[i].assign(tf.identity(original.weights[i]))
            elite.biases[i].assign(tf.identity(original.biases[i]))
        
        elite.score = original.score
        return elite

    def adaptive_mutation_rate(self, population):
        """Ajusta la tasa de mutación basada en el progreso"""
        current_best = max(population, key=lambda x: x.score).score
        self.best_scores_history.append(current_best)
        
        # Si no hay mejora en las últimas 5 generaciones, aumentar mutación
        if len(self.best_scores_history) >= 5:
            recent_scores = self.best_scores_history[-5:]
            if max(recent_scores) == min(recent_scores):  # Sin mejora
                self.stagnation_counter += 1
                self.current_mutation_rate = min(0.4, self.base_mutation_rate * (1 + self.stagnation_counter * 0.1))
            else:
                self.stagnation_counter = 0
                self.current_mutation_rate = self.base_mutation_rate

    def classify_individuals(self, population):
        """Clasifica individuos según su rendimiento"""
        elites = []
        greedy_individuals = []
        super_elites = []
        regular = []
        
        for dino in population:
            if dino.score >= self.super_elite_threshold:
                super_elites.append(dino)
            elif dino.score >= self.greedy_threshold:
                greedy_individuals.append(dino)
            elif dino.score >= 200:  # Individuos prometedores
                elites.append(dino)
            else:
                regular.append(dino)
        
        return super_elites, greedy_individuals, elites, regular

    def tournament_selection(self, candidates, tournament_size=5):
        """Selección por torneo para mayor diversidad"""
        if len(candidates) < tournament_size:
            return random.choice(candidates)
        
        tournament = random.sample(candidates, tournament_size)
        return max(tournament, key=lambda x: x.score)

    def advanced_crossover(self, parent1, parent2, crossover_type="uniform"):
        """Cruzamiento avanzado con diferentes estrategias"""
        child = Dinosaur(parent1.id, parent1.color, True)
        
        for i in range(len(parent1.weights)):
            if crossover_type == "uniform":
                # Cruzamiento uniforme
                w1 = parent1.weights[i].numpy()
                w2 = parent2.weights[i].numpy()
                mask = np.random.rand(*w1.shape) < 0.5
                child_w = np.where(mask, w1, w2)
                
            elif crossover_type == "arithmetic":
                # Cruzamiento aritmético (promedio ponderado)
                alpha = np.random.rand()
                child_w = alpha * parent1.weights[i].numpy() + (1 - alpha) * parent2.weights[i].numpy()
                
            elif crossover_type == "simulated_binary":
                # Cruzamiento binario simulado
                eta = 20  # Parámetro de distribución
                u = np.random.rand(*parent1.weights[i].shape)
                beta = np.where(u <= 0.5, 
                               (2 * u) ** (1 / (eta + 1)),
                               (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
                
                child_w = 0.5 * ((parent1.weights[i].numpy() + parent2.weights[i].numpy()) - 
                                beta * np.abs(parent2.weights[i].numpy() - parent1.weights[i].numpy()))
            
            child.weights[i].assign(tf.convert_to_tensor(child_w, dtype=tf.float32))
            
            # Mismo proceso para sesgos
            if crossover_type == "uniform":
                b1, b2 = parent1.biases[i].numpy(), parent2.biases[i].numpy()
                mask_b = np.random.rand(*b1.shape) < 0.5
                child_b = np.where(mask_b, b1, b2)
            else:
                child_b = 0.5 * (parent1.biases[i].numpy() + parent2.biases[i].numpy())
            
            child.biases[i].assign(tf.convert_to_tensor(child_b, dtype=tf.float32))
        
        return child

    def intelligent_mutation(self, individual, mutation_strength="adaptive"):
        """Mutación inteligente que preserva características importantes"""
        if mutation_strength == "adaptive":
            # Mutación más suave para individuos exitosos
            if individual.score > self.greedy_threshold:
                rate = self.current_mutation_rate * 0.5
            else:
                rate = self.current_mutation_rate
        else:
            rate = self.current_mutation_rate
            
        for i in range(len(individual.weights)):
            # Mutación gaussiana con diferentes intensidades
            if np.random.rand() < rate:
                noise_scale = 0.1 if individual.score > self.greedy_threshold else 0.2
                noise = np.random.normal(0, noise_scale, individual.weights[i].shape)
                new_weights = individual.weights[i].numpy() + noise
                individual.weights[i].assign(tf.convert_to_tensor(new_weights, dtype=tf.float32))
            
            if np.random.rand() < rate:
                noise_scale = 0.1 if individual.score > self.greedy_threshold else 0.2
                noise = np.random.normal(0, noise_scale, individual.biases[i].shape)
                new_biases = individual.biases[i].numpy() + noise
                individual.biases[i].assign(tf.convert_to_tensor(new_biases, dtype=tf.float32))

def updateNetwork(population):
    """Función principal de actualización con algoritmo genético avanzado"""
    global genetic_algorithm
    
    # Inicializar algoritmo genético si no existe
    if 'genetic_algorithm' not in globals():
        genetic_algorithm = GeneticAlgorithm()
    
    genetic_algorithm.generation += 1
    
    # CRÍTICO: Guardar élites ANTES de cualquier modificación
    genetic_algorithm.update_hall_of_fame(population)
    genetic_algorithm.adaptive_mutation_rate(population)
    
    # Mostrar información de preservación
    current_best = max(population, key=lambda x: x.score).score if population else 0
    print(f"\n=== PRESERVACIÓN DE ÉLITES ===")
    print(f"Mejor de esta generación: {current_best}")
    print(f"Mejor histórico: {genetic_algorithm.best_score_ever}")
    print("===============================")
    
    # Clasificar individuos
    super_elites, greedy_individuals, elites, regular = genetic_algorithm.classify_individuals(population)
    
    print(f"Generación {genetic_algorithm.generation}:")
    print(f"  Súper élites: {len(super_elites)}")
    print(f"  Individuos voraces: {len(greedy_individuals)}")
    print(f"  Élites: {len(elites)}")
    print(f"  Regulares: {len(regular)}")
    print(f"  Tasa de mutación actual: {genetic_algorithm.current_mutation_rate:.3f}")
    
    new_population = []
    population_size = len(population)
    
    # 1. Preservar súper élites (15% de la población)
    elite_count = min(len(super_elites + greedy_individuals), int(population_size * 0.15))
    if elite_count > 0:
        best_individuals = sorted(super_elites + greedy_individuals, 
                                key=lambda x: x.score, reverse=True)[:elite_count]
        for elite in best_individuals:
            elite_copy = genetic_algorithm.create_elite_copy(elite)
            new_population.append(elite_copy)
    
    # 2. Incluir algunos del hall of fame (10% de la población) - SIEMPRE DISPONIBLES
    hall_count = min(len(genetic_algorithm.hall_of_fame), int(population_size * 0.1))
    if hall_count > 0:
        hall_sample = random.sample(genetic_algorithm.hall_of_fame, hall_count)
        for hall_member in hall_sample:
            hall_copy = genetic_algorithm.create_elite_copy(hall_member)
            # Dar color especial a los del Hall of Fame
            hall_copy.color = (255, 215, 0)  # Dorado
            new_population.append(hall_copy)
        print(f"Incluidos {hall_count} individuos del Hall of Fame")
    
    # 3. Incluir del pool élite adicional (5% de la población)
    pool_count = min(len(genetic_algorithm.elite_pool), int(population_size * 0.05))
    if pool_count > 0:
        pool_sample = random.sample(genetic_algorithm.elite_pool, pool_count)
        for pool_member in pool_sample:
            pool_copy = genetic_algorithm.create_elite_copy(pool_member)
            pool_copy.color = (255, 165, 0)  # Naranja para pool élite
            new_population.append(pool_copy)
        print(f"Incluidos {pool_count} individuos del pool élite")
    
    # 3. Generar el resto mediante cruzamiento estratégico
    breeding_pool = super_elites + greedy_individuals + elites
    if not breeding_pool:  # Si no hay buenos individuos, usar todos
        breeding_pool = population
    
    while len(new_population) < population_size:
        # Selección de padres con sesgo hacia mejores individuos
        if len(breeding_pool) >= 2:
            if np.random.rand() < 0.6:  # 60% del tiempo, usar selección por torneo
                parent1 = genetic_algorithm.tournament_selection(breeding_pool)
                parent2 = genetic_algorithm.tournament_selection(breeding_pool)
            else:  # 40% del tiempo, selección completamente aleatoria para diversidad
                parent1, parent2 = random.sample(breeding_pool, 2)
        else:
            parent1 = parent2 = breeding_pool[0]
        
        # Cruzamiento con diferentes estrategias
        crossover_methods = ["uniform", "arithmetic", "simulated_binary"]
        crossover_type = random.choice(crossover_methods)
        child = genetic_algorithm.advanced_crossover(parent1, parent2, crossover_type)
        
        # Mutación inteligente
        genetic_algorithm.intelligent_mutation(child)
        
        new_population.append(child)
    
    # Actualizar la población original
    for i in range(len(population)):
        if i < len(new_population):
            population[i].weights = new_population[i].weights
            population[i].biases = new_population[i].biases
            population[i].color = new_population[i].color

def select_fittest(population, top_k=10):
    sorted_population = sorted(population, key=lambda dino: dino.score, reverse=True)
    return sorted_population[:top_k]

def evolve(parent1, parent2, mutation_rate=0.1):
    if 'genetic_algorithm' in globals():
        return genetic_algorithm.advanced_crossover(parent1, parent2, "uniform")
    
    # Fallback al método original
    child = Dinosaur(parent1.id, parent1.color, True)
    
    for i in range(len(parent1.weights)):
        w1 = parent1.weights[i].numpy().flatten()
        w2 = parent2.weights[i].numpy().flatten()
        mask = np.random.rand(len(w1)) < 0.5
        child_w = np.where(mask, w1, w2)

        mutation = np.random.randn(*child_w.shape) * mutation_rate
        child_w += (np.random.rand(*child_w.shape) < mutation_rate * 5) * mutation

        shape = parent1.weights[i].shape
        child.weights[i].assign(tf.convert_to_tensor(child_w.reshape(shape), dtype=tf.float32))

        b1 = parent1.biases[i].numpy()
        b2 = parent2.biases[i].numpy()
        mask_b = np.random.rand(len(b1)) < 0.5
        child_b = np.where(mask_b, b1, b2)
        mutation_b = np.random.randn(*child_b.shape) * mutation_rate
        child_b += (np.random.rand(*child_b.shape) < mutation_rate * 5) * mutation_b
        child.biases[i].assign(tf.convert_to_tensor(child_b, dtype=tf.float32))

    return child