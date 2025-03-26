import heapq
import pygame
import numpy as np

def num2counter(num):
    estanteriah = ((num-1)//8)%3
    estanteriav = (num-1)//24
    x = (num-1)%2 + 2 + (-1)**(num%2) + estanteriah*4
    y = ((num-1)%8)//2 + 1 + estanteriav*5
    return (y,x)
    
def heuristic(a, b):
    """Distancia Manhattan."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos, warehouse):
    """Devuelve las celdas vecinas accesibles."""
    x, y = pos
    neighbors = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(warehouse) and 0 <= ny < len(warehouse[0]) and warehouse[nx][ny] != '#':
            neighbors.append((nx, ny))
    
    return neighbors

def closest_reachable(warehouse, goal):
    """Encuentra la celda más cercana a un objetivo si es una estantería."""
    if warehouse[goal[0]][goal[1]] != '#':
        return goal  # Si no es una estantería, es válido
    
    queue = [(goal, 0)]
    visited = set()
    
    while queue:
        (x, y), dist = queue.pop(0)
        if (x, y) in visited:
            continue
        visited.add((x, y))
        
        for nx, ny in get_neighbors((x, y), warehouse):
            if warehouse[nx][ny] == '.':
                return (nx, ny)  # Primera celda libre encontrada
            queue.append(((nx, ny), dist + 1))
    
    return None  # No hay camino disponible

def astar(warehouse, start, goal, occupied_positions):
    """Ejecuta A* evitando posiciones ocupadas por otros agentes."""
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while frontier:
        _, current = heapq.heappop(frontier)
        
        if current == goal:
            break
        
        for next_pos in get_neighbors(current, warehouse):
            if next_pos in occupied_positions:
                continue  # Evita colisiones en el mismo instante
            
            new_cost = cost_so_far[current] + 1
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + heuristic(next_pos, goal)
                heapq.heappush(frontier, (priority, next_pos))
                came_from[next_pos] = current
    
    # Reconstruir el camino
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    
    return path if path and path[0] == start else []  # Retorna camino o vacío si no hay solución

#Detecta colisiones, considerando que las trayectorias pueden pausarse al inicio
def detect_collisions(paths, waits = None):
    if waits is None:
        waits = np.zeros(len(paths), dtype=int)
    
    paths = [np.pad(paths[i],[[waits[i],0],[0,0]],"edge") for i in range(len(paths))]
    
    max_size = max([p.shape[0] for p in paths])
    paths = np.array([np.pad(path, [[0,max_size - path.shape[0]],[0,0]], "edge") for path in paths])
    
    collisions = []
    for indice in range(paths.shape[1]):
        if len(np.unique(paths[:, indice], axis=0)) < paths.shape[0]:
            collisions.append(indice)
    
    return collisions

def cruce_colision(paths,time):
    current_positions = [tuple(path[min(time,len(path)-1)]) for path in paths]
    next_moves = [tuple(path[min(time + 1, len(path) - 1)]) for path in paths]
    colision = len(set(next_moves)) < len(next_moves)
    cruce = len({frozenset(sublista) for sublista in zip(current_positions, next_moves)}) < len(paths)
    return cruce or colision

#Explora en anchura alrededor de la posicion actual hasta encontrar una que no se encuentre en el camino del otro agente
def calc_posicion_desocupada(path_stay, initial_pos, mapa):
    frontier = [(initial_pos,0)]
    found = False
    
    while frontier:
        current = frontier.pop(0)
        if current[0] not in path_stay[min(current[1],len(path_stay)-1):]:
            found = True
            break
        for next_pos in get_neighbors(current[0], mapa):
            if not cruce_colision([[current[0],next_pos], path_stay[min(current[1], len(path_stay)):]], 0):
                frontier.append((next_pos, current[1]+1, current))
    
    if not found:
        return None
    #ahora reconstruimos el camino hacia atrás
    path = [current[0]]
    while current[1] != 0:
        current = current[2]
        path.append(current[0])
    path.reverse()
    return path

def wait_colision_calc(path_stay, path_move, time, max_wait):
    path_stay = path_stay[min(time,len(path_stay)-1):]
    camino_espera = path_move[min(time,len(path_move)-1):]
    for wait_time in range(max_wait+1):
        colision = any( cruce_colision([camino_espera, path_stay], wait_t) for wait_t in range(wait_time+1) )
        if not colision: return wait_time
        camino_espera.insert(0,camino_espera[0])
    return -1

def plan_initial_paths(warehouse, agents):
    """Calcula las rutas iniciales para todos los agentes."""
    paths = []
    for agent in agents:
        path = astar(warehouse, agent['start'], agent['goal'], set())
        paths.append(path)
    
    print("Se calcularon los caminos")
    return paths

def resolve_collisions(warehouse, agents, paths, max_wait=2):
    """Resuelve las colisiones entre agentes y planifica rutas finales."""
    if not all(paths):  # Verificar si todos los caminos se pudieron planificar
        print("No se pudieron planificar los caminos")
        return paths
    
    # Determinar qué agente se mueve y cuál se queda quieto en caso de colisión
    path_lengths = [len(path) for path in paths]
    if path_lengths[0] == path_lengths[1]:
        # Si los caminos son igual de largos, elegir uno al azar
        stay_agent = np.random.randint(0, 2) 
    else:
        stay_agent = np.argmin(path_lengths)
    move_agent = 1 - stay_agent
    
    time = 0
    success = all(path[-1] == path[0] for path in paths)  # Si todos los agentes llegaron a su destino
    
    while not success:
        current_positions = [path[min(time, len(path)-1)] for path in paths]  # posición actual
        next_moves = [path[min(time + 1, len(path) - 1)] for path in paths]  # posición en el siguiente turno
        
        if cruce_colision(paths, time):  # El siguiente movimiento lleva a colisión o cruce de caminos
            print(f"Se detectó una colisión en {next_moves[stay_agent]}, tiempo", time+1)
            
            # Intentar resolver la colisión haciendo que un agente espere
            wait_time = wait_colision_calc(paths[stay_agent], paths[move_agent], time, max_wait)
            
            if wait_time != -1:
                print("Se pudo evitar esperando")
                paths[move_agent][time+1:time+1] = [paths[move_agent][time] for _ in range(wait_time)]
            else:
                print("No se pudo resolver la colisión esperando")
                
                # Intentar calcular un camino alternativo que esquive al otro agente
                paths = calculate_alternative_path(warehouse, paths, current_positions, next_moves, stay_agent, move_agent, time)
        
        if time > 1000:  # Por si se rompe el programa
            break
            
        success = all(pos == paths[i][-1] for i, pos in enumerate(current_positions))
        time += 1
    
    return paths

def calculate_alternative_path(warehouse, paths, current_positions, next_moves, stay_agent, move_agent, time):
    """Calcula un camino alternativo cuando esperar no es suficiente para resolver la colisión."""
    # Intento 1: Esquivar al agente que se queda quieto
    camino_esquive = astar(
        warehouse, 
        current_positions[move_agent], 
        paths[move_agent][-1], 
        [current_positions[stay_agent], next_moves[stay_agent]]
    )
    
    if camino_esquive is None:
        print("No se logró calcular el camino esquive")
    else:
        print("Camino esquive calculado con éxito")
        if cruce_colision([camino_esquive, [current_positions[stay_agent], next_moves[stay_agent]]], 0):
            print("Camino esquive falló")
            camino_esquive = None
    
    # Si el camino esquivando es muy largo o no se pudo calcular, probamos con otro método
    if camino_esquive is None or len(camino_esquive) > len(paths[move_agent])-time + 3:
        # Intento 2: Calcular un camino de retroceso
        camino_retroceso = calculate_backtrack_path(
            warehouse, 
            paths, 
            current_positions, 
            stay_agent, 
            move_agent, 
            time
        )
        
        if camino_esquive is None and camino_retroceso is None:
            print("No se pudo planear un camino adecuado para evitar la colisión")
            return paths
        
        # Elegir el camino más corto entre los disponibles
        if camino_esquive: 
            print("Longitud camino esquive:", len(camino_esquive))
        if camino_retroceso: 
            print("Longitud camino retroceso:", len(camino_retroceso))
        
        camino_elegido = min(
            (path for path in [[camino_esquive, "esquiva"], [camino_retroceso, "retrocede"]] if path[0] is not None),
            key=lambda p: len(p[0])
        )
        print("Se eligió el camino que", camino_elegido[1])
        
        paths[move_agent][min(time, len(paths[move_agent])):] = camino_elegido[0]
    else:
        paths[move_agent][min(time, len(paths[move_agent])):] = camino_esquive
    
    return paths

def calculate_backtrack_path(warehouse, paths, current_positions, stay_agent, move_agent, time, max_wait=2):
    """Calcula un camino de retroceso cuando no se puede esquivar al otro agente."""
    # Buscamos una casilla cercana accesible que no esté en el camino del stay_agent
    inicio_camino_retroceso = calc_posicion_desocupada(
        paths[stay_agent][min(time, len(paths[stay_agent])):], 
        current_positions[move_agent], 
        warehouse
    )
    
    if inicio_camino_retroceso is None:
        print(f"El agente {move_agent} se encuentra acorralado por el agente {stay_agent}")
        return None
    
    final_camino_retroceso = list(reversed(inicio_camino_retroceso[:-1])) + paths[move_agent][min(time+1, len(paths[move_agent])):]
    camino_retroceso = inicio_camino_retroceso + final_camino_retroceso
    
    wait_time = wait_colision_calc(
        paths[stay_agent][min(time, len(paths[stay_agent])):], 
        camino_retroceso, 
        time + len(inicio_camino_retroceso) - 1, 
        max_wait
    )
    
    if wait_time == -1:
        print("No se pudo calcular el camino retroceso")
        return None
    
    print("Camino retroceso calculado con éxito")
    camino_retroceso[len(inicio_camino_retroceso):len(inicio_camino_retroceso)] = [inicio_camino_retroceso[-1]] * wait_time
    
    return camino_retroceso

# === PYGAME VISUALIZATION ===
def setup_pygame(warehouse):
    """Initialize Pygame and set up display."""
    pygame.init()
    TILE_SIZE = 40
    WIDTH, HEIGHT = len(warehouse[0]) * TILE_SIZE, len(warehouse) * TILE_SIZE
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Warehouse Pathfinding")
    clock = pygame.time.Clock()
    return screen, clock, TILE_SIZE

def create_agent_surfaces(agents, TILE_SIZE):
    """Create surfaces for agents and trails."""
    COLORS = {
        'blue': pygame.Color(0, 0, 255),
        'red': pygame.Color(255, 0, 0),
        'blue_trail': pygame.Color(100, 100, 255, 128),
        'red_trail': pygame.Color(255, 100, 100, 128),
        'start_point': pygame.Color(255, 255, 0)  # Yellow for start points
    }
    
    agent_surfaces = {}
    trail_surfaces = {}
    start_surface = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
    pygame.draw.rect(start_surface, COLORS['start_point'], (0, 0, TILE_SIZE, TILE_SIZE))

    for agent in agents:
        color = agent['color']
        # Create agent surface
        agent_surf = pygame.Surface((TILE_SIZE // 1.5, TILE_SIZE // 1.5), pygame.SRCALPHA)
        pygame.draw.circle(agent_surf, COLORS[color], (agent_surf.get_width() // 2, agent_surf.get_height() // 2), agent_surf.get_width() // 2)
        agent_surfaces[color] = agent_surf
        
        # Create trail surface
        trail_color = f"{color}_trail"
        trail_surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(trail_surf, COLORS[trail_color], (0, 0, TILE_SIZE, TILE_SIZE))
        trail_surfaces[color] = trail_surf
    
    return agent_surfaces, trail_surfaces, start_surface, COLORS

def draw_warehouse(screen, warehouse, TILE_SIZE):
    """Draw the warehouse grid."""
    screen.fill((255, 255, 255))
    for y, row in enumerate(warehouse):
        for x, cell in enumerate(row):
            color = (200, 200, 200) if cell == '.' else (100, 100, 100)
            pygame.draw.rect(screen, color, (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(screen, (0, 0, 0), (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)

def animate_agents(screen, clock, warehouse, agents, paths, agent_surfaces, trail_surfaces, start_surface, TILE_SIZE):
    """Animate agents following their paths."""
    agent_states = ['moving'] * len(agents)
    wait_counters = [0] * len(agents)
    agent_positions = [agent['start'] for agent in agents]
    step = [0] * len(agents)
    running = True
    max_wait = 10
    
    # Track visited tiles
    visited_tiles = [set() for _ in range(len(agents))]
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        draw_warehouse(screen, warehouse, TILE_SIZE)
        
        # Draw starting points in yellow
        for agent in agents:
            start_pos = agent['start']
            screen.blit(start_surface, (start_pos[1] * TILE_SIZE, start_pos[0] * TILE_SIZE))
        
        # Draw agent trails
        for i, agent in enumerate(agents):
            for pos in visited_tiles[i]:
                screen.blit(trail_surfaces[agent['color']], 
                          (pos[1] * TILE_SIZE, pos[0] * TILE_SIZE))
        
        # Update and draw agents
        for i, (agent, path) in enumerate(zip(agents, paths)):
            # Update agent state
            if step[i] >= len(path) - 1:
                agent_states[i] = "finished"
            elif path[step[i] + 1] == path[step[i]]:
                agent_states[i] = "waiting"
            else: 
                agent_states[i] = "moving"
            
            # Handle movement state
            if agent_states[i] == "moving":
                current_pos = path[step[i]]
                target_pos = path[step[i] + 1]
                
                # Mark current tile as visited
                visited_tiles[i].add(current_pos)
                
                # Smooth movement interpolation
                dx, dy = target_pos[1] - agent_positions[i][1], target_pos[0] - agent_positions[i][0]
                distance = (dx ** 2 + dy ** 2) ** 0.5
                speed = 0.1
                if distance > speed:
                    new_x = agent_positions[i][1] + dx * speed / distance
                    new_y = agent_positions[i][0] + dy * speed / distance
                    agent_positions[i] = (new_y, new_x)
                else:
                    agent_positions[i] = target_pos
                    step[i] += 1
                
                # Draw agent
                agent_pos = (agent_positions[i][1] * TILE_SIZE + TILE_SIZE // 2 - agent_surfaces[agent['color']].get_width() // 2,
                             agent_positions[i][0] * TILE_SIZE + TILE_SIZE // 2 - agent_surfaces[agent['color']].get_height() // 2)
                screen.blit(agent_surfaces[agent['color']], agent_pos)
            
            # Handle finished state
            elif agent_states[i] == "finished":
                # Mark all path tiles as visited
                for pos in path:
                    visited_tiles[i].add(pos)
                
                # Draw agent at final position
                agent_pos = (path[-1][1] * TILE_SIZE + TILE_SIZE // 2 - agent_surfaces[agent['color']].get_width() // 2,
                             path[-1][0] * TILE_SIZE + TILE_SIZE // 2 - agent_surfaces[agent['color']].get_height() // 2)
                screen.blit(agent_surfaces[agent['color']], agent_pos)
            
            # Handle waiting state
            elif agent_states[i] == "waiting":
                wait_counters[i] += 1
                if wait_counters[i] == max_wait:
                    wait_counters[i] = 0
                    step[i] += 1
                
                # Mark current tile as visited
                visited_tiles[i].add(path[step[i]])
                
                # Draw agent
                agent_pos = (agent_positions[i][1] * TILE_SIZE + TILE_SIZE // 2 - agent_surfaces[agent['color']].get_width() // 2,
                             agent_positions[i][0] * TILE_SIZE + TILE_SIZE // 2 - agent_surfaces[agent['color']].get_height() // 2)
                screen.blit(agent_surfaces[agent['color']], agent_pos)
        
        # Check if animation is complete
        if all(state == "finished" for state in agent_states):
            pygame.time.wait(1000)  # Pause to show final state
            break
        
        pygame.display.flip()
        clock.tick(60)
    
    # Keep window open until manually closed
    waiting_for_close = True
    while waiting_for_close and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                waiting_for_close = False
        
        draw_warehouse(screen, warehouse, TILE_SIZE)
        
        # Draw start points
        for agent in agents:
            start_pos = agent['start']
            screen.blit(start_surface, (start_pos[1] * TILE_SIZE, start_pos[0] * TILE_SIZE))
        
        # Draw final trail state
        for i, agent in enumerate(agents):
            for pos in visited_tiles[i]:
                screen.blit(trail_surfaces[agent['color']], 
                          (pos[1] * TILE_SIZE, pos[0] * TILE_SIZE))
        
        # Draw agents in final positions
        for i, (agent, path) in enumerate(zip(agents, paths)):
            agent_pos = (path[-1][1] * TILE_SIZE + TILE_SIZE // 2 - agent_surfaces[agent['color']].get_width() // 2,
                         path[-1][0] * TILE_SIZE + TILE_SIZE // 2 - agent_surfaces[agent['color']].get_height() // 2)
            screen.blit(agent_surfaces[agent['color']], agent_pos)
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

def main():
    # Modelo del almacén
    warehouse = [
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
        list("............."),
    ]

    agents = [
        {'start': (5, 1), 'goal': 36, 'color': 'blue'},  # (y, x)
        {'start': (5, 12), 'goal': 33, 'color': 'red'},  # (y, x)
    ]

    # Ajustar objetivos si están en estanterías
    for agent in agents:
        agent["goal"] = num2counter(agent["goal"])

    # Planificación inicial de rutas
    paths = plan_initial_paths(warehouse, agents)
    
    # Resolver colisiones y planificar rutas finales
    final_paths = resolve_collisions(warehouse, agents, paths)
    
    # Mostrar resultados
    for i in range(len(agents)):
        print(f"Camino para el agente {agents[i]['color']}: {final_paths[i]}")

    # Visualizacion
    screen, clock, TILE_SIZE = setup_pygame(warehouse)
    agent_surfaces, trail_surfaces, start_surface, COLORS = create_agent_surfaces(agents, TILE_SIZE)
    
    # Animar a los agentes
    animate_agents(screen, clock, warehouse, agents, paths, agent_surfaces, trail_surfaces, start_surface, TILE_SIZE)

if __name__ == "__main__":
    main()
