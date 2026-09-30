import numpy as np
import matplotlib.pyplot as plt



NUM_DEMAND_POINTS = 40   
NUM_CANDIDATE_LOCATIONS = 15 
NUM_STATIONS_TO_SELECT = 4 

POP_SIZE = 80           
GENERATIONS = 150         
MUTATION_RATE = 0.15     


demand_coords = np.random.rand(NUM_DEMAND_POINTS, 2) * 100
candidate_coords = np.random.rand(NUM_CANDIDATE_LOCATIONS, 2) * 100

COVERAGE_WEIGHT = 1.5   
UTILIZATION_WEIGHT = 40.0 

dist_matrix = np.zeros((NUM_DEMAND_POINTS, NUM_CANDIDATE_LOCATIONS))
for i in range(NUM_DEMAND_POINTS):
    for j in range(NUM_CANDIDATE_LOCATIONS):
        dist_matrix[i, j] = np.linalg.norm(demand_coords[i] - candidate_coords[j])


def calculate_fitness(chromosome):
    selected_indices = np.where(chromosome == 1)[0]

    if len(selected_indices) == 0:
        return float('inf')
    

    sub_dist_matrix = dist_matrix[:, selected_indices]
    min_distances = np.min(sub_dist_matrix, axis=1)
    coverage_cost = np.sum(min_distances) * COVERAGE_WEIGHT
    
 
    nearest_station_sub_idx = np.argmin(sub_dist_matrix, axis=1)
    actual_station_indices = selected_indices[nearest_station_sub_idx]
    

    counts = np.bincount(actual_station_indices, minlength=NUM_CANDIDATE_LOCATIONS)
    active_counts = counts[selected_indices]
    

    idle_stations_penalty = np.sum(active_counts == 0) * 150
    

    utilization_balance = np.std(active_counts) if len(active_counts) > 0 else 0
    utilization_cost = (utilization_balance + idle_stations_penalty) * UTILIZATION_WEIGHT
    
    return coverage_cost + utilization_cost


def create_individual():
    
    ind = np.zeros(NUM_CANDIDATE_LOCATIONS, dtype=int)
    chosen = np.random.choice(NUM_CANDIDATE_LOCATIONS, NUM_STATIONS_TO_SELECT, replace=False)
    ind[chosen] = 1
    return ind

def crossover(parent1, parent2):

    point = np.random.randint(1, NUM_CANDIDATE_LOCATIONS - 1)
    child = np.concatenate([parent1[:point], parent2[point:]])
    
    active = np.sum(child)
    if active > NUM_STATIONS_TO_SELECT:
        ones = np.where(child == 1)[0]
        child[np.random.choice(ones, active - NUM_STATIONS_TO_SELECT, replace=False)] = 0
    elif active < NUM_STATIONS_TO_SELECT:
        zeros = np.where(child == 0)[0]
        child[np.random.choice(zeros, NUM_STATIONS_TO_SELECT - active, replace=False)] = 1
    return child

def mutate(individual):

    if np.random.rand() < MUTATION_RATE:
        ones = np.where(individual == 1)[0]
        zeros = np.where(individual == 0)[0]
        if len(ones) > 0 and len(zeros) > 0:
            individual[np.random.choice(ones)] = 0
            individual[np.random.choice(zeros)] = 1
    return individual



population = [create_individual() for _ in range(POP_SIZE)]
best_overall = None
best_fitness_overall = float('inf')

print("Starting Genetic Algorithm Optimisation...")

for gen in range(GENERATIONS):
    fitnesses = [calculate_fitness(ind) for ind in population]
    

    gen_best_idx = np.argmin(fitnesses)
    if fitnesses[gen_best_idx] < best_fitness_overall:
        best_fitness_overall = fitnesses[gen_best_idx]
        best_overall = population[gen_best_idx].copy()
 
    new_pop = []
    for _ in range(POP_SIZE):
        i1, i2 = np.random.randint(0, POP_SIZE), np.random.randint(0, POP_SIZE)
        parent1 = population[i1] if fitnesses[i1] < fitnesses[i2] else population[i2]
        
        i3, i4 = np.random.randint(0, POP_SIZE), np.random.randint(0, POP_SIZE)
        parent2 = population[i3] if fitnesses[i3] < fitnesses[i4] else population[i4]
        
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_pop.append(child)
        
    population = new_pop


optimal_locations = np.where(best_overall == 1)[0]

sub_dist = dist_matrix[:, optimal_locations]
nearest_station_sub_idx = np.argmin(sub_dist, axis=1)
final_assignments = optimal_locations[nearest_station_sub_idx]
final_counts = np.bincount(final_assignments, minlength=NUM_CANDIDATE_LOCATIONS)

print("\n=== OPTIMISATION COMPLETE ===")
print(f"Optimal Station Allocated Indices: {optimal_locations}")
print("\nUtilization Metric Breakdown:")
for loc in optimal_locations:
    print(f"  Station candidate #{loc} satisfies {final_counts[loc]} nearby EV demand clusters.")
print(f"\nFinal Best Minimised Fitness Value: {best_fitness_overall:.4f}")


plt.figure(figsize=(10, 8))


colors = plt.cm.get_cmap('tab10', NUM_STATIONS_TO_SELECT)
for i, station_idx in enumerate(optimal_locations):
 
    pts = np.where(final_assignments == station_idx)[0]
    color = colors(i)
    

    plt.scatter(demand_coords[pts, 0], demand_coords[pts, 1], 
                color=color, alpha=0.6, edgecolors='k', label=f'Demand served by St. {station_idx}' if len(pts)>0 else "")
    

    for pt in pts:
        plt.plot([demand_coords[pt, 0], candidate_coords[station_idx, 0]], 
                 [demand_coords[pt, 1], candidate_coords[station_idx, 1]], 
                 color=color, linestyle='--', alpha=0.3)


unselected = np.where(best_overall == 0)[0]
plt.scatter(candidate_coords[unselected, 0], candidate_coords[unselected, 1], 
            color='grey', marker='x', s=80, alpha=0.5, label='Unselected Candidate Locations')


plt.scatter(candidate_coords[optimal_locations, 0], candidate_coords[optimal_locations, 1], 
            color='red', marker='^', s=150, edgecolors='black', linewidths=2, label='OPTIMIZED STATIONS')

plt.title("EV Charging Station Optimal Allocation Map\n(Minimizing Distance Coverage & Maximizing Utilization Balance)")
plt.xlabel("X Coordinate Map Scale")
plt.ylabel("Y Coordinate Map Scale")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()



