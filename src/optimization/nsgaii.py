import random
import logging

from evaluation.evaluation_method   import EvaluationMethod
from .individual                    import Individual

logger = logging.getLogger(__name__)

"""
The NSGA-II algorithm is a popular genetic algorithm for solving multi-objective optimization problems. 
It was first proposed by Deb et al. in 2002 as an improvement over the original Non-dominated Sorting Genetic Algorithm (NSGA).
https://ieeexplore.ieee.org/abstract/document/996017?casa_token=dDGNqrFAX2YAAAAA:zLPH-VzJoi06EqWav3xZuAB8pgTXSV7SsAm7sLSv9yU4g_-VStrzhtsJjnDwe6hTkMeuXAf1nA
"""

# TODO: avaliar implementação exemplo:
# https://github.com/smkalami/nsga2-in-python/blob/main/nsga2.py

class NSGAII:
    def __init__(self, population_size, maximum_generations, aerial_evaluation_method: EvaluationMethod, aquatic_evaluation_method: EvaluationMethod, seed=0, mutation_rate=0.1):
        self.population_size            = population_size
        self.maximum_generations        = maximum_generations
        self.aerial_evaluation_method   = aerial_evaluation_method
        self.aquatic_evaluation_method  = aquatic_evaluation_method
        self.seed                       = seed
        self.mutation_rate              = mutation_rate

    def run(self):
        """
        Executes the NSGA-II optimization process and returns Pareto fronts.
        """
        population = self._initialize_population()
        for generation in range(1, self.maximum_generations + 1):
            logger.info(f"Generation {generation}/{self.maximum_generations}")

            # Evaluate population
            for individual in population:
                individual.aerial_fitness = self.aerial_evaluation_method.evaluate(individual.gene1, individual.gene2, individual.gene3)
                individual.aquatic_fitness = self.aquatic_evaluation_method.evaluate(individual.gene1, individual.gene2, individual.gene3)

            # Perform non-dominated sorting
            fronts = self._fast_non_dominated_sort(population)
            
            # TODO: implementar max generations ou convergência
            if (generation == self.maximum_generations):
                continue

            # Calculate crowding distance for each front
            for front in fronts:
                self._crowding_distance(front)

            # Selection, crossover, and mutation
            population = self._create_next_generation(population)
            
        return fronts

    def _initialize_population(self):
        """
        Initializes a random population of individuals.
        """
        population = []
        
        random.seed(self.seed)
        
        for _ in range(self.population_size):
            # TODO: ajustar de acordo com os genes
            individual = Individual(
                gene1=random.uniform(-5, 5),
                gene2=random.uniform(-5, 5),
                gene3=random.uniform(-5, 5),
            )
            population.append(individual)
        return population

    def _fast_non_dominated_sort(self, population):
        """
        Performs non-dominated sorting on the population.
        Returns a list of fronts.
        """
        fronts = [[]]
        for p in population:
            p.dominated_solutions = []
            p.domination_count = 0

            for q in population:
                if self._dominates(p, q):
                    p.dominated_solutions.append(q)
                elif self._dominates(q, p):
                    p.domination_count += 1

            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            fronts.append(next_front)
            i += 1

        return fronts[:-1]

    def _crowding_distance(self, front):
        """
        Calculates the crowding distance for each individual in a front.
        """
        if not front:
            return

        for individual in front:
            individual.crowding_distance = 0

        for m in range(2):  # Two objectives: aerial and aquatic fitness
            # Ordena pelo objetivo atual
            if m == 0:
                front.sort(key=lambda x: x.aerial_fitness)
            else:
                front.sort(key=lambda x: x.aquatic_fitness)
            
            # Define distância infinita para as extremidades    
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Calcula a distância para os indivíduos internos
            min_value = front[0].aerial_fitness if m == 0 else front[0].aquatic_fitness
            max_value = front[-1].aerial_fitness if m == 0 else front[-1].aquatic_fitness

            if max_value - min_value == 0:
                continue  # Evita divisão por zero

            for i in range(1, len(front) - 1):
                next_fitness = front[i + 1].aerial_fitness if m == 0 else front[i + 1].aquatic_fitness
                prev_fitness = front[i - 1].aerial_fitness if m == 0 else front[i - 1].aquatic_fitness

                front[i].crowding_distance += (next_fitness - prev_fitness) / (max_value - min_value)

    def _dominates(self, individual1:Individual, individual2:Individual):
        """
        Checks if individual1 dominates individual2.
        An individual dominates another if:
        - It is better (lower value) in at least one objective
        - It is not worse in any objective
        """
        
        # TODO: ajustar de acordo com a métrica objetivo definida (melhor significa maior ou menor?)
        
        better_in_one = False

        # Comparação para cada objetivo (fitness aéreo e fitness aquático)
        if individual1.aerial_fitness < individual2.aerial_fitness:
            better_in_one = True
        elif individual1.aerial_fitness > individual2.aerial_fitness:
            return False  # Se piora em algum objetivo, não domina

        if individual1.aquatic_fitness < individual2.aquatic_fitness:
            better_in_one = True
        elif individual1.aquatic_fitness > individual2.aquatic_fitness:
            return False  # Se piora em algum objetivo, não domina

        return better_in_one

    def _create_next_generation(self, population):
        """
        Creates the next generation using tournament selection, crossover, and mutation.
        """
        next_generation = []

        # Perform tournament selection to choose parents
        while len(next_generation) < self.population_size:
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)

            # Apply crossover to generate a child
            child = self._crossover(parent1, parent2)

            # Apply mutation to the child
            self._mutate(child)

            # Add the child to the next generation
            next_generation.append(child)

        return next_generation

    def _tournament_selection(self, population, tournament_size=2):
        """
        Perform tournament selection to choose one individual from the population.
        """
        # Randomly select individuals for the tournament
        tournament = random.sample(population, tournament_size)
        
        # Sort by rank first, then by crowding distance (higher is better)
        tournament.sort(key=lambda x: (x.rank, -x.crowding_distance))
        
        # Return the best individual from the tournament
        return tournament[0]

    def _crossover(self, parent1, parent2):
        """
        Performs crossover between two parents to produce a child.
        """
        # TODO: adequar ao formato do chromossomo
        child = Individual(
            gene1=(parent1.gene1 + parent2.gene1) / 2,
            gene2=(parent1.gene2 + parent2.gene2) / 2,
            gene3=(parent1.gene3 + parent2.gene3) / 2,
        )
        return child

    def _mutate(self, individual, mutation_rate=0.1):
        """
        Perform mutation on an individual's parameters.
        """
        # TODO: adequar ao formato do chromossomo
        if random.random() < mutation_rate:
            individual.gene1 += random.uniform(-1, 1)
        if random.random() < mutation_rate:
            individual.gene2 += random.uniform(-1, 1)
        if random.random() < mutation_rate:
            individual.gene3 += random.uniform(-1, 1)
