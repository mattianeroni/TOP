import math
import heapq
import random
import logging 
import operator
import collections
import numpy as np 

from typing import Callable

from top.problem import Problem, Solution, Route, Customer
from top.heuristic import savings_heuristic, constructive_heuristic
from top.localsearch import opt2, shaking, insert, remove

logger = logging.getLogger(__name__)

def generate_greedy_solution(p: Problem, heuristic: Callable) -> Solution:
    """ Generate a greedy solution to the problem using the savings heuristic and testing 
    different ranges for the alpha value (i.e., weight assignet to the reward vs distance covered) 
    to exhaustively find the best. 
    It returns the greedy solution and the corresponding alpha value. """
    solution, reward, best_alpha = None, 0.0, None
    for alpha in np.arange(0.0, 1.05, 0.05):
        new_sol = heuristic(p, alpha=alpha, beta=0.99999)
        new_reward = new_sol.reward
        if solution is None or new_reward > reward:
            solution, reward, best_alpha = new_sol, new_reward, alpha
    return solution, best_alpha 

def multi_start_metaheuristic(
        p: Problem, 
        heuristic: Callable,
        max_iter: int = 3000, 
        enlarge_search_iter: int = 30, 
        beta_start: float = 0.99, 
        beta_step: float = 0.05, 
        min_beta: float = 0.1,
        n_elites: int = 5,
        short_simulation_n_iter: int = 1000,
        long_simulation_n_iter: int = 50_000,
    ) -> tuple[Solution, Solution]:
    """ A multi-start metaheuristic implementing the following steps:
            1) Initialises the starting solution using a greedy version of the savings heuristic
            2) Set best solution equal to the initial one and mark it as elite solution
            3) Generate a new solution with the savings heuristic using a biased-randomised selection of savings
            4) Eventually update the best solution and the set of elite stochastic solutions
            5) If a long number of iterations occur without any improvement, beta value is made bigger to enable the algorithm to explore a larger solution space.
            6) Repeat from 3 until the maximum number of iterations has not been reached
            7) Run a long simulation of elite solutions to verify the most robust (i.e., best stochastic solution)
            8) Return the best solution and the best stochastic solution found

        Parameters: 
            - p: the problem instance to solve 
            - max_iter: the number of iterations 
            - enlarge_search_iter: number of iterations with no improvement after which beta is increased
            - beta_start: starting value of beta
            - beta_step: how much beta is decreased 
            - min_beta: minimum value beta can assume (to avoid too random search)
            - n_elites: number of elite solutions considered in the stochastic version
            - short_simulation_n_iter: number of iterations for a short simulation in the stochastic version
            - long_simulation_n_iter: number of iterations for a long simulation in the stochastic version
    """
    logger.info(f"Running multi-start metaheuristic on instance {p.instance}.")
    logger.info(f"Heuristic used: {heuristic.__name__}.")
    best_solution, alpha = generate_greedy_solution(p, heuristic)
    best_reward, best_stochastic_solution, best_stochastic_reward = best_solution.reward, best_solution, best_solution.stochastic_reward(n_iter=short_simulation_n_iter)
    no_improvement_count, beta, improvements_count = 0, beta_start, 0
    elites = collections.deque([best_stochastic_solution], maxlen=n_elites)
    logger.info(f"Best alpha value detected: {alpha}.")
    
    logger.info("Iterative exploration started.")
    for i in range(max_iter):
        # Generate a new solution using biased randomisation
        solution = heuristic(p, alpha=alpha, beta=beta)
        reward = solution.reward 
        # NOTE: Simulation is super fast, no need to check if the solution is "elite" before running it
        stochastic_reward = solution.stochastic_reward(n_iter=short_simulation_n_iter)

        # Stochastic solution update
        if stochastic_reward > best_stochastic_reward:
            best_stochastic_solution, best_stochastic_reward = solution, stochastic_reward
            elites.append(solution)
        
        # Eventually update the best solution found so far 
        if reward > best_reward:
            improvements_count += 1
            best_solution, best_reward, beta = solution, reward, beta_start 
            continue
        
        # After no_imporvement_count iterations with no improvement we reduce beta to allow
        # the algorithm to explore a larger space of solutions.
        no_improvement_count += 1
        if no_improvement_count > enlarge_search_iter:
            no_improvement_count = 0
            beta = max(beta - beta_step, min_beta) 

    logger.info(f"Iterative exploration concluded with {improvements_count} improvements on top of greedy solution.")
    logger.info(f"Long simulation of {len(elites)} elite solutions started.")
    elite_sol_to_stochastic_reward = [(i, i.stochastic_reward(n_iter=long_simulation_n_iter)) for i in elites]
    logger.info(f"Long simulation of {len(elites)} elite solutions concluded.")

    best_stochastic_solution, _ = heapq.nlargest(1, elite_sol_to_stochastic_reward, key=operator.itemgetter(1))[0]

    logger.info(f"Concluded run of multi-start metaheuristic on instance {p.instance}.")
    return best_solution, best_stochastic_solution


def local_search_metaheuritic(
        p: Problem, 
        max_iter: int = 1000,
        n_elites: int = 5,
        short_simulation_n_iter: int = 1000,
        long_simulation_n_iter: int = 50_000,
    ) -> tuple[Solution, Solution]:
    """
        A metaheuristic where the local search is based on 3 operators: 
            - insert: add to a random route as many customers as possibles
            - remove: remove from a random route a customer 
            - shaking: delete a random route and re-build it using the savings heuristic 

        At each iteration, even if the new solution is not outperforming the base solution, 
        we offer a possibility to upadate anyway the base solution, thorugh a random probability
        depending on the temperature, where the concept of temperature is similar to the one 
        of the Simulated Annealing.

        Parameters: 
            - p: the problem instance to solve 
            - max_iter: the number of iterations 
            - n_elites: number of elite solutions considered in the stochastic version
            - short_simulation_n_iter: number of iterations for a short simulation in the stochastic version
            - long_simulation_n_iter: number of iterations for a long simulation in the stochastic version
    """
    logger.info(f"Running local search metaheuristic on instance {p.instance}.")
    # Best solution and best stochastic solution 
    best_solution, alpha = generate_greedy_solution(p, savings_heuristic)
    best_stochastic, best_reward, best_stochastic_reward = best_solution, best_solution.reward, best_solution.stochastic_reward(n_iter=short_simulation_n_iter)
    # Base solution 
    base_solution, base_reward = best_solution, best_reward
    # Init temperature
    temperature, dt = 1000, 0.999
    # init stochastic elite solutions
    elites = collections.deque([best_solution], maxlen=n_elites)
    logger.info(f"Best alpha value detected: {alpha}.")
    
    logger.info("Iterative exploration started.")
    for i in range(max_iter):
        local_operator = random.randrange(0, 3)
        if local_operator == 0:
            solution = remove(p, base_solution)
        elif local_operator == 1: 
            solution = insert(p, base_solution)
        elif local_operator == 2:
            solution = shaking(p, base_solution, alpha=alpha, beta=0.3)
        
        reward = solution.reward 

        # Eventually update the best solution and the base solution
        if reward >= base_reward:
            base_solution, base_reward = solution, reward
            stochastic_reward = solution.stochastic_reward(n_iter=short_simulation_n_iter)

            if reward > best_reward:
                best_solution, best_reward = solution, reward
            
            if stochastic_reward > best_stochastic_reward:
                best_stochastic_solution, best_stochastic_reward = solution, stochastic_reward
                elites.append(solution)

        else:
            update_prob = math.exp((reward - base_reward) / temperature)
            if update_prob > random.random():
                base_solution, base_reward = solution, reward

        temperature *= dt
    
    logger.info(f"Iterative exploration concluded.")
    logger.info(f"Long simulation of {len(elites)} elite solutions started.")
    elite_sol_to_stochastic_reward = [(i, i.stochastic_reward(n_iter=long_simulation_n_iter)) for i in elites]
    logger.info(f"Long simulation of {len(elites)} elite solutions concluded.")

    best_stochastic_solution, _ = heapq.nlargest(1, elite_sol_to_stochastic_reward, key=operator.itemgetter(1))[0]

    logger.info(f"Concluded run of multi-start metaheuristic on instance {p.instance}.")
    return best_solution, best_stochastic_solution