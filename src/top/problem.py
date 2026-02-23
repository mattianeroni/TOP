import math
import itertools
from dataclasses import dataclass
from collections import namedtuple
from top.simulation import get_route_stochastic_reward 

@dataclass 
class Customer:
    id: int 
    x: float 
    y: float 
    reward: int 

@dataclass(frozen=True)
class Problem:
    instance: str 
    n_trucks: int
    tmax: float
    customers: dict[int, Customer]
    # matrix of distances between customers
    dists: dict[tuple[int, int], float] 

@dataclass 
class Route:
    id: int 
    problem: Problem
    customers: list[int]
    length: float 
    reward: int 

    def stocastic_reward(self, n_iter: int) -> float:
        """ Run a montecarlo simulation of n_iter iterations to get the 
        stochastic reward of the route as reward * p (with p probability to 
        conclude the route in tmax) """
        return get_route_stochastic_reward(self.problem, self, n_iter)

@dataclass
class Solution:
    problem: Problem 
    routes: dict[int, Route]
    c_to_r: dict[int, int]

    @property 
    def reward(self) -> int:
        return sum(i.reward for i in self.routes.values())

    @property 
    def length(self) -> float:
        return sum(i.length for i in routes.values())

    def stochastic_reward(self, n_iter: int) -> float:
        """ Run a montecarlo simulation of each route to get the total stochastic reward of the solution. """
        return sum(get_route_stochastic_reward(self.problem, i, n_iter) for i in self.routes.values())



def euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    return round(math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2)), 2)

def read_instance(filename: str) -> Problem:
    with open(filename, 'r') as f:
        lines = f.readlines()
    n_trucks = int(lines[1].split(" ")[1])
    tmax = float(lines[2].split(" ")[1])

    # NOTE: in the benchmarks origin is always forst customer and destination is always the last
    customers = {}
    for i, line in enumerate(lines[3:]):
        x, y, reward = line.split("\t")
        customers[i] = Customer(id=i, x=float(x), y=float(y), reward=int(reward))
    
    dists = {}
    for (id_1, c_1), (id_2, c_2) in itertools.combinations(customers.items(), 2):
        distance = euclidean(c_1.x, c_1.y, c_2.x, c_2.y)
        dists[id_1, id_2] = distance 
        dists[id_2, id_1] = distance 
    
    return Problem(
        instance = filename, 
        n_trucks = n_trucks, 
        tmax = tmax, 
        customers = customers, 
        dists = dists,
    )