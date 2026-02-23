import math 
import heapq
import random
import operator
import itertools
import collections

from typing import Any

from top.problem import Problem, Solution, Route, Customer


Saving = collections.namedtuple('Saving', ['inode', 'jnode', 'value', 'distance_saving', 'reward_saving'])
ConstructiveStep = collections.namedtuple('ConstructiveStep', ['id', 'customer', 'value'])

def init_solution(p: Problem) -> Solution:
    """ We build a first dummy solution consisting of a route visiting each customer. """
    source, target, routes, route_id, c_to_r = 0, len(p.customers) - 1, {}, 0, {}
    tmax, dists = p.tmax, p.dists

    for id, c in p.customers.items(): 
        if id == source or id == target:
            continue
        if (t := dists[source, id] + dists[id, target]) <= tmax:
            route = Route(id=route_id, problem=p, customers=[source, id, target], reward=c.reward, length=t)
            c_to_r[id] = route_id
            routes[route_id] = route 
            route_id += 1

    return Solution(problem=p, routes=routes, c_to_r=c_to_r)

def generate_savings(p: Problem, alpha: float = 0.3) -> list[Saving]:
    """ Return the list of saving obtained from visiting 2 customers with the same route. """
    source, target, dists = 0, len(p.customers) - 1, p.dists
    savings = []
    for (inode, jnode), dist in p.dists.items():
        if inode == source or inode == target or jnode == source or jnode == target:
            continue
        distance_saving = dists[inode, target] + dists[source, jnode] - dist 
        reward_saving = p.customers[inode].reward + p.customers[jnode].reward
        value = alpha * distance_saving + (1.0 - alpha) * reward_saving
        savings.append(Saving(inode=inode, jnode=jnode, value=value, distance_saving=distance_saving, reward_saving=reward_saving))
    return savings

def bra_selector(lst: collections.deque[Any], beta: float = 0.3) -> Any:
    """ A generator to select elements from a linked list. It can work as a greedy selector if beta is very close to 1, 
    or a biased randomised selection for lower beta. 
    Elements must have a <value> attribute by which they will be sorted from the highest to the lowest.
    Biased randomization consists of selecting the elements not from the best to the worst (greedy), but using a randomised 
    selection giving higher priority to best elements and lower priority to worse elements. 
    The probability function used in this case is quasi-geometric function:
                    f(x) = (1 - beta) ^ x
    For beta close to 1, it behaves as a greedy selection, and, as beta decreases, it gets closer to a uniform selection 
    where each candidate has the same probability to be selected. The correct approach lays in the middle.
    """
    n, options = len(lst), sorted(lst, key=operator.attrgetter("value"), reverse=True)
    for _ in range(n):
        idx = int(math.log(random.random(), 1.0 - beta)) % len(options)
        element = options.pop(idx)
        yield element

def merge_routes(p: Problem, solution: Solution, iroute: Route, jroute: Route) -> Route:
    """ Method to merge two routes. The id of the first route is reused, but a new instance is created. """
    source, target = 0, len(p.customers) - 1
    dists, ic_id, jc_id = p.dists, iroute.customers[-2], jroute.customers[1]
    for i in jroute.customers[1:-1]:
        solution.c_to_r[i] = iroute.id
    return Route(
        id = iroute.id,
        problem = iroute.problem,
        reward = iroute.reward + jroute.reward, 
        length = iroute.length - dists[ic_id, target] + dists[ic_id, jc_id] + jroute.length - dists[source, jc_id],
        customers = iroute.customers[:-1] + jroute.customers[1:]
    )   

def savings_heuristic(p: Problem, alpha: float = 0.3, beta: float = 0.3, solution: Solution | None = None) -> Solution:
    """ Basic savings based heuristic to generate a solution. For beta close to 1 it generates a greedy solution, 
    for beta closer to zero a different one. """
    source, target, dists, customers, tmax = 0, len(p.customers) - 1, p.dists, p.customers, p.tmax

    # build initial solution (if not provided) and savings
    if solution is None:
        solution = init_solution(p)
    savings_list = generate_savings(p, alpha)

    # merge routes iterative process
    for saving in bra_selector(savings_list, beta):

        # extract useful info and references
        ic_id, jc_id = saving.inode, saving.jnode
        ic, jc = customers[ic_id], customers[jc_id]
        iroute_id, jroute_id = solution.c_to_r.get(ic_id), solution.c_to_r.get(jc_id)
        if iroute_id is None or jroute_id is None or iroute_id == jroute_id:
            continue 
        iroute, jroute = solution.routes[iroute_id], solution.routes[jroute_id]

        # merge if possible
        if ic_id == iroute.customers[-2] and jc_id == jroute.customers[1]:
            if iroute.length - dists[ic_id, target] + jroute.length - dists[source, jc_id] + dists[ic_id, jc_id] <= tmax:
                merged_route = merge_routes(p, solution, iroute, jroute)
                solution.routes.pop(iroute_id)
                solution.routes.pop(jroute_id)
                solution.routes[merged_route.id] = merged_route

        # if number of routes is already equal to number of trucks early return 
        if len(solution.routes) <= p.n_trucks:
            break
    
    # take top n_trucks routes
    top_routes = heapq.nlargest(p.n_trucks, list(solution.routes.values()), key=operator.attrgetter("reward"))
    solution.routes = {i.id: i for i in top_routes}
    solution.c_to_r = {c: route.id for route in top_routes for c in route.customers if c != source and c != target}
    return solution

def constructive_heuristic(p: Problem, alpha: float = 0.3, beta: float = 0.3) -> Solution:
    """Constructive heuristic to build a feasible solution.

    Builds up to `p.n_trucks` routes sequentially. For each truck the
    algorithm repeatedly selects the next customer using a biased
    randomised selector (controlled by `beta`) that scores candidates by a
    linear combination of their reward and a distance-based gain (weight
    `alpha`). A customer is appended only if adding it keeps the route
    feasible within `p.tmax`. Returns a `Solution` containing the
    constructed `Route` objects and the `c_to_r` mapping.
    """
    routes, route_id, c_to_r, tmax, n_trucks, dists = {}, 0, {}, p.tmax, p.n_trucks, p.dists
    source, target = 0, len(p.customers) - 1
    visited_customers, c_to_r = set([source, target]), {}
    for i in range(n_trucks):
        route, cnode, cdist, creward = [source], source, 0.0, 0
        while len(
                (
                    customers := collections.deque(
                        [
                            ConstructiveStep(id=id, customer=c, value=c.reward * (1.0 - alpha) + alpha * (dists[cnode, target] - dists[cnode, id] - dists[id, target])) 
                                for id, c in p.customers.items() 
                            if id not in visited_customers and cdist + dists[cnode, id] + dists[id, target] <= tmax
                        ]
                    )
                )
            ) > 0:
            cs = next(bra_selector(customers, beta=beta))
            nnode = cs.id
            visited_customers.add(nnode)
            c_to_r[nnode] = route_id
            route.append(nnode)
            cdist += dists[cnode, nnode]
            creward += cs.customer.reward
            cnode = nnode

        route.append(target)
        cdist += dists[cnode, target]
        routes[route_id] = Route(id=route_id, problem=p, customers=route, reward=creward, length=cdist)
        route_id += 1
    
    return Solution(problem=p, routes=routes, c_to_r=c_to_r)