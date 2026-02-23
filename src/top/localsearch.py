import random 
import operator
import itertools 
from top.problem import Problem, Route, Solution 
from top.heuristic import savings_heuristic

def opt2(p: Problem, route: Route) -> Route:
    """ Run a 2-OPT optimization of the route returning the shortest found. 
    For efficiency, instead of computing the entire length of the route, since 
    distance matrix is simmetric, we just compute the deltas. """
    improved, customers, length, dists = True, list(route.customers), route.length, p.dists
    while improved:
        improved = False
        for i in range(1, len(customers) - 2):
            for j in range(i + 2, len(customers)):
                current_dist = dists[(customers[i-1], customers[i])] + dists[(customers[j-1], customers[j])]
                new_dist = dists[(customers[i-1], customers[j-1])] + dists[(customers[i], customers[j])]
                if new_dist < current_dist:
                    customers[i:j] = reversed(customers[i:j])
                    length = length - current_dist + new_dist
                    improved = True
    return Route(id=route.id, problem=p, customers=customers, reward=route.reward, length=length)

def shaking(p: Problem, solution: Solution, alpha: float, beta: float) -> Solution: 
    """ Remove a route and re-run the savings heuristic to generate a new solution. """ 
    routes, c_to_r, source, target, dists = dict(solution.routes), dict(solution.c_to_r), 0, len(p.customers) - 1, p.dists
    # Pick a random route
    route_id = random.choice(list(routes.keys()))
    route = routes.pop(route_id)
    # Destroy the route and build a new route for each uncovered customer
    for i in route.customers[1:-1]:
        c_to_r.pop(i)
    
    new_route_id = max(routes.keys()) + 1  # NOTE: to avoid duplication of routes ids
    for i, c in p.customers.items():
        if c_to_r.get(i) is not None or i == source or i == target: 
            continue
        if (t := dists[source, i] + dists[i, target]) <= p.tmax:
            new_route = Route(id=new_route_id, problem=p, customers=[source, i, target], reward=c.reward, length=t)
            c_to_r[i] = new_route_id
            routes[new_route_id] = new_route 
            new_route_id += 1
    # Build the starting solution 
    starting_solution = Solution(problem=p, routes=routes, c_to_r=c_to_r)
    return savings_heuristic(p, alpha=alpha, beta=beta, solution=starting_solution)


def insert(p: Problem, solution: Solution) -> Solution:
    """ Take a random route and try to insert unvisited customers into it optimizing the 
    sequence via 2-OPT. """
    source, target, dists, tmax = 0, len(p.customers) - 1, p.dists, p.tmax
    routes, c_to_r = dict(solution.routes), dict(solution.c_to_r)
    # Pick a random route
    route_id = random.choice(list(routes.keys()))
    # Try for each customer not yet in a route
    for id, c in p.customers.items():
        if c_to_r.get(id) is not None or id == source or id == target:
            continue
        # Insert the customer in the route and optimize the sequence via 2-OPT
        route = routes.get(route_id)
        new_customers = route.customers[:1] + [id] + route.customers[1:]
        new_route = Route(
            id=route_id, 
            problem=p, 
            customers=new_customers, 
            reward=route.reward + c.reward, 
            length=sum(dists[i, j] for i, j in zip(new_customers[:-1], new_customers[1:])),
        )
        new_route = opt2(p, new_route)
        # If route is feasible, update solution and reward
        if new_route.length <= tmax: 
            routes[route_id] = new_route 
            c_to_r[id] = route_id
            continue

    return Solution(problem=p, routes=routes, c_to_r=c_to_r)
        

def remove(p: Problem, solution: Solution) -> Solution:
    """ Remove a random node from a random route of the solution. """ 
    routes, c_to_r, dists, tmax = dict(solution.routes), dict(solution.c_to_r), p.dists, p.tmax
    # Pick a random route
    route_id = random.choice(list(routes.keys()))
    route = routes.get(route_id)
    if len(route.customers) <= 3:
        return Solution(problem=p, routes=routes, c_to_r=c_to_r)
    # Remove a random customer and optimize via 2-OPT
    new_customers = list(route.customers)
    idx = random.randrange(1, len(new_customers) - 1)
    removed_customer_id = new_customers.pop(idx)
    new_route = Route(
        id=route_id, 
        problem=p, 
        customers=new_customers, 
        reward=route.reward - p.customers[removed_customer_id].reward, 
        length=(
            route.length 
            - dists[(route.customers[idx - 1], route.customers[idx])] 
            - dists[(route.customers[idx], route.customers[idx + 1])] 
            + dists[(route.customers[idx - 1], route.customers[idx + 1])]
        ),
    )
    new_route = opt2(p, new_route)
    # If route is feasible, update solution and reward
    if new_route.length <= tmax: 
        routes[route_id] = new_route 
        c_to_r.pop(removed_customer_id)
    
    return Solution(problem=p, routes=routes, c_to_r=c_to_r)