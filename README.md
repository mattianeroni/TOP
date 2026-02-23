# Team Orienteering Problem Solver

A Python package for solving the Team Orienteering Problem (TOP) using a savings heuristic algorithm.

## Overview

The Team Orienteering Problem is a variant of the traveling salesman problem where:
- Multiple vehicles (trucks) start from a depot and arrive to a target
- Each customer has an associated reward
- Each vehicle has a maximum travel time limit
- The goal is to maximize the total reward collected while respecting time constraints


## Project Structure

```
top/
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # This file
├── requirements.txt        # Legacy requirements file
├── instances/              # Problem instances (in .txt format)
│   ├── results.csv         # Results of literature algorithms
│   ├── results_report.csv  # Results of implemented algorithms
│   ├── p4.2.a.txt
│   ├── p4.2.b.txt
│   └── ...
└── src/
    └── top/                  # Main package
        ├── __init__.py       # Package initialization
        ├── main.py           # Entry point script
        ├── simulation.py     # Methods for stochastic optimization
        ├── problem.py        # Data structures (Problem, Customer, Route, Solution)
        ├── metaheuristic.py  # Metaheuristic solvers implementation
        ├── localsearch.py    # Local search operators the metaheuristics can use
        └── heuristic.py      # Heuristic solvers implementation
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Install

```bash
pip install -e .
```

This will install the package and all required dependencies.

### Install with Development Tools

If you want to contribute or run tests, install with development dependencies:
```bash
pip install -e ".[dev]"
```

## Dependencies

- **numpy** (>=2.0.0): Numerical computing
- **matplotlib** (>=3.10.0): Visualization
- **polars** (>=1.38.0): Data manipulation
- **pytest** (dev): Testing framework

## Usage

### As a Command-Line Tool

After installation, you can run the solver from the command line:

```bash
run
```

This will process the instances in the `instances/` directory and print the results.

### As a Python Package

You can also import and use the package in your own Python code:

```python
from top import Problem, read_instance, savings_heuristic

# Read an instance file
problem = read_instance("instances/p4.2.a.txt")

# Solve using the savings heuristic
solution = savings_heuristic(problem, alpha=0.3, beta=0.99)

# Access results
print(f"Number of routes: {len(solution.routes)}")
print(f"Total reward: {solution.reward}")
print(f"Total distance: {solution.length}")
```

### Heuristic Parameters

The `savings_heuristic` function accepts:

- **problem** (Problem): The TOP instance to solve
- **alpha** (float, default=0.3): Weight for distance savings (0 to 1)
  - Higher values prioritize distance savings
  - Lower values prioritize reward collection
- **beta** (float, default=0.99): Parameter controlling the heuristic behavior

## Instance File Format

Instance files should be in plaintext format with the following structure:

```
<num_trucks> <max_time> <num_customers>
<x1> <y1> <reward1>      # Depot
<x2> <y2> <reward2>      # Customer 1
...
<xn> <yn> <rewardn>      # Sink/End depot
```

Where:
- `num_trucks`: Number of vehicles available
- `max_time`: Maximum travel time per truck
- `num_customers`: Number of customers (excluding depot and sink)
- `x, y`: Euclidean coordinates
- `reward`: Reward value for visiting that customer


## References

- Chao, I. M., Golden, B. L., & Wasil, E. A. (1996). "The team orienteering problem." 
  Proceedings of the Fourth ORSA/TIMS Logistics Conference, 88-104.
