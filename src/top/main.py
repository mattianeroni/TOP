import os
import sys
import logging
import polars as pl 
from pathlib import Path

from top.problem import read_instance
from top.heuristic import savings_heuristic, constructive_heuristic
from top.metaheuristic import multi_start_metaheuristic, local_search_metaheuritic
from top.plot import plot_solution

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    current_dir = Path(__file__).parent.parent.parent
    instances_dir = current_dir / "instances"

    results = {
        "instance" : [],
        "constructive_determistic_dreward": [],
        "constructive_stochastic_dreward": [],
        "constructive_deterministic_sreward": [],
        "constructive_stochastic_sreward": [],
        "savings_determistic_dreward": [],
        "savings_stochastic_dreward": [],
        "savings_deterministic_sreward": [],
        "savings_stochastic_sreward": [],
        "local_determistic_dreward": [],
        "local_stochastic_dreward": [],
        "local_deterministic_sreward": [],
        "local_stochastic_sreward": [],
    }

    df = pl.read_csv(instances_dir / "results.csv")
    files = set(df["Instance"].to_list())
    
    for filename in sorted(instances_dir.iterdir()):
        if filename.suffix == ".txt" and str(filename).split("/")[-1] in files:
            logger.info(f"Processing: {filename.name}")
            p = read_instance(str(filename))
            instance = p.instance.split("/")[-1]
            results["instance"].append(instance)
            
            # Multi-Start Constructive Meta-Heuristic 
            sol, stochastic_sol = None, None
            for i in range(5):
                isol, istochastic_sol = multi_start_metaheuristic(
                    p, 
                    heuristic=constructive_heuristic,
                    max_iter = 3000, 
                    enlarge_search_iter = 30, 
                    beta_start = 0.99, 
                    beta_step = 0.05, 
                    min_beta = 0.1,
                    n_elites = 5,
                    short_simulation_n_iter = 1000,
                    long_simulation_n_iter = 50_000,
                )
                if sol is None or isol.reward > sol.reward:
                    sol, stochastic_sol = isol, istochastic_sol
            logger.info("Multi-start Constructive Metaheuristic")
            logger.info(f"Number of routes: {len(sol.routes)}")
            logger.info(f"Total reward: {sol.reward}")
            results["constructive_determistic_dreward"].append(sol.reward)
            results["constructive_stochastic_dreward"].append(stochastic_sol.reward)
            results["constructive_deterministic_sreward"].append(sol.stochastic_reward(n_iter=50_000))
            results["constructive_stochastic_sreward"].append(stochastic_sol.stochastic_reward(n_iter=50_000))
            #plot_solution(sol)

            # Multi-Start Savings Meta-Heuristic 
            sol, stochastic_sol = None, None
            for i in range(5):
                isol, istochastic_sol = multi_start_metaheuristic(
                    p, 
                    heuristic=savings_heuristic,
                    max_iter = 3000, 
                    enlarge_search_iter = 30, 
                    beta_start = 0.99, 
                    beta_step = 0.05, 
                    min_beta = 0.1,
                    n_elites = 5,
                    short_simulation_n_iter = 1000,
                    long_simulation_n_iter = 50_000,
                )
                if sol is None or isol.reward > sol.reward:
                    sol, stochastic_sol = isol, istochastic_sol
            logger.info("Multi-start Savings Metaheuristic")
            logger.info(f"Number of routes: {len(sol.routes)}")
            logger.info(f"Total reward: {sol.reward}")
            results["savings_determistic_dreward"].append(sol.reward)
            results["savings_stochastic_dreward"].append(stochastic_sol.reward)
            results["savings_deterministic_sreward"].append(sol.stochastic_reward(n_iter=50_000))
            results["savings_stochastic_sreward"].append(stochastic_sol.stochastic_reward(n_iter=50_000))
            #plot_solution(sol)
            
            # Local Search Metaheuristic
            sol, stochastic_sol = None, None
            for i in range(5):
                isol, istochastic_sol = local_search_metaheuritic(
                    p, 
                    max_iter = 3000, 
                    n_elites = 5,
                    short_simulation_n_iter = 1000,
                    long_simulation_n_iter = 50_000,
                )
                if sol is None or isol.reward > sol.reward:
                    sol, stochastic_sol = isol, istochastic_sol
            logger.info("Local Search Metaheuristic")
            logger.info(f"Number of routes: {len(sol.routes)}")
            logger.info(f"Total reward: {sol.reward}")
            results["local_determistic_dreward"].append(sol.reward)
            results["local_stochastic_dreward"].append(stochastic_sol.reward)
            results["local_deterministic_sreward"].append(sol.stochastic_reward(n_iter=50_000))
            results["local_stochastic_sreward"].append(stochastic_sol.stochastic_reward(n_iter=50_000))

    df = (
        pl.read_csv(instances_dir / "results.csv")
        .join(pl.DataFrame(results), left_on="Instance", right_on="instance", how="inner")
    )
    df.write_csv(instances_dir / "results_report.csv")


if __name__ == "__main__":
    main()
