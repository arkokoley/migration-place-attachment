import abc
import os
import itertools
import pandas as pd
import numpy as np
from multiprocessing import Pool
from pynetlogo import NetLogoLink
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

# Global variable for worker processes
_worker_netlogo_link = None

def _initialize_worker(model_path: str, gui_mode: bool):
    """Initialize a NetLogoLink instance for each worker process."""
    global _worker_netlogo_link
    print(f"Initializing worker process {os.getpid()}...")
    try:
        _worker_netlogo_link = NetLogoLink(
            gui=gui_mode,
        )
        _worker_netlogo_link.load_model(model_path)
        print(f"Worker {os.getpid()} initialized.")
    except Exception as e:
        print(f"Error initializing worker {os.getpid()}: {e}")
        _worker_netlogo_link = None # Ensure it's None if initialization fails
    finally:
        # Ensure cleanup happens even if initialization fails partially
        if _worker_netlogo_link is None:
             print(f"Worker {os.getpid()} failed initialization.")


def _run_single_simulation_worker(task_info: Tuple[Dict[str, Any], int, Dict[str, str]]) -> Dict[str, Any]:
    """Wrapper function to run a single simulation in a worker process."""
    global _worker_netlogo_link
    params, ticks, metrics_to_collect = task_info

    if _worker_netlogo_link is None:
        print(f"NetLogoLink not initialized in worker {os.getpid()}. Skipping.")
        return {'success': False, **params} # Return structure indicating failure

    try:
        # print(f"Worker {os.getpid()} running simulation with params: {params}")
        results = {}
        results.update(params) # Start with input parameters

        # Set parameters
        for param, value in params.items():
            # Skip non-model parameters like 'iteration'
            if param == 'iteration': continue
            # Handle string parameters that need quotes in NetLogo
            if isinstance(value, str):
                _worker_netlogo_link.command(f'set {param} "{value}"')
            else:
                _worker_netlogo_link.command(f'set {param} {value}')

        # Setup and run
        _worker_netlogo_link.command('setup')
        _worker_netlogo_link.command(f'repeat {ticks} [ go ]') # Use repeat for efficiency

        # Collect metrics
        for key, reporter in metrics_to_collect.items():
            try:
                value = _worker_netlogo_link.report(reporter)
                # Basic type conversion and NaN handling
                if isinstance(value, (list, pd.Series, np.ndarray)):
                     # Convert lists/arrays of numbers, handle potential NaNs within
                     try:
                         numeric_list = [float(v) if not np.isnan(v) else np.nan for v in value]
                         results[key] = numeric_list
                     except (TypeError, ValueError):
                         results[key] = value # Keep as is if not purely numeric
                elif isinstance(value, (int, float, np.number)):
                     results[key] = float(value) if not np.isnan(value) else np.nan
                else:
                     results[key] = value # Store other types directly
            except Exception as report_error:
                print(f"Worker {os.getpid()} error reporting '{reporter}' for params {params}: {report_error}")
                results[key] = np.nan # Assign NaN on error

        results['success'] = True
        # print(f"Worker {os.getpid()} finished simulation.")
        return results

    except Exception as e:
        print(f"Error in worker {os.getpid()} running simulation {params}: {e}")
        # Return structure indicating failure, include params and mark as failed
        failed_results = {k: np.nan for k in metrics_to_collect.keys()}
        return {'success': False, **params, **failed_results}
    # No finally block for cleanup here; rely on Pool termination


class BaseExperiment(abc.ABC):
    """Abstract base class for running NetLogo experiments."""

    def __init__(self, model_path: str):
        """
        Initialize BaseExperiment.

        :param model_path: Path to the .nlogo file.
        """
        self.model_path = model_path

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    @abc.abstractmethod
    def _define_parameter_sets(self) -> List[Dict[str, Any]]:
        """
        Define the list of parameter dictionaries for the experiment.
        Each dictionary represents one unique combination of base parameters.
        """
        pass

    @abc.abstractmethod
    def _get_metrics_to_collect(self) -> Dict[str, str]:
        """
        Define the metrics to collect from NetLogo.
        Returns a dictionary mapping desired result keys to NetLogo reporter strings.
        Example: {'total_moves': 'total_moves', 'avg_satisfaction': 'mean [get-satisfaction] of people'}
        """
        pass

    @abc.abstractmethod
    def analyze_results(self, results_df: pd.DataFrame, output_dir: str):
        """
        Perform experiment-specific analysis on the aggregated results.
        :param results_df: DataFrame with results averaged over iterations.
        :param output_dir: Directory to save analysis artifacts.
        """
        pass

    @abc.abstractmethod
    def visualize_results(self, results_df: pd.DataFrame, output_dir: str):
        """
        Generate experiment-specific visualizations from the aggregated results.
        :param results_df: DataFrame with results averaged over iterations.
        :param output_dir: Directory to save plots.
        """
        pass

    def run_experiments(self, n_iterations: int = 5, num_processes: int = None, num_ticks: int = 60, gui_mode: bool = False):
        """
        Run the experiments in parallel using the Template Method pattern.

        :param n_iterations: Number of times to repeat each parameter combination.
        :param num_processes: Number of worker processes. Defaults to os.cpu_count() - 1.
        :param num_ticks: Number of ticks to run each simulation.
        :param gui_mode: Whether to run NetLogo with GUI (primarily for debugging, usually False for parallel runs).
        :return: DataFrame with results averaged over iterations.
        """
        if num_processes is None:
            num_cores = os.cpu_count() or 1
            num_processes = max(1, num_cores - 1)

        base_param_sets = self._define_parameter_sets()
        metrics_to_collect = self._get_metrics_to_collect()

        # Create tasks: list of (params_with_iteration, ticks, metrics_to_collect)
        tasks = []
        for params in base_param_sets:
            for i in range(n_iterations):
                run_params = {**params, 'iteration': i}
                tasks.append((run_params, num_ticks, metrics_to_collect))

        print(f"Running {len(tasks)} simulation runs ({len(base_param_sets)} unique parameter sets x {n_iterations} iterations) across {num_processes} processes...")

        raw_results = []
        try:
            # Use context manager for the pool
            with Pool(num_processes, initializer=_initialize_worker, initargs=(self.model_path, gui_mode)) as pool:
                raw_results = pool.map(_run_single_simulation_worker, tasks)
        except Exception as pool_error:
             print(f"Error during parallel execution: {pool_error}")
             return pd.DataFrame() # Return empty DataFrame on pool error
        finally:
             print("Pool closed.") # Confirm pool closure

        print("Parallel execution finished. Processing results...")

        # Process results: Group by base parameters and average over iterations
        results_dict = {}
        param_names = list(base_param_sets[0].keys()) if base_param_sets else []

        for result in raw_results:
            if result is None or not result.get('success', False):
                print(f"Skipping failed or None result: {result}")
                continue

            # Create a unique key for the base parameter combination
            param_key_dict = {k: v for k, v in result.items() if k in param_names}
            param_key = tuple(sorted(param_key_dict.items()))

            if param_key not in results_dict:
                results_dict[param_key] = []
            results_dict[param_key].append(result)

        # Average results for each parameter combination
        averaged_results = []
        metric_keys_template = list(metrics_to_collect.keys()) # Use defined keys

        for param_key, iteration_results in results_dict.items():
            if not iteration_results: continue

            avg_result = dict(param_key) # Start with the parameters

            for k in metric_keys_template:
                values = [r.get(k) for r in iteration_results if k in r] # Get values safely

                # Filter out None before attempting analysis
                valid_values = [v for v in values if v is not None]
                if not valid_values:
                    avg_result[k] = np.nan
                    continue

                # Check type of the first valid value to determine averaging method
                first_val = valid_values[0]
                if isinstance(first_val, (list, np.ndarray)):
                     try:
                         # Average lists/arrays element-wise if numeric
                         numeric_arrays = [np.array(v) for v in valid_values if isinstance(v, (list, np.ndarray)) and all(isinstance(i, (int, float, np.number)) for i in v)]
                         if numeric_arrays:
                             # Use nanmean for robustness
                             avg_result[k] = np.nanmean(np.array(numeric_arrays, dtype=float), axis=0).tolist()
                         else:
                             avg_result[k] = [] # Or handle non-numeric lists differently
                     except (TypeError, ValueError) as e:
                         # print(f"Warning: Could not average list/array metric '{k}' for params {dict(param_key)}: {e}")
                         avg_result[k] = np.nan # Fallback
                elif isinstance(first_val, (int, float, np.number)):
                     # Average numeric scalars, ignoring NaNs
                     numeric_values = [float(v) for v in valid_values if isinstance(v, (int, float, np.number))]
                     avg_result[k] = np.nanmean(numeric_values) if numeric_values else np.nan
                else:
                     # For non-numeric types, maybe take the first one or mode? Or NaN?
                     # Taking the first valid one for now.
                     avg_result[k] = first_val

            averaged_results.append(avg_result)

        results_df = pd.DataFrame(averaged_results)

        if results_df.empty:
             print("Warning: No valid results after processing.")
             return results_df

        # --- Output and Post-processing ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'results_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Save aggregated results
        agg_results_path = os.path.join(output_dir, f'aggregated_results_{timestamp}.csv')
        results_df.to_csv(agg_results_path, index=False)
        print(f"Aggregated results saved to {agg_results_path}")

        # Save raw results (optional, can be large)
        # raw_results_path = os.path.join(output_dir, f'raw_results_{timestamp}.csv')
        # pd.DataFrame(raw_results).to_csv(raw_results_path, index=False)
        # print(f"Raw results saved to {raw_results_path}")

        # Call experiment-specific analysis and visualization
        try:
            print("Running analysis...")
            self.analyze_results(results_df, output_dir)
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()

        try:
            print("Generating visualizations...")
            self.visualize_results(results_df, output_dir)
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.close('all') # Ensure plots are closed

        print(f"Experiment run complete. Results are in {output_dir}")
        return results_df
