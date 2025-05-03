# Migration Place Attachment Model Specification
Version: 1.4.0
Last Updated: 2025-04-19

## Project Overview

This project simulates and analyzes human migration patterns using a NetLogo agent-based model orchestrated by Python scripts. It specifically investigates how psychological place attachment, alongside utility maximization (based on factors like quality of life and income) and social network interactions, influences agents' decisions to migrate or stay. The analysis involves running different types of experiments (parameter sweeps, complexity analysis, sensitivity analysis) using a structured, object-oriented approach. The model includes functionality for simulating shocks that directly reduce place attachment, occurring either at the **individual level** (affecting a single agent's attachment to their current location) or at the **place level** (affecting the attachment of all agents currently at a specific location).

## Project Components

### 1. NetLogo Model (`migration_place_attachment_model.nlogo`)
- **Core Simulation Engine:** Defines the agent-based simulation environment.
- **Agents:**
    - `people`: Mobile agents with attributes like `attachment_level`, `attachment` (table storing attachment to specific places), `preferences`, `observations` (utility memory), `layer_observations` (memory of place attributes), `pInitiate` (interaction probability), `decisionType` (`consumat` or `maximize`), `satisfied?`, `uncertain?`, `ever_shocked?` (individual shock flag), `moves_count`.
    - `places`: Stationary agents representing locations, generated procedurally with hierarchical administrative levels. Attributes include `place_mean_q` (quality of life), `place_mean_i` (income potential), `place_a` (capacity).
- **Environment:** A spatial grid where `patches` belong to specific `places`. `setup-map` procedure generates the landscape.
- **Agent Behavior:**
    - **Place Attachment:** Dynamically updated (`update-place-attachment`) based on `attachment_form_rate` and `attachment_decay_rate`. Can also be suddenly reduced by:
        - **Individual Shocks:** Based on `individual_shock_probability` and `attachment_shock_magnitude`, affecting a single agent's attachment to their current place. The `ever_shocked?` flag tracks if an agent has experienced this.
        - **Place Shocks:** Based on `place_shock_probability` and `place_shock_magnitude`. If a `place` experiences a shock, the attachment of *all* `people` currently located at that place is reduced. (Requires new procedures like `trigger-place-shocks` and modifications to `update-place-attachment` or a new shock application procedure).
    - **Information Gathering:** Agents collect utility (`get-utility`, `add-observation`) and place attribute (`add-layer-observation`) information about their current location.
    - **Social Interaction (`interact`):** Agents share information (place observations) with network neighbors (`link-neighbors`) based on `pInitiate` and `max_nInformation` .
    - **Decision Making (`decide`):** Agents choose a `target_place` based on their `decisionType`:
        - `maximize`: Selects the place with the highest remembered mean utility (`decide-maximize`).
        - `consumat`: Decision based on `satisfied?` and `uncertain?` states, potentially involving social comparison or deliberate moves (`decide-consumat`, `decide-deliberate-move`, `define-satisfaction-certainty`). `pConsumat` influences the likelihood of using this model, potentially switching based on `switchInterval`.
    - **Movement (`move`):** Agents move towards their `target_place`.
- **Metrics Export:** Provides reporters (`get-experiment-metrics`, `get-agent-data`, `get-avg-improvement`, `get-metrics-by-shock-status`, potentially new reporters for place shock effects) to output simulation results for Python analysis. Tracks metrics like `total_moves`, `avg_move_distance`, `satisfaction_levels`, `utility_changes`, `far_from_home`, `at_home`.

### 2. Base Experiment Class (`base_experiment.py`)
- **Core Experiment Runner:** Defines an abstract base class (`BaseExperiment`) using the Template Method pattern.
- **Responsibilities:**
    - **NetLogo Management:** Handles `pynetlogo.NetLogoLink` creation, command/report execution, and cleanup within worker processes.
    - **Parallel Execution:** Manages `multiprocessing.Pool` for running simulations in parallel. Includes worker initialization (`_initialize_worker`) and the worker task function (`_run_single_simulation_worker`).
    - **Template Methods:**
        - `run_experiments`: Orchestrates the overall experiment execution flow (parameter generation, parallel runs, result aggregation, calling analysis/visualization).
        - Abstract methods (`_define_parameter_sets`, `_get_metrics_to_collect`, `analyze_results`, `visualize_results`) must be implemented by subclasses.
    - **Result Aggregation:** Collects results from worker processes and averages them over iterations for each parameter set.
    - **Output Handling:** Creates output directories and saves aggregated results to CSV.

### 3. Parameter Sweep Experiment (`migration_analysis.py`)
- **Inherits `BaseExperiment`:** Implements the template methods for a full parameter sweep.
- **`ParameterSweepExperiment` Class:**
    - `_define_parameter_sets`: Generates all combinations from defined parameter ranges (`param_ranges`).
    - `_get_metrics_to_collect`: Specifies NetLogo reporters for relevant metrics.
    - `analyze_results`: Performs Random Forest analysis (`sklearn`) and exports data in `.mat` format for MATLAB (`scipy.io`).
    - `visualize_results`: Generates plots (`matplotlib`, `seaborn`) showing parameter effects (e.g., place attachment vs. moves, RCT shock effects).
- **Output:** Saves aggregated results, `.mat` files, RF importance plots, and parameter effect visualizations.

### 4. Complexity Experiment (`ex1.py`)
- **Inherits `BaseExperiment`:** Implements the template methods for analyzing landscape complexity effects.
- **`ComplexityAnalysisExperiment` Class:**
    - `_define_parameter_sets`: Generates parameter combinations focusing on landscape parameters (`admin_levels`, `mean_parts`, `sd_parts`) and `mean_attachment`, using `ExperimentConfig`.
    - `_get_metrics_to_collect`: Specifies necessary NetLogo reporters.
    - `analyze_results`: Calculates correlations between `mean_attachment` and migration metrics, grouped by complexity parameters. Computes composite effect size and consistency.
    - `visualize_results`: Generates heatmaps and scatter plots showing how complexity and shocks mediate the effect of place attachment.
- **Output:** Exports analysis results (correlations, effect sizes) to CSV and generates complexity effect plots.

### 5. OFAT Sensitivity Analysis Experiment (`ofat_experiment.py`)
- **Inherits `BaseExperiment`:** Implements the template methods for One-Factor-At-a-Time sensitivity analysis.
- **`OFATExperiment` Class:**
    - `_define_parameter_sets`: Generates parameter sets including a baseline and variations where only one parameter is changed at a time.
    - `_get_metrics_to_collect`: Specifies necessary NetLogo reporters.
    - `analyze_results`: Calculates the absolute and percentage change in metrics for each variation compared to the baseline run.
    - `visualize_results`: Generates bar plots showing the sensitivity of output metrics to changes in individual parameters.
- **Output:** Exports sensitivity analysis results (changes from baseline) to CSV and generates sensitivity bar plots.

### 6. Random Forest Analysis (`random_forrest.m`)
- **Advanced Statistical Analysis:** Intended for use in MATLAB, likely leveraging its statistical toolboxes.
- **Input:** Takes `.mat` files generated by `ParameterSweepExperiment` (`migration_analysis.py`).
- **Analysis:** Performs Random Forest analysis to determine variable importance.
- **Visualization:** Generates plots related to the Random Forest results.

### 7. Shock Visualization Module (`shock_visualization.py`)
- **Purpose:** Standalone visualization tool for shock effects analysis
- **Key Features:**
  - Framework-agnostic: Works with any data source that provides correctly formatted input
  - Supports both JSON and CSV input formats
  - Generates comparative visualizations of individual vs. place shock effects
  - Includes error bars and statistical summaries
- **Input Requirements:**
  - Required fields: shock_type, mean_attachment, avg_satisfaction_shocked, avg_satisfaction_not_shocked, avg_moves_shocked, avg_moves_not_shocked
  - Optional metadata: shock probabilities, magnitudes, dates, sample sizes
- **Output:**
  - Two-panel figure comparing satisfaction and movement effects
  - Optional automatic file saving
  - Returnable figure object for further customization

## Analysis Pipeline

1.  **Instantiate Experiment:** Create an instance of the desired experiment class (`ParameterSweepExperiment`, `ComplexityAnalysisExperiment`, `OFATExperiment`), providing the model path.
2.  **Configure Experiment:** The experiment class's `__init__` and `_define_parameter_sets` methods define the specific parameters and ranges/variations to be tested.
3.  **Run Simulations:** Call the `run_experiments` method on the experiment instance.
    - `BaseExperiment` handles generating the full list of simulation runs (including iterations).
    - `BaseExperiment` manages the `multiprocessing.Pool` to execute `_run_single_simulation_worker` for each run in parallel.
    - Workers set parameters, run `setup` and `go` in NetLogo, and report metrics defined by `_get_metrics_to_collect`.
4.  **Aggregate Results:** `BaseExperiment.run_experiments` collects results from workers, groups them by unique base parameter sets, and calculates the mean (or other statistics) across iterations. Saves the aggregated DataFrame.
5.  **Analyze Results:** `BaseExperiment.run_experiments` calls the subclass's `analyze_results` method (e.g., RF analysis, correlation analysis, sensitivity calculation). Analysis artifacts are saved.
6.  **Generate Visualizations:** `BaseExperiment.run_experiments` calls the subclass's `visualize_results` method to create plots specific to the experiment type. Plots are saved.
7.  **Export for External Analysis:** `ParameterSweepExperiment` exports data in `.mat` format for MATLAB Random Forest analysis.
8.  **External Analysis (Optional):** Run `random_forrest.m` in MATLAB using the exported `.mat` file.

### Shock Analysis Data Format

1. **JSON Structure:**
```json
{
    "experiments": [
        {
            "shock_type": "individual|place",
            "mean_attachment": <float>,
            "avg_satisfaction_shocked": <float>,
            "avg_satisfaction_not_shocked": <float>,
            "avg_moves_shocked": <float>,
            "avg_moves_not_shocked": <float>,
            "n_shocked": <int>,
            "n_not_shocked": <int>
        }
    ],
    "metadata": {
        "date": "YYYY-MM-DD",
        "shock_probability": {
            "individual": <float>,
            "place": <float>
        },
        "shock_magnitude": {
            "individual": <float>,
            "place": <float>
        }
    }
}
```

2. **CSV Alternative:**
   - Required columns match JSON experiment fields
   - One row per experimental condition
   - Headers must exactly match field names
   - Metadata can be included as additional columns

3. **Field Descriptions:**
   - `shock_type`: Either "individual" or "place"
   - `mean_attachment`: Base attachment level (0-1)
   - `avg_satisfaction_shocked`: Mean satisfaction of shocked agents
   - `avg_satisfaction_not_shocked`: Mean satisfaction of non-shocked agents
   - `avg_moves_shocked`: Mean moves per shocked agent
   - `avg_moves_not_shocked`: Mean moves per non-shocked agent
   - `n_shocked`: Number of agents in shocked group
   - `n_not_shocked`: Number of agents in control group

## Technical Details

### Parameters

1.  **Place Attachment:**
    - `mean_attachment`, `sd_attachment`: Distribution parameters for agent `attachment_level`.
    - `attachment_form_rate`, `attachment_decay_rate`: Rates for dynamic attachment changes.
    - `initial_attachment`: Starting attachment value for the home location.
2.  **Network & Interaction:**
    - `mean_pInitiate`, `sd_pInitiate`: Distribution for agent probability of initiating interaction.
    - `maxInitiateInteractions`: Max interactions per tick.
    - `max_nInformation`: Max pieces of info shared per interaction.
    - `p_link_state`, `p_link_network`, `p_link_random`: Probabilities governing network formation.
    - `ave_links_person`: Target average number of links per agent.
3.  **Decision Making:**
    - `pConsumat`: Probability of using the "consumat" decision model.
    - `switchInterval`: Ticks between potential switches in `decisionType`.
    - `mean_pref`, `sd_pref`: Distribution for agent preferences over utility components (QoL, income).
    - `max_memory`: Maximum number of observations stored per place.
4.  **Landscape & Population:**
    - `admin_levels`: Number of hierarchical levels in the landscape.
    - `mean_parts`, `sd_parts`: Parameters controlling subdivision of places.
    - `initial_population`: Number of agents.
    - `num_classes`: Number of agent classes (potentially affecting income).
5.  **Individual Shocks:**
    - `individual_shock_probability` (0-0.1): Probability per agent per tick of experiencing an attachment shock.
    - `attachment_shock_magnitude` (0-1): The value to which an agent's attachment to their current place drops when an individual shock occurs.
6.  **Place Shocks:**
    - `place_shock_probability` (0-0.1): Probability per *place* per tick of experiencing an attachment shock event.
    - `place_shock_magnitude` (0-1): The value to which the attachment of *all agents currently at the shocked place* drops.

### Metrics Tracked (Examples from `_get_metrics_to_collect`)

1.  **Movement:**
    - `total_moves`: Total number of moves per simulation.
    - `avg_move_distance`: Average distance of moves made.
    - `FractionMovesHome`: Fraction of agents currently at their home location.
    - `TimeStepsAwayFromHome`: Average distance from home (proxy for time away).
2.  **Agent State & Utility:**
    - `avg_satisfaction`: Average satisfaction level across agents.
    - `avg_utility_change`: Average change in utility experienced after decisions.
    - `num_uncertain`: Number of agents currently in the `uncertain?` state.
3.  **RCT Metrics (Individual Shock Effects):**
    - `avg_satisfaction_shocked`, `avg_satisfaction_not_shocked`: Average satisfaction for agents who have/haven't experienced an *individual* shock.
    - `avg_moves_shocked`, `avg_moves_not_shocked`: Average move count for agents who have/haven't experienced an *individual* shock.
4.  **Place Shock Metrics (Potential New Metrics):**
    - `num_place_shocks_triggered`: Total count of place shock events.
    - `avg_satisfaction_at_shocked_places`: Average satisfaction of agents currently at a place that has experienced a shock (potentially averaged over time or agents).
    - `avg_moves_from_shocked_places`: Average moves originating from places after they experienced a shock.

### Working Solutions

1. Memory Management:
   ```python
   # BaseExperiment uses a context manager for multiprocessing.Pool
   with Pool(...) as pool:
       # ... map tasks ...
   # Pool automatically cleans up worker processes on exit.
   # Individual NetLogo instances are managed by _initialize_worker and closed implicitly when worker process terminates.
   ```

2. Parameter Space Reduction:
   ```python
   # Example from ParameterSweepExperiment
   'mean_attachment': np.linspace(0, 1, 3),  # Reduced from 5 to 3
   'sd_attachment': np.linspace(0.1, 0.5, 2),  # Reduced from 3 to 2
   ```

3. Error Handling:
   ```python
   # Example from _run_single_simulation_worker in base_experiment.py
   try:
       # ... NetLogo commands/reports ...
       value = _worker_netlogo_link.report(reporter)
       # ... process value ...
   except Exception as report_error:
       print(f"Worker {os.getpid()} error reporting '{reporter}'...")
       results[key] = np.nan # Assign NaN on error
   # Overall worker errors are caught and return {'success': False, ...}
   ```

### Known Issues

1.  **NetLogo Communication:**
    - Handling potential `None` or empty list returns from NetLogo reporters (partially addressed by `ifelse-value` in reporters and NaN checks in Python).
    - Ensuring correct data types (e.g., converting NetLogo numbers/lists to Python floats/lists).
    - Robust error handling for NetLogo commands/reports within workers.
2.  **Data Handling:**
    - Explicit `NaN` checking and handling for metrics reported from NetLogo or calculated in Python is crucial during aggregation and analysis.
    - Ensuring consistent data types in Pandas DataFrames.
3.  **Performance:**
    - Large parameter sweeps/many iterations can be time-consuming, even with parallelization.
    - Memory usage for storing raw results (if enabled) or large aggregated DataFrames.
4.  **Parallel Initialization:** Occasional errors during worker initialization if NetLogo instances fail to start correctly (mitigated by checks in `_initialize_worker`).

## Development Notes

### Testing Protocol

1. Experiment Configuration Testing:
   - Verify parameter combinations generated by `_define_parameter_sets` for each experiment type.
   - Check metric collection reporters in `_get_metrics_to_collect`.
2. Execution Testing:
   - Run small tests (few iterations, low ticks) for each experiment type.
   - Monitor RAM usage during parallel runs.
   - Verify that output directories and files are created correctly.
3. Data Validation:
   - Check aggregated results for NaNs or unexpected values.
   - Validate analysis outputs (e.g., correlations, sensitivity changes, RF importances).
   - Ensure plots are generated without errors and reflect the data.

### Best Practices

1. Implement robust error handling within worker functions (`_run_single_simulation_worker`) and analysis/visualization methods.
2. Use the context manager (`with Pool(...)`) for parallel processing to ensure proper cleanup.
3. Validate metrics (check for NaNs, reasonable ranges) before analysis and visualization.
4. Keep experiment-specific logic within the respective subclasses (`ParameterSweepExperiment`, `ComplexityAnalysisExperiment`, `OFATExperiment`). Modify `BaseExperiment` only for core execution changes.
5. Use `gui=True` in `BaseExperiment` constructor only for debugging single runs, not for large parallel experiments.

## Environment Setup

```requirements
python>=3.9
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
seaborn>=0.11
pynetlogo>=0.5
scikit-learn>=1.0
scipy>=1.7 # For saving .mat files
# networkx # Not currently used
```

### NetLogo Requirements
- NetLogo 6.4.0
- Minimum 4GB RAM recommended (more for larger populations/complex landscapes)
- Java 8 or higher (usually bundled with NetLogo)
- **Environment Variable:** `NETLOGO_HOME` should point to the NetLogo installation directory (e.g., `C:\Program Files\NetLogo 6.4.0`).

## Future Improvements

1. Optimization:
   - Profile worker execution time and memory usage.
   - Explore alternative parallelization backends if `multiprocessing` becomes limiting.
2. Analysis:
   - Implement more sophisticated sensitivity analysis methods (e.g., Sobol indices).
   - Add time-series analysis of metrics within runs.
   - **Compare the differential impacts of individual vs. place-level shocks on migration patterns, satisfaction, and network effects.**
3. Model Features:
   - Refine the `consumat` decision model logic (e.g., `decide-look-to-peers`, `decide-social-comparison`).
   - Incorporate more sophisticated network dynamics or evolution.
   - Add event-based triggers or external shocks (e.g., place-based utility shocks).
   - Refine individual and place shock mechanisms (e.g., variable magnitude, recovery time, correlation between shock types).
4. Workflow:
   - Add configuration files (e.g., YAML) for defining experiment parameters instead of hardcoding them.
   - Implement more comprehensive logging.

## For LLMs

When working with this project:

1.  **Code Structure:**
    - `migration_place_attachment_model.nlogo`: The core simulation logic. Focus on agent behaviors (`go`, `decide`, `interact`, `move`, `update-place-attachment`), setup (`setup`, `setup-map`, `setup-agents`, `setup-network`), and **shock mechanisms (individual and place-level)**. Pay attention to reporter procedures used by Python (`get-experiment-metrics`, etc.).
    - `base_experiment.py`: Contains the `BaseExperiment` class handling parallel execution (`run_experiments`, `_run_single_simulation_worker`) and NetLogo interaction within workers. Defines the template methods.
    - `migration_analysis.py`: Implements `ParameterSweepExperiment` for full sweeps, including RF analysis (`_analyze_random_forest`) and `.mat` export (`_export_for_matlab`).
    - `ex1.py`: Implements `ComplexityAnalysisExperiment` focusing on landscape complexity effects via correlation analysis (`analyze_results`).
    - `ofat_experiment.py`: Implements `OFATExperiment` for one-factor-at-a-time sensitivity analysis (`analyze_results`).
    - `random_forrest.m`: Assumed to perform advanced RF analysis in MATLAB using `.mat` files.
2.  **Common Issues & Handling:**
    - **NetLogo Output:** Be prepared for NetLogo reporters to return `None`, empty lists, or potentially `NaN`. Python code in `_run_single_simulation_worker` and analysis methods includes checks.
    - **Data Types:** Ensure conversions between NetLogo's types and Python/NumPy/Pandas types are handled correctly (e.g., using `float()`, `np.nanmean`).
    - **Parameter Mismatches:** Ensure parameter names used in Python experiment definitions match the `globals` or `*-own` variables in NetLogo (including new place shock parameters like `place_shock_probability`).
    - **File Paths & Environment:** Verify paths to the NetLogo model file and that the `NETLOGO_HOME` environment variable is set correctly.
3.  **Modification Guidelines:**
    - **NetLogo:**
        - **Adding Place Shocks:** Implement procedures to trigger place shocks (e.g., `ask places [ if random-float 1.0 < place_shock_probability [ trigger-shock ] ]`) and apply the effect to agents at that place (e.g., `ask people-here [ set-attachment-to current-place place_shock_magnitude ]`). Add new `globals` (`place_shock_probability`, `place_shock_magnitude`) and corresponding sliders.
        - **Adding Metrics:** Create new `globals`, update `setup-experiment-metrics`, `reset-metrics`, `compute-metrics`, and relevant reporters (e.g., for place shock impacts).
        - **Changing Behavior:** Modify agent procedures.
    - **Python (Core Execution):** Modify `BaseExperiment` or worker functions (`_initialize_worker`, `_run_single_simulation_worker`) in `base_experiment.py`.
    - **Python (Specific Experiment):** Modify the relevant subclass (`ParameterSweepExperiment`, `ComplexityAnalysisExperiment`, `OFATExperiment`).
        - To change parameters: Modify parameter definitions to include `place_shock_probability` and `place_shock_magnitude`.
        - To change metrics: Modify `_get_metrics_to_collect` to include new place shock reporters.
        - To change analysis/plots: Modify `analyze_results` or `visualize_results` to incorporate analysis comparing individual vs. place shocks or analyzing place shock effects.
    - **Consistency:** Maintain consistency between NetLogo variable names and Python parameter dictionaries/DataFrame columns.
4.  **Tips:**
    - Test changes with small parameter sets (`n_iterations=1`, `num_ticks=10`) first.
    - Validate metric calculations and check for `NaN` values in output DataFrames.
    - Use `gui=True` in the `BaseExperiment` constructor for debugging single, non-parallel runs.
    - Consider experiments where only one type of shock is active (`individual_shock_probability = 0` or `place_shock_probability = 0`) to isolate effects before comparing combined scenarios.

[End of Specification]
