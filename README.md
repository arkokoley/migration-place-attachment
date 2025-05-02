# Migration Place Attachment Project

This project performs a migration analysis using a NetLogo model to simulate migration patterns based on place attachment levels. The analysis includes setting up experiments, running simulations, generating visualizations, and creating summary reports.

## Prerequisites

* Python 3.x
* NetLogo 6.x
* Required Python packages: `numpy`, `pandas`, `seaborn`, `matplotlib`, `pyNetLogo`

## Installation

1. **Clone the repository** :

```shell
   git clone https://github.com/gabemgem/migration-place-attachment
```

1. **Install Python dependencies** :

```shell
   pip install numpy pandas seaborn matplotlib pyNetLogo
```

1. **Download and install NetLogo** :

* Download NetLogo from [NetLogo&#39;s official website](vscode-file://vscode-app/c:/Users/arkok/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html).
* Install NetLogo following the instructions for your operating system.

## Files

* [migration-analysis.py](vscode-file://vscode-app/c:/Users/arkok/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Main Python script to run the migration analysis.
* [migration_place_attachment_model.nlogo](vscode-file://vscode-app/c:/Users/arkok/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): NetLogo model file defining the migration simulation.
* [README.md](vscode-file://vscode-app/c:/Users/arkok/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): This readme file.

## Usage

1. **Prepare the NetLogo model** :

* Ensure the [migration_place_attachment_model.nlogo](vscode-file://vscode-app/c:/Users/arkok/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) file is in the same directory as the [migration-analysis.py](vscode-file://vscode-app/c:/Users/arkok/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) script.

1. **Run the analysis** :

   **python** **migration-analysis.py**

1. **Output** :

* The script will generate visualizations and save them as PNG files.
* The raw data and summary report will be saved as CSV files.

## Example

Here is an example of how to run the migration analysis:

**python** **migration-analysis.py**

The script will perform the following steps:

1. Load the NetLogo model.
2. Set up and run experiments with different attachment levels.
3. Generate visualizations of the results.
4. Create a summary report with correlations and summary statistics.
5. Save the results to files.

## Code Overview

### [migration-analysis.py](vscode-file://vscode-app/c:/Users/arkok/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)

This script contains the `MigrationAnalyzer` class, which handles the entire analysis process:

* **Initialization** :

```python
  analyzer = MigrationAnalyzer(model_path)
```

* **Setup and run experiments** :

```python
  analyzer.setup_experiment(attachment_levels=np.linspace(0, 1, 10))  
  analyzer.run_experiments(n_iterations=10)
```

* **Generate visualizations and report** :

```
  fig = analyzer.visualize_results()
  report = analyzer.generate_report()
```

* **Save results** :

```
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  fig.savefig(f'migration_patterns_{timestamp}.png')
  report['raw_data'].to_csv(f'migration_results_{timestamp}.csv')
```

* **Cleanup** :

```
  analyzer.cleanup()
```

### [migration_place_attachment_model.nlogo](vscode-file://vscode-app/c:/Users/arkok/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)

This NetLogo model defines the simulation environment, including agents (people and places), their properties, and behaviors. The model includes procedures for setting up the simulation, running it, and reporting results.

## Troubleshooting

* Ensure that NetLogo is correctly installed and the path to the NetLogo executable is correctly set in your environment.
* Verify that all required Python packages are installed.
* Check for any error messages in the console and address them accordingly.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

* NetLogo: [NetLogo](vscode-file://vscode-app/c:/Users/arkok/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)
* Python: [Python](vscode-file://vscode-app/c:/Users/arkok/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)

For any questions or issues, please open an issue on the GitHub repository.

## Analysis Scripts

### Shock Interaction Analysis (`analyze_shock_interactions.py`)

This script analyzes aggregated results (specifically those containing RCT metrics like `avg_satisfaction_shocked`) to visualize how the effect of attachment shocks **interacts** with other parameters, particularly `mean_attachment`.

**Functionality:**

1.  Reads an aggregated results CSV file.
2.  Calculates the difference in outcomes (satisfaction and moves) between agents who experienced shocks and those who didn't (`satisfaction_diff`, `moves_diff`). This represents the Average Treatment Effect (ATE) of the shock.
3.  **Automatically identifies potential parameter columns** in the CSV (excluding known metric/shock/attachment columns).
4.  Groups the data based on unique combinations of these identified parameters.
5.  For each parameter combination (group), it generates and saves a **faceted plot** (`shock_interaction_group_*.png`). This plot shows:
    *   The calculated ATE (`satisfaction_diff` and `moves_diff`) on the y-axis.
    *   The `mean_attachment` level on the x-axis.
    *   Points colored by the `individual_shock_probability`.
    *   Separate columns for the effect on 'Satisfaction' and 'Moves'.
    *   This visualization helps understand how the impact of shocks (ATE) changes depending on the baseline attachment level and the probability of the shock itself, within specific experimental conditions defined by the other parameters.

**Prerequisites:**

*   Python 3.x
*   Required Python libraries: `pandas`, `matplotlib`, `seaborn`
    ```bash
    pip install pandas matplotlib seaborn
    ```

**Input Data:**

*   A CSV file containing aggregated simulation results. This file **must** include:
    *   The shock probability column (e.g., `individual_shock_probability`).
    *   The mean attachment column (e.g., `mean_attachment`).
    *   The RCT outcome metric columns (`avg_satisfaction_shocked`, `avg_satisfaction_not_shocked`, `avg_moves_shocked`, `avg_moves_not_shocked`).
    *   Columns for all other parameters that define the different simulation runs.

**Configuration:**

Before running, **edit `analyze_shock_interactions.py`** and configure the following variables:

*   `INPUT_CSV`: **(Required)** Set the path to your specific aggregated results CSV file.
*   `OUTPUT_DIR`: (Optional) Change the directory name for saved plots (default: `shock_interaction_graphs_v2`).
*   `SHOCK_PROB_COL`, `ATTACHMENT_COL`, `SAT_SHOCKED_COL`, etc.: (Verify) Ensure these match the exact column names in your CSV.
*   `NON_PARAMETER_COLS`: (Optional) Review this list. Add any other column names from your CSV that should *not* be treated as parameters for grouping (e.g., run IDs, iteration counters, specific metric columns not used for grouping).

**Running the Script:**

1.  Ensure your results CSV file exists and the path in `INPUT_CSV` is correct.
2.  Verify the column name configurations in the script match your data file.
3.  Run from the terminal:
    ```bash
    python analyze_shock_interactions.py
    ```

**Output:**

*   Log messages printed to the console, including the list of automatically identified parameter columns.
*   A directory named `shock_interaction_graphs_v2` (or your configured name).
*   Inside the output directory:
    *   One PNG image file (`shock_interaction_group_*.png`) for each unique combination of the automatically identified parameters, where `*` is a numerical index. Each plot shows the interaction effect of shocks and attachment on satisfaction and moves for that specific parameter setting.
    *   A CSV file named `group_parameter_mapping.csv` that links each `group_index` (used in the PNG filenames) to the specific parameter values defining that group.
