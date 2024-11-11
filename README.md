
# Migration Analysis

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
