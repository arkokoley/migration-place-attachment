import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

from base_experiment import BaseExperiment

class OFATExperiment(BaseExperiment):
    """
    Runs a One-Factor-At-a-Time (OFAT) sensitivity analysis experiment.
    Compares results when varying one parameter at a time against a baseline.
    """

    def __init__(self, model_path: str, baseline_params: Dict[str, Any],
                 parameter_variations: Dict[str, List[Any]],
                 netlogo_home: str = None, netlogo_version: str = '6.4'):
        """
        Initialize OFATExperiment.

        :param model_path: Path to the .nlogo file.
        :param baseline_params: Dictionary of baseline parameter values.
        :param parameter_variations: Dictionary mapping parameter names to lists of values to test for that parameter.
        :param netlogo_home: Path to NetLogo installation.
        :param netlogo_version: NetLogo version.
        """
        super().__init__(model_path, netlogo_home, netlogo_version)
        self.baseline_params = baseline_params
        self.parameter_variations = parameter_variations
        self.sensitivity_results = pd.DataFrame()

    def _define_parameter_sets(self) -> List[Dict[str, Any]]:
        """Generate parameter sets for OFAT analysis."""
        param_sets = []
        # Add the baseline run
        param_sets.append({'variation': 'baseline', 'param_varied': 'baseline', 'param_value': 'baseline', **self.baseline_params})

        # Add variations
        for param_name, values in self.parameter_variations.items():
            if param_name not in self.baseline_params:
                print(f"Warning: Parameter '{param_name}' in variations not found in baseline parameters. Skipping.")
                continue
            for value in values:
                # Skip if value is same as baseline
                if value == self.baseline_params[param_name]:
                    continue
                variation_params = self.baseline_params.copy()
                variation_params[param_name] = value
                param_sets.append({
                    'variation': f'{param_name}_{value}',
                    'param_varied': param_name,
                    'param_value': value,
                    **variation_params
                })
        return param_sets

    def _get_metrics_to_collect(self) -> Dict[str, str]:
        """Define metrics for OFAT analysis (likely similar to sweep)."""
        # Use a common set of important output metrics
        return {
            'total_moves': 'total_moves',
            'avg_move_distance': 'ifelse-value empty? move_distances [0] [mean move_distances]',
            'avg_satisfaction': 'ifelse-value empty? satisfaction_levels [0] [mean satisfaction_levels]',
            'avg_utility_change': 'ifelse-value empty? utility_changes [0] [mean utility_changes]',
            'FractionMovesHome': 'at_home / count people',
        }

    def analyze_results(self, results_df: pd.DataFrame, output_dir: str):
        """Analyze OFAT results by comparing variations to the baseline."""
        print("Analyzing OFAT sensitivity...")
        if results_df.empty:
            print("No results to analyze.")
            return

        baseline_run = results_df[results_df['variation'] == 'baseline']
        if baseline_run.empty:
            print("Error: Baseline run results not found.")
            return

        # Use .iloc[0] safely after checking it's not empty
        baseline_metrics = baseline_run.iloc[0]
        metric_cols = list(self._get_metrics_to_collect().keys())

        sensitivity_data = []
        variation_runs = results_df[results_df['variation'] != 'baseline']

        for _, row in variation_runs.iterrows():
            sensitivity = {'param_varied': row['param_varied'], 'param_value': row['param_value']}
            for metric in metric_cols:
                if metric in baseline_metrics and metric in row and pd.notna(baseline_metrics[metric]) and pd.notna(row[metric]):
                    baseline_val = baseline_metrics[metric]
                    variation_val = row[metric]
                    # Calculate percent change, handle baseline zero
                    if baseline_val != 0:
                        sensitivity[f'{metric}_pct_change'] = ((variation_val - baseline_val) / baseline_val) * 100
                    else:
                        sensitivity[f'{metric}_pct_change'] = np.inf if variation_val != 0 else 0 # Or NaN?
                    sensitivity[f'{metric}_abs_change'] = variation_val - baseline_val
                else:
                    sensitivity[f'{metric}_pct_change'] = np.nan
                    sensitivity[f'{metric}_abs_change'] = np.nan
            sensitivity_data.append(sensitivity)

        self.sensitivity_results = pd.DataFrame(sensitivity_data)

        # Save sensitivity results
        sensitivity_path = os.path.join(output_dir, 'ofat_sensitivity_summary.csv')
        self.sensitivity_results.to_csv(sensitivity_path, index=False)
        print(f"OFAT sensitivity summary saved to {sensitivity_path}")

    def visualize_results(self, results_df: pd.DataFrame, output_dir: str):
        """Visualize OFAT sensitivity results."""
        print("Generating OFAT sensitivity plots...")
        if self.sensitivity_results.empty:
            print("No sensitivity results to visualize. Run analyze_results first.")
             # Try running analysis if results_df is available
            if not results_df.empty:
                 print("Running analysis to generate sensitivity data...")
                 self.analyze_results(results_df, output_dir)
            if self.sensitivity_results.empty: # Check again
                 return # Still no data


        metrics_to_plot = [col.replace('_pct_change', '') for col in self.sensitivity_results.columns if '_pct_change' in col]

        num_metrics = len(metrics_to_plot)
        if num_metrics == 0:
            print("No sensitivity metrics found to plot.")
            return

        ncols = 2
        nrows = (num_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
        axes = axes.flatten()

        unique_params = self.sensitivity_results['param_varied'].unique()
        palette = sns.color_palette("husl", len(unique_params))
        param_color_map = dict(zip(unique_params, palette))

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            metric_col = f'{metric}_pct_change'
            plot_data = self.sensitivity_results.dropna(subset=[metric_col])

            if plot_data.empty:
                 ax.set_title(f'Sensitivity of {metric.replace("_"," ").title()} (No Data)')
                 continue

            # Create a combined label for the x-axis or use hue
            plot_data['label'] = plot_data['param_varied'] + '=' + plot_data['param_value'].astype(str)

            sns.barplot(data=plot_data, x='label', y=metric_col, hue='param_varied',
                        palette=param_color_map, ax=ax, dodge=False) # Use hue, don't dodge

            ax.set_title(f'Sensitivity of {metric.replace("_"," ").title()}')
            ax.set_xlabel('Parameter Variation')
            ax.set_ylabel('% Change from Baseline')
            ax.tick_params(axis='x', rotation=90)
            ax.axhline(0, color='grey', linestyle='--')
            # Improve legend
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles)) # Get unique handles/labels
            ax.legend(unique_labels.values(), unique_labels.keys(), title="Parameter Varied", bbox_to_anchor=(1.05, 1), loc='upper left')


        # Hide unused axes
        for i in range(num_metrics, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
        fig.savefig(os.path.join(output_dir, 'ofat_sensitivity_plots.png'))
        plt.close(fig)


# Example usage
if __name__ == "__main__":
    try:
        model_path = "./migration_place_attachment_model.nlogo"
        print("Running OFAT Experiment...")

        # Define baseline and variations
        baseline = {
            'mean_attachment': 0.5,
            'sd_attachment': 0.2,
            'attachment_form_rate': 0.02,
            'attachment_decay_rate': 0.02,
            'initial_attachment': 0.5,
            'mean_pInitiate': 0.3,
            'sd_pInitiate': 0.1,
            'pConsumat': 0.4,
            'individual_shock_probability': 0.0, # Baseline without shocks
            'attachment_shock_magnitude': 0.1,
            'initial_population': 500,
            'admin_levels': 2,
            'mean_parts': 3,
            'sd_parts': 1.0
            # Add all other necessary parameters
        }
        variations = {
            'mean_attachment': [0.1, 0.9],
            'attachment_form_rate': [0.01, 0.05],
            'pConsumat': [0.1, 0.8],
            'individual_shock_probability': [0.005, 0.01] # Test effect of adding shocks
        }

        # Instantiate the specific experiment
        experiment = OFATExperiment(
            model_path=model_path,
            baseline_params=baseline,
            parameter_variations=variations
        )

        # Run using the base class method
        results_df = experiment.run_experiments(
            n_iterations=5,   # More iterations for stability
            # num_processes= Use default or specify
            num_ticks=60
        )

        if not results_df.empty:
            print("\nOFAT Experiment finished. Aggregated results summary:")
            print(results_df.head())
            print(f"\nSensitivity analysis and plots saved in the 'results_YYYYMMDD_HHMMSS' directory.")
        else:
            print("\nOFAT Experiment finished, but no results were generated.")

    except Exception as e:
        print(f"\nError during OFAT experiment execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure all matplotlib figures are closed after processing
        plt.close('all')
        print("\nMain process finished.")
