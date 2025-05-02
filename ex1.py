from dataclasses import dataclass # Keep dataclass if ExperimentConfig uses it
# Remove Pool import
import numpy as np
import pandas as pd
# Remove MigrationAnalyzer import
import matplotlib.pyplot as plt
import seaborn as sns
import os # Keep for path joining
from datetime import datetime # Keep for timestamp
import itertools # Keep for combinations

# Import the new base class
from base_experiment import BaseExperiment

# Keep ExperimentConfig or adapt it
class ExperimentConfig:
    # Landscape parameters
    admin_levels: list = [2, 3] # Vary complexity
    mean_parts: list = [2, 4] # Vary complexity
    sd_parts: list = [0.5, 1.0] # Vary complexity variation

    # Agent parameters
    population_sizes: list = [500] # Keep fixed or vary
    attachment_levels: list = np.linspace(0, 1, 5) # Focus on attachment effect

    # Shock parameters
    shock_probabilities: list = [0, 0.01] # Test with and without shocks
    shock_magnitudes: list = [0.1] # Keep fixed or vary

    # Other fixed parameters (ensure these are set in NetLogo or here)
    fixed_params: dict = {
        'sd_attachment': 0.2,
        'attachment_form_rate': 0.019,
        'attachment_decay_rate': 0.02,
        'initial_attachment': 0.5,
        'mean_pInitiate': 0.3,
        'sd_pInitiate': 0.1,
        'pConsumat': 0.4,
        # Add other necessary fixed params
    }

    def get_parameter_combinations(self):
        """Generates parameter sets for complexity experiment."""
        varying_param_names = ['admin_levels', 'mean_parts', 'sd_parts',
                               'population_sizes', 'attachment_levels',
                               'shock_probabilities', 'shock_magnitudes']
        param_lists = [getattr(self, name) for name in varying_param_names]

        combinations = list(itertools.product(*param_lists))

        param_sets = []
        for combo in combinations:
            params = dict(zip(varying_param_names, combo))
            # Rename keys to match NetLogo globals
            params['initial_population'] = params.pop('population_sizes')
            params['mean_attachment'] = params.pop('attachment_levels')
            params['individual_shock_probability'] = params.pop('shock_probabilities')
            params['attachment_shock_magnitude'] = params.pop('shock_magnitudes')
            # Add fixed parameters
            params.update(self.fixed_params)
            param_sets.append(params)
        return param_sets

# Remove initialize_worker and run_single_experiment functions

class ComplexityAnalysisExperiment(BaseExperiment): # Renamed class
    """Analyzes the effect of landscape complexity on place attachment impact."""

    def __init__(self, model_path: str, netlogo_home: str = None, netlogo_version: str = '6.4'):
        super().__init__(model_path, netlogo_home, netlogo_version)
        self.config = ExperimentConfig()
        self.effects_df = pd.DataFrame() # To store correlation results

    def _define_parameter_sets(self) -> List[Dict[str, Any]]:
        """Define parameter sets using ExperimentConfig."""
        return self.config.get_parameter_combinations()

    def _get_metrics_to_collect(self) -> Dict[str, str]:
        """Define metrics needed for complexity analysis."""
        # Similar metrics as ParameterSweepExperiment, focus on those used in correlation analysis
        return {
            'total_moves': 'total_moves',
            'avg_move_distance': 'ifelse-value empty? move_distances [0] [mean move_distances]',
            'avg_satisfaction': 'ifelse-value empty? satisfaction_levels [0] [mean satisfaction_levels]',
            'avg_utility_change': 'ifelse-value empty? utility_changes [0] [mean utility_changes]',
            'num_uncertain': 'count people with [uncertain?]',
            'TimeStepsAwayFromHome': 'far_from_home',
            'FractionMovesHome': 'at_home / count people',
        }

    def analyze_results(self, results_df: pd.DataFrame, output_dir: str):
        """Analyze complexity effects: correlation with attachment."""
        print("Analyzing complexity effects (correlations)...")
        if results_df.empty:
            print("No results to analyze.")
            return

        df = results_df.copy()
        # Ensure 'mean_attachment' is numeric
        df['mean_attachment'] = pd.to_numeric(df['mean_attachment'], errors='coerce')

        # Define metrics for correlation analysis
        metrics_for_corr = [
            'total_moves', 'avg_move_distance', 'avg_satisfaction',
            'avg_utility_change', 'num_uncertain', 'TimeStepsAwayFromHome',
            'FractionMovesHome'
        ]
        metrics_available = [m for m in metrics_for_corr if m in df.columns]

        # Drop rows where conversion failed or essential metrics are missing
        df.dropna(subset=['mean_attachment'] + metrics_available, inplace=True)
        if df.empty:
             print("No valid data remaining after dropping NaNs for correlation analysis.")
             return

        # Group by complexity and shock parameters
        grouping_params = ['admin_levels', 'mean_parts', 'sd_parts',
                           'initial_population', # Include if varied
                           'individual_shock_probability', 'attachment_shock_magnitude']
        grouping_params = [p for p in grouping_params if p in df.columns]

        if not grouping_params:
             print("Warning: No valid grouping parameters found for complexity analysis.")
             return

        # Function to safely calculate correlation
        def safe_corr(x, col1, col2):
            if x[col1].nunique() > 1 and x[col2].nunique() > 1:
                return x[col1].corr(x[col2])
            return np.nan

        # Calculate correlations within each group
        grouped = df.groupby(grouping_params)
        correlation_results = []
        for name, group in grouped:
            group_corr = dict(zip(grouping_params, name)) # Start with group params
            for metric in metrics_available:
                group_corr[f'{metric}_corr'] = safe_corr(group, metric, 'mean_attachment')
            correlation_results.append(group_corr)

        self.effects_df = pd.DataFrame(correlation_results)

        # --- Composite Effect Analysis ---
        corr_cols = [col for col in self.effects_df.columns if '_corr' in col]
        if corr_cols:
            self.effects_df['composite_effect'] = self.effects_df[corr_cols].abs().mean(axis=1, skipna=True)
            self.effects_df['effect_consistency'] = self.effects_df[corr_cols].std(axis=1, skipna=True)
        else:
            print("Warning: No correlation columns found for composite effect calculation.")
            self.effects_df['composite_effect'] = np.nan
            self.effects_df['effect_consistency'] = np.nan


        # Save the effects dataframe
        effects_path = os.path.join(output_dir, 'complexity_effects_summary.csv')
        self.effects_df.to_csv(effects_path, index=False)
        print(f"Complexity effects summary saved to {effects_path}")

        # Find and save best parameters
        if 'composite_effect' in self.effects_df.columns:
             best_params = self.effects_df.nlargest(10, 'composite_effect')
             best_params_path = os.path.join(output_dir, 'complexity_best_params.csv')
             best_params.to_csv(best_params_path, index=False)
             print(f"Top 10 parameter combinations saved to {best_params_path}")


    def visualize_results(self, results_df: pd.DataFrame, output_dir: str):
        """Visualize complexity effects."""
        print("Generating complexity effect visualizations...")
        if self.effects_df.empty:
            print("No effects data to plot. Run analyze_results first.")
            # Try running analysis if results_df is available
            if not results_df.empty:
                 print("Running analysis to generate effects data...")
                 self.analyze_results(results_df, output_dir)
            if self.effects_df.empty: # Check again
                 return # Still no data

        # --- Plot 1: Heatmaps of Correlations ---
        try:
            metrics_to_plot = [col for col in self.effects_df.columns if '_corr' in col]
            num_metrics = len(metrics_to_plot)
            if num_metrics > 0:
                ncols = 3
                nrows = (num_metrics + ncols - 1) // ncols
                fig_heatmap, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
                axes = axes.flatten()

                for idx, metric in enumerate(metrics_to_plot):
                    ax = axes[idx]
                    try:
                        # Pivot for heatmap, index by complexity, columns by shock prob
                        pivot_data = self.effects_df.pivot_table(
                            values=metric,
                            index=['admin_levels', 'mean_parts', 'sd_parts'], # Combine complexity params
                            columns='individual_shock_probability', # Show shock effect in columns
                            aggfunc='mean' # Average if multiple runs per combo exist
                        )
                        sns.heatmap(
                            pivot_data, ax=ax, center=0, cmap='RdBu_r', annot=True, fmt='.2f',
                            cbar_kws={'label': 'Corr w/ Attachment'}
                        )
                        title = metric.replace('_corr', '').replace('_', ' ').title()
                        ax.set_title(f'PA Effect on {title}')
                        ax.set_xlabel('Shock Probability')
                        ax.set_ylabel('Complexity (Levels, Parts, SD)')
                        ax.tick_params(axis='y', rotation=0)
                    except Exception as heatmap_err:
                         print(f"Could not generate heatmap for {metric}: {heatmap_err}")
                         ax.set_title(f'PA Effect on {metric.replace("_corr","").title()} (Error)')


                for i in range(num_metrics, len(axes)): fig_heatmap.delaxes(axes[i])
                plt.tight_layout()
                fig_heatmap.savefig(os.path.join(output_dir, 'complexity_correlation_heatmaps.png'))
                plt.close(fig_heatmap)
            else:
                print("No correlation metrics found to plot heatmaps.")

        except Exception as e:
            print(f"Error generating heatmap plots: {e}")
            plt.close('all') # Close any partially created figures

        # --- Plot 2: Composite Effect Scatter Plot ---
        try:
            if 'composite_effect' in self.effects_df.columns:
                fig_scatter = plt.figure(figsize=(12, 7))
                # Use mean_parts as an example complexity axis
                complexity_axis = 'mean_parts' if 'mean_parts' in self.effects_df.columns else 'admin_levels'

                sns.scatterplot(
                    data=self.effects_df,
                    x=complexity_axis,
                    y='composite_effect',
                    size='effect_consistency',
                    hue='individual_shock_probability',
                    style='admin_levels' if complexity_axis != 'admin_levels' and 'admin_levels' in self.effects_df.columns else None,
                    alpha=0.8,
                    palette='viridis',
                    sizes=(30, 300) # Adjust size range
                )
                plt.title('Parameter Effects on Place Attachment Impact (Complexity vs Shock)')
                plt.xlabel(f'{complexity_axis.replace("_"," ").title()} (Complexity)')
                plt.ylabel('Composite Effect Size (Avg |Corr w/ PA|)')
                plt.legend(title='Shock Prob / Style', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                fig_scatter.savefig(os.path.join(output_dir, 'complexity_composite_effects.png'))
                plt.close(fig_scatter)
            else:
                 print("Composite effect not calculated, skipping scatter plot.")

        except Exception as e:
            print(f"Error generating composite effect scatter plot: {e}")
            plt.close('all')


    # Remove old methods: run_parallel_experiments, analyze_results (old), plot_complexity_effects (old),
    # analyze_parameter_effects, export_effects_to_csv


if __name__ == "__main__":
  try:
    model_path = "./migration_place_attachment_model.nlogo"
    print("Running Complexity Analysis Experiment...")

    # Instantiate the specific experiment
    experiment = ComplexityAnalysisExperiment(model_path)

    # Run using the base class method
    results_df = experiment.run_experiments(
        n_iterations=2,   # Number of repetitions per parameter set
        # num_processes= Use default or specify
        num_ticks=60      # Number of NetLogo ticks per run
    )

    if not results_df.empty:
        print("\nExperiment finished. Aggregated results summary:")
        print(results_df.head())
        print(f"\nFull analysis and plots saved in the 'results_YYYYMMDD_HHMMSS' directory.")
    else:
        print("\nExperiment finished, but no results were generated.")

  except Exception as e:
      print(f"\nError during experiment execution: {e}")
      import traceback
      traceback.print_exc()
  finally:
      # Ensure all matplotlib figures are closed after processing
      plt.close('all')
      print("\nMain process finished.")