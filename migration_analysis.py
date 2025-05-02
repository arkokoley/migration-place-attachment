import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import matplotlib
import itertools
from base_experiment import BaseExperiment
from sklearn.ensemble import RandomForestRegressor
import scipy.io as sio # Keep for .mat export
from matplotlib.patches import Patch # Keep for RF plot legend

matplotlib.use('Agg')  # Set the backend to non-interactive

class PlaceConfig:
    mean_quality: float
    sd_quality: float
    mean_income: float
    sd_income: float
    capacity: float
    
    @classmethod
    def high_income_high_qol(cls):
        return cls(
            mean_quality=45,
            sd_quality=5,
            mean_income=45,
            sd_income=5,
            capacity=100
        )
    
    @classmethod
    def low_income_low_qol(cls):
        return cls(
            mean_quality=15,
            sd_quality=5,
            mean_income=15,
            sd_income=5,
            capacity=100
        )

class LandscapeGenerator:
    def __init__(self, admin_levels=2, mean_parts=3, sd_parts=1):
        self.admin_levels = admin_levels
        self.mean_parts = mean_parts
        self.sd_parts = sd_parts
        
    def generate_place_configs(self, base_config: PlaceConfig) -> Dict[str, PlaceConfig]:
        places = {}
        # Generate hierarchical place structure
        self._generate_places([], 0, places, base_config)
        return places
        
    def _generate_places(self, prefix: List[int], level: int, 
                        places: Dict[str, PlaceConfig], base_config: PlaceConfig):
        if level == self.admin_levels:
            place_name = '_'.join(map(str, prefix))
            # Create variation in place configurations
            variation = np.random.normal(0, 0.2)  # 20% standard deviation
            places[place_name] = PlaceConfig(
                mean_quality=base_config.mean_quality * (1 + variation),
                sd_quality=base_config.sd_quality,
                mean_income=base_config.mean_income * (1 + variation),
                sd_income=base_config.sd_income,
                capacity=base_config.capacity
            )
            return
            
        num_parts = max(1, int(np.random.normal(self.mean_parts, self.sd_parts)))
        for i in range(num_parts):
            new_prefix = prefix + [i]
            self._generate_places(new_prefix, level + 1, places, base_config)

class ParameterSweepExperiment(BaseExperiment):
    """Runs a parameter sweep experiment for the migration model."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        # Define parameter ranges for both shock types
        self.param_ranges = {
            # --- Shock Parameters ---
            'shock_type': ["none", "individual", "place"],  # Control which shock mechanism
            'individual_shock_probability': [0, 0.01, 0.05],  # Include 0 for control
            'attachment_shock_magnitude': [0.1, 0.5],  # Both individual & place use same magnitudes
            'place_shock_probability': [0, 0.01, 0.05],
            'place_shock_magnitude': [0.1, 0.5],

            # --- Key Parameters ---
            'mean_attachment': np.linspace(0.2, 0.8, 4),  # Test range of attachments
            
            # --- Fixed Parameters ---
            'sd_attachment': [0.2],
            'attachment_form_rate': [0.019],
            'attachment_decay_rate': [0.02],
            'initial_attachment': [0.5],
            'mean_pInitiate': [0.3],
            'sd_pInitiate': [0.1],
            'pConsumat': [0.4],
            # Add other necessary fixed params if they are not default in NetLogo
            # 'initial_population': [500], # Example
        }

    def _define_parameter_sets(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations, filtering invalid shock combinations."""
        keys = self.param_ranges.keys()
        values = self.param_ranges.values()
        raw_combinations = list(itertools.product(*values))
        
        parameter_sets = []
        for combo in raw_combinations:
            param_set = dict(zip(keys, combo))
            
            # Filter based on shock_type to ensure consistent parameters
            shock_type = param_set['shock_type']
            if shock_type == "none":
                param_set['individual_shock_probability'] = 0
                param_set['place_shock_probability'] = 0
                param_set['attachment_shock_magnitude'] = 0
                param_set['place_shock_magnitude'] = 0
            elif shock_type == "individual":
                param_set['place_shock_probability'] = 0
                param_set['place_shock_magnitude'] = 0
            elif shock_type == "place":
                param_set['individual_shock_probability'] = 0
                param_set['attachment_shock_magnitude'] = 0
                
            parameter_sets.append(param_set)
        
        # Remove duplicates
        return [dict(t) for t in {tuple(d.items()) for d in parameter_sets}]

    def _get_metrics_to_collect(self) -> Dict[str, str]:
        """Define MINIMAL metrics for testing shocks."""
        return {
            # Core metrics from get-experiment-metrics
            'total_moves': 'total_moves', # General activity indicator
            # RCT metrics are essential
            'avg_satisfaction_shocked': '(item 0 get-metrics-by-shock-status)',
            'avg_satisfaction_not_shocked': '(item 1 get-metrics-by-shock-status)',
            'avg_moves_shocked': '(item 2 get-metrics-by-shock-status)',
            'avg_moves_not_shocked': '(item 3 get-metrics-by-shock-status)',
            # Optional: Add avg_satisfaction for context if needed
            # 'avg_satisfaction': 'ifelse-value empty? satisfaction_levels [0] [mean satisfaction_levels]',
        }

    def analyze_results(self, results_df: pd.DataFrame, output_dir: str):
        """Perform minimal analysis (or none) for prototyping."""
        print("Skipping Random Forest analysis for prototype.")
        # rf_results = self._analyze_random_forest(results_df, output_dir)
        # if rf_results:
        #      print("Random Forest analysis complete.")
        # else:
        #      print("Random Forest analysis skipped or failed.")

        print("Skipping MATLAB export for prototype.")
        # self._export_for_matlab(results_df, output_dir)
        # print("MATLAB export complete.")

        # No other analysis needed for minimal prototype

    def visualize_results(self, results_df: pd.DataFrame, output_dir: str):
        """Generate minimal visualizations for prototyping."""
        print("Skipping Place Attachment effect plots for prototype.")
        # try:
        #     pa_fig = self._visualize_place_attachment_effects(results_df)
        #     pa_fig.savefig(os.path.join(output_dir, 'place_attachment_effects.png'))
        #     plt.close(pa_fig)
        # except Exception as e:
        #     print(f"Error generating PA effect plots: {e}")

        print("Skipping Migration Pattern plots for prototype.")
        # try:
        #     mig_fig = self._analyze_migration_patterns(results_df)
        #     mig_fig.savefig(os.path.join(output_dir, 'migration_patterns.png'))
        #     plt.close(mig_fig)
        # except Exception as e:
        #     print(f"Error generating migration pattern plots: {e}")

        print("Generating Shock RCT effect plots...") # Keep this one
        try:
            rct_fig = self._visualize_shock_effect_rct(results_df)
            if rct_fig: # Check if figure was created (might return None on error/no data)
                 rct_fig.savefig(os.path.join(output_dir, 'shock_rct_effect.png'))
                 plt.close(rct_fig)
            else:
                 print("RCT plot not generated (likely missing data or error).")
        except Exception as e:
            print(f"Error generating RCT plots: {e}")

    def _visualize_shock_effect_rct(self, results_df: pd.DataFrame):
        """Visualize RCT effects comparing individual and place shocks across attachment levels."""
        if not all(col in results_df.columns for col in ['shock_type', 'mean_attachment']):
            print("Warning: Missing required columns for RCT visualization")
            return None

        # Create figure with 2 subplots (satisfaction and moves)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Process data for each shock type
        for shock_type in ['individual', 'place']:
            shock_data = results_df[results_df['shock_type'] == shock_type].copy()
            if shock_data.empty:
                continue

            # Calculate effect sizes
            shock_data['satisfaction_effect'] = shock_data['avg_satisfaction_shocked'] - shock_data['avg_satisfaction_not_shocked']
            shock_data['moves_effect'] = shock_data['avg_moves_shocked'] - shock_data['avg_moves_not_shocked']

            # Group by attachment level and calculate mean effects
            grouped = shock_data.groupby('mean_attachment').agg({
                'satisfaction_effect': ['mean', 'std'],
                'moves_effect': ['mean', 'std']
            }).reset_index()

            # Plot satisfaction effects
            ax1.errorbar(grouped['mean_attachment'], 
                        grouped['satisfaction_effect']['mean'],
                        yerr=grouped['satisfaction_effect']['std'],
                        label=f'{shock_type.title()} Shock',
                        marker='o',
                        capsize=5)

            # Plot movement effects
            ax2.errorbar(grouped['mean_attachment'], 
                        grouped['moves_effect']['mean'],
                        yerr=grouped['moves_effect']['std'],
                        label=f'{shock_type.title()} Shock',
                        marker='o',
                        capsize=5)

        # Customize satisfaction plot
        ax1.set_title('Shock Effects on Satisfaction by Attachment Level')
        ax1.set_xlabel('Mean Attachment Level')
        ax1.set_ylabel('Satisfaction Effect\n(Shocked - Not Shocked)')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Customize moves plot
        ax2.set_title('Shock Effects on Movement by Attachment Level')
        ax2.set_xlabel('Mean Attachment Level')
        ax2.set_ylabel('Movement Effect\n(Shocked - Not Shocked)')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _export_for_matlab(self, results_df: pd.DataFrame, output_dir: str):
        """Exports data in a format suitable for the MATLAB RF script."""
        # Define inputs and metrics expected by the MATLAB script
        input_names_matlab = [
            'placeAttachmentMean', 'placeAttachmentSD', 'attachmentFormRate',
            'attachmentDecayRate', 'initialAttachment', 'pInitiateMean',
            'pInitiateSD', 'pConsumat', 'individualShockProb', 'attachmentShockMag'
        ]
        metric_names_matlab = [
            'totalMoves', 'avgMoveDistance', 'avgSatisfaction',
            'timeAwayFromHome', 'fractionMovesHome', 'utilityChange'
        ]

        # Map DataFrame columns to MATLAB names
        param_map = {
            'mean_attachment': 'placeAttachmentMean',
            'sd_attachment': 'placeAttachmentSD',
            'attachment_form_rate': 'attachmentFormRate',
            'attachment_decay_rate': 'attachmentDecayRate',
            'initial_attachment': 'initialAttachment',
            'mean_pInitiate': 'pInitiateMean',
            'sd_pInitiate': 'pInitiateSD',
            'pConsumat': 'pConsumat',
            'individual_shock_probability': 'individualShockProb',
            'attachment_shock_magnitude': 'attachmentShockMag'
        }
        metric_map = {
            'total_moves': 'totalMoves',
            'avg_move_distance': 'avgMoveDistance',
            'avg_satisfaction': 'avgSatisfaction',
            'TimeStepsAwayFromHome': 'timeAwayFromHome',
            'FractionMovesHome': 'fractionMovesHome',
            'avg_utility_change': 'utilityChange'
        }

        # Select and rename columns
        all_inputs_df = results_df[[k for k in param_map if k in results_df.columns]].rename(columns=param_map)
        all_metrics_df = results_df[[k for k in metric_map if k in results_df.columns]].rename(columns=metric_map)

        # Ensure columns match the expected order and fill missing with NaN
        all_inputs_df = all_inputs_df.reindex(columns=input_names_matlab)
        all_metrics_df = all_metrics_df.reindex(columns=metric_names_matlab)

        # Convert to numpy arrays
        all_inputs = all_inputs_df.to_numpy()
        all_metrics = all_metrics_df.to_numpy()

        # Check for NaNs which might cause issues in MATLAB
        if np.isnan(all_inputs).any() or np.isnan(all_metrics).any():
            print("Warning: NaN values found in data being exported to MATLAB.")
            # Optionally handle NaNs, e.g., replace with a placeholder or drop rows
            # all_inputs = np.nan_to_num(all_inputs)
            # all_metrics = np.nan_to_num(all_metrics)

        timestamp = os.path.basename(output_dir).replace('results_', '')
        mat_file_path = os.path.join(output_dir, f'PA_Sensitivity_Results_{timestamp}.mat')
        try:
            sio.savemat(mat_file_path, {
                'allInputs': all_inputs,
                'allMetrics': all_metrics,
                'inputNames': np.array(input_names_matlab, dtype=object),
                'metricNames': np.array(metric_names_matlab, dtype=object)
            })
        except Exception as e:
            print(f"Error saving .mat file: {e}")


    def _analyze_random_forest(self, results_df: pd.DataFrame, output_dir: str):
        """Generate Random Forest analysis plots similar to MATLAB implementation."""
        # Define potential parameters including new shock params
        potential_params = list(self.param_ranges.keys()) # Use defined ranges
        available_params = [param for param in potential_params if param in results_df.columns]

        # Define potential metrics based on what was collected
        potential_metrics = list(self._get_metrics_to_collect().keys())
        available_metrics = [metric for metric in potential_metrics if metric in results_df.columns]

        if not available_params or not available_metrics:
            print("Warning: Not enough parameters or metrics for random forest analysis")
            return None

        # Prepare data - handle potential NaNs
        results_df_rf = results_df.dropna(subset=available_params + available_metrics).copy() # Use copy
        if results_df_rf.empty:
             print("Warning: No valid data remaining after dropping NaNs for RF analysis.")
             return None

        # Ensure numeric types for RF
        for col in available_params + available_metrics:
             results_df_rf[col] = pd.to_numeric(results_df_rf[col], errors='coerce')
        results_df_rf.dropna(subset=available_params + available_metrics, inplace=True)
        if results_df_rf.empty:
             print("Warning: No numeric data remaining after coercion/dropping NaNs for RF analysis.")
             return None


        X = results_df_rf[available_params].values
        importance_scores = {}
        rf_models = {}

        # Colors for parameter types
        pa_params = ['mean_attachment', 'sd_attachment', 'attachment_form_rate',
                    'attachment_decay_rate', 'initial_attachment', 'attachment_shock_magnitude'] # Added shock mag
        shock_params = ['individual_shock_probability'] # Separate shock prob
        param_colors = []
        for param in available_params:
            if param in pa_params:
                param_colors.append('#3399CC') # Blue for PA
            elif param in shock_params:
                param_colors.append('#FF8C00') # Orange for Shock
            else:
                param_colors.append('#CC6644') # Red-Brown for Other

        # Generate plots for each metric
        for metric in available_metrics:
            y = results_df_rf[metric].values
            if len(np.unique(y)) <= 1: # Skip if metric is constant
                 print(f"Skipping RF for constant metric: {metric}")
                 continue

            # Train Random Forest
            try:
                rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1) # Use more cores
                rf.fit(X, y)
            except Exception as rf_error:
                print(f"Error training RF for metric {metric}: {rf_error}")
                continue


            # Store model and importance scores
            rf_models[metric] = rf
            importance_scores[metric] = rf.feature_importances_

            # Plot variable importance
            plt.figure(figsize=(12, 8))
            sorted_idx = np.argsort(rf.feature_importances_)
            pos = np.arange(len(sorted_idx)) + .5

            # Create bar plot with color coding
            bars = plt.barh(pos, rf.feature_importances_[sorted_idx], align='center') # Use align='center'
            # Apply colors correctly based on sorted index
            sorted_colors = [param_colors[i] for i in sorted_idx]
            for bar, color in zip(bars, sorted_colors):
                 bar.set_color(color)


            plt.yticks(pos, np.array(available_params)[sorted_idx])
            plt.xlabel('Relative Importance')
            plt.title(f'Variable Importance for {metric}')

            # Add legend
            legend_elements = [
                Patch(facecolor='#3399CC', label='Place Attachment Params'),
                Patch(facecolor='#FF8C00', label='Shock Params'), # Add shock legend
                Patch(facecolor='#CC6644', label='Other Params')
            ]
            plt.legend(handles=legend_elements)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'rf_importance_{metric}.png')) # Save in output dir
            plt.close()


        # Generate heatmap if we have data
        if importance_scores:
            plt.figure(figsize=(14, 10)) # Increase size slightly
            # Ensure metrics used here match those looped above
            valid_metrics_for_heatmap = [m for m in available_metrics if m in importance_scores]
            if valid_metrics_for_heatmap:
                 importance_matrix = np.array([importance_scores[m] for m in valid_metrics_for_heatmap])
                 sns.heatmap(importance_matrix, xticklabels=available_params,
                            yticklabels=valid_metrics_for_heatmap, # Use valid metrics
                            cmap='YlOrRd', annot=True, fmt='.3f') # Increase precision
                 plt.title('Parameter Importance Across Metrics')
                 plt.xticks(rotation=45, ha='right') # Adjust rotation
                 plt.yticks(rotation=0)
                 plt.tight_layout()
                 plt.savefig(os.path.join(output_dir, f'rf_importance_heatmap.png')) # Save in output dir
                 plt.close()


        return {
            'models': rf_models,
            'importance_scores': importance_scores,
            'input_params': available_params,
            'metrics': available_metrics # Return available metrics used
        }


    def _visualize_place_attachment_effects(self, results_df: pd.DataFrame):
        """Create visualizations showing place attachment effects on various metrics."""
        # Use 'mean_attachment' if 'attachment_level' was just a placeholder
        x_param = 'mean_attachment' # This should be the varied parameter

        required_cols = [x_param, 'total_moves', 'avg_move_distance',
                        'FractionMovesHome', 'avg_satisfaction']

        available_cols = [col for col in required_cols if col in results_df.columns]
        if not available_cols or x_param not in available_cols: # Ensure x_param is available
            print(f"Warning: Required columns missing for PA visualization (need {x_param} and metrics)")
            return plt.figure()

        # Drop rows with NaN in essential columns for plotting
        plot_df = results_df[available_cols].dropna()
        if plot_df.empty:
             print(f"Warning: No valid data for PA visualization after dropping NaNs.")
             return plt.figure()


        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 3, figure=fig)

        # Plot 1: Place Attachment vs Total Moves
        ax1 = fig.add_subplot(gs[0, 0])
        if 'total_moves' in plot_df.columns:
             sns.regplot(data=plot_df, x=x_param, y='total_moves', ax=ax1, line_kws={'color': 'red'}, scatter_kws={'alpha':0.5})
        ax1.set_title(f'{x_param.replace("_", " ").title()} vs Total Moves')
        ax1.set_xlabel(x_param.replace("_", " ").title())


        # Plot 2: Place Attachment vs Distance
        ax2 = fig.add_subplot(gs[0, 1])
        if 'avg_move_distance' in plot_df.columns:
            sns.regplot(data=plot_df, x=x_param, y='avg_move_distance', ax=ax2, line_kws={'color': 'red'}, scatter_kws={'alpha':0.5})
        ax2.set_title(f'{x_param.replace("_", " ").title()} vs Move Distance')
        ax2.set_xlabel(x_param.replace("_", " ").title())


        # Plot 3: Place Attachment vs Returns Home
        ax3 = fig.add_subplot(gs[0, 2])
        if 'FractionMovesHome' in plot_df.columns:
            sns.regplot(data=plot_df, x=x_param, y='FractionMovesHome', ax=ax3, line_kws={'color': 'red'}, scatter_kws={'alpha':0.5})
        ax3.set_title(f'{x_param.replace("_", " ").title()} vs Fraction Home')
        ax3.set_xlabel(x_param.replace("_", " ").title())


        # Plot 4: Correlation Heatmap
        ax4 = fig.add_subplot(gs[1, :])
        # Ensure only numeric columns are used for correlation
        numeric_cols_for_corr = plot_df.select_dtypes(include=np.number).columns
        if len(numeric_cols_for_corr) > 1:
             correlation_matrix = plot_df[numeric_cols_for_corr].corr()
             sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', center=0, ax=ax4, fmt=".2f")
        ax4.set_title('Correlation Heatmap of Key Metrics')

        plt.tight_layout()
        return fig


    def _analyze_migration_patterns(self, results_df: pd.DataFrame):
        """Analyze and visualize migration patterns."""
        # Use 'mean_attachment' if 'attachment_level' was just a placeholder
        x_param = 'mean_attachment' # This should be the varied parameter

        if x_param not in results_df.columns:
            print(f"Warning: {x_param} column missing for migration patterns")
            return plt.figure()

        # Drop NaNs for relevant columns
        plot_df = results_df.dropna(subset=[x_param, 'total_moves', 'avg_move_distance', 'pConsumat', 'mean_pInitiate']).copy()
        if plot_df.empty:
             print("Warning: No valid data for migration pattern plots after dropping NaNs.")
             return plt.figure()


        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, figure=fig)

        # Plot 1: Migration Flow by Attachment Level
        ax1 = fig.add_subplot(gs[0, 0])
        sns.lineplot(data=plot_df, x=x_param, y='total_moves', ax=ax1, errorbar='sd') # Add error bars
        ax1.set_title(f'Migration Flow by {x_param.replace("_", " ").title()}')
        ax1.set_xlabel(x_param.replace("_", " ").title())
        ax1.set_ylabel('Total Moves')

        # Plot 2: Distribution of Move Distances
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(data=plot_df, x='avg_move_distance', kde=True, ax=ax2)
        ax2.set_title('Distribution of Average Move Distances')
        ax2.set_xlabel('Average Move Distance')
        ax2.set_ylabel('Count')

        # Plot 3: Attachment vs Migration Decision Type
        ax3 = fig.add_subplot(gs[1, 0])
        # Ensure pConsumat is categorical for boxplot
        plot_df['pConsumat_cat'] = pd.Categorical(plot_df['pConsumat'])
        hue_param = x_param if plot_df[x_param].nunique() > 1 and plot_df[x_param].nunique() < 10 else None # Limit hue categories
        sns.boxplot(data=plot_df, x='pConsumat_cat', y='total_moves',
                    hue=hue_param, ax=ax3) # Use x_param for hue if suitable
        ax3.set_title(f'Migration by Decision Type and {x_param.replace("_", " ").title()}')
        ax3.set_xlabel('Probability of Consumat Decision')
        ax3.set_ylabel('Total Moves')
        if hue_param and ax3.get_legend():
             ax3.get_legend().set_title(x_param.replace("_", " ").title()) # Use x_param


        # Plot 4: Network Effects and Attachment
        ax4 = fig.add_subplot(gs[1, 1])
        scatter = sns.scatterplot(data=plot_df, x='mean_pInitiate', y='total_moves',
                                size=x_param, hue=x_param, # Use x_param
                                sizes=(50, 200), ax=ax4, alpha=0.7)
        if scatter.legend_ is not None:
            try: # Handle potential legend errors
                 scatter.legend_.set_title(x_param.replace("_", " ").title()) # Use x_param
            except AttributeError: pass # Ignore if legend title cannot be set
        ax4.set_title('Network Effects and Place Attachment')
        ax4.set_xlabel('Mean Initiate Probability')
        ax4.set_ylabel('Total Moves')

        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    try:
        model_path = "./migration_place_attachment_model.nlogo"
        print("Running MINIMAL Parameter Sweep Experiment (Shock Prototype)...")

        # Instantiate the specific experiment
        experiment = ParameterSweepExperiment(model_path)

        # Run using the base class method - use fewer iterations/ticks for speed
        results_df = experiment.run_experiments(
            n_iterations=2,   # Reduced iterations
            # num_processes= Use default or specify
            num_ticks=30      # Reduced ticks
        )

        if not results_df.empty:
            print("\nExperiment finished. Aggregated results summary:")
            print(results_df.head())
            # Focus on RCT columns
            rct_cols = [c for c in results_df.columns if 'shocked' in c]
            if rct_cols:
                 print("\nRCT Metrics:")
                 print(results_df[['individual_shock_probability', 'mean_attachment'] + rct_cols])

            print(f"\nMinimal results/plots saved in the 'results_YYYYMMDD_HHMMSS' directory.")
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
