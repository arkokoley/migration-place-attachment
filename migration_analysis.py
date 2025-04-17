import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pynetlogo import NetLogoLink
import os
from datetime import datetime
import networkx as nx
from typing import List, Dict
import numpy as np
import matplotlib
import itertools
matplotlib.use('Agg')  # Set the backend to non-interactive before other imports

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

class MigrationAnalyzer:
    def __init__(self, model_path):
        """Initialize the NetLogo connection and setup basic parameters."""
        try:
            self.netlogo = NetLogoLink(gui=True)  # Change to headless mode
            self.netlogo.load_model(model_path)
            self.results = {}
        except Exception as e:
            print(f"Error initializing NetLogo: {e}")
            raise
        
    def setup_experiment(self, params=None):
        """Setup experiment parameters with defaults."""
        self.params = params or {
            'mean_attachment': np.linspace(0, 1, 3),  # Reduced from 5 to 3
            'sd_attachment': np.linspace(0.1, 0.5, 2),  # Reduced from 3 to 2
            'attachment_form_rate': np.linspace(0.01, 0.05, 2),
            'attachment_decay_rate': np.linspace(0.01, 0.05, 2),
            'initial_attachment': np.linspace(0.2, 0.8, 2),
            'mean_pInitiate': np.linspace(0.1, 0.5, 2),
            'sd_pInitiate': np.linspace(0.05, 0.2, 2),
            'pConsumat': [0.2, 0.6]  # Reduced to just low/high
        }
        
        # Initialize metrics dictionary with additional metrics
        self.metrics = {
            'move_distances': [],
            'yearly_moves': [],
            'income_improvement': [],
            'qol_improvement': [],
            'cyclical_moves': [],
            'time_to_next_move': [],
            'attachment_level': [],
            'mean_pInitiate': [],
            'pConsumat': [],
            'avg_satisfaction': [],
            'num_uncertain': [],
            'avg_move_distance': [],
            'total_moves': [],
            'satisfaction_levels': [],
            'utility_changes': [],
            'total_satisfaction': [],
            'total_uncertainty': [],
            'TimeStepsAwayFromHome': [],
            'AvgDistanceAwayFromHome': [],
            'FractionMovesHome': [],
            'TotalDistanceTraveled': [],
            'NumReturnsToHome': [],
            'numReturns': [],
            'distanceTraveled': [],
            'totalMoves': [],
            'avgDistancePerMove': [],
            'fractionMovesHome': [],
            'avgDistanceAwayFromHome': [],
            'timeStepsAwayFromHome': [],
            'numAgents': []
        }
        
    def setup_landscape(self, landscape_config: PlaceConfig):
        """Setup landscape with specific configuration"""
        generator = LandscapeGenerator(
            admin_levels=2,  # Start simple
            mean_parts=2,    # Few subdivisions
            sd_parts=0.5     # Low variation
        )
        
        place_configs = generator.generate_place_configs(landscape_config)
        
        # Set NetLogo parameters
        for place_name, config in place_configs.items():
            self.netlogo.command(f'''
                ask places with [name = "{place_name}"] [
                    set place_mean_q {config.mean_quality}
                    set place_sd_q {config.sd_quality}
                    set place_mean_i {config.mean_income}
                    set place_sd_i {config.sd_income}
                    set place_a {config.capacity}
                ]
            ''')

    def run_parameter_sweep(self, num_ticks=50, runs_per_combo=3):
        """Run experiments for all parameter combinations and save comprehensive metrics."""
        results = []
        all_inputs = []
        all_metrics = []
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations()
        
        for params in param_combinations:
            for run in range(runs_per_combo):
                # Run single experiment with parameter combination
                run_metrics = self.run_single_experiment(params=params, ticks=num_ticks)
                
                # Collect input parameters in order
                input_values = [
                    params['mean_attachment'],
                    params.get('sd_attachment', 0.2),
                    params.get('attachment_form_rate', 0.019),
                    params.get('attachment_decay_rate', 0.02),
                    params.get('initial_attachment', 0.5),
                    params['mean_pInitiate'],
                    params.get('sd_pInitiate', 0.1),
                    params['pConsumat'],
                ]
                
                # Collect metrics in order
                metric_values = [
                    run_metrics['total_moves'],
                    run_metrics['avg_move_distance'],
                    run_metrics['avg_satisfaction'],
                    run_metrics['timeStepsAwayFromHome'],
                    run_metrics['fractionMovesHome'],
                    run_metrics['avg_utility_change']
                ]
                
                all_inputs.append(input_values)
                all_metrics.append(metric_values)
                results.append({**run_metrics, 'run_id': run, **params})
        
        # Save results in both formats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed CSV
        pd.DataFrame(results).to_csv(f'migration_results_detailed_{timestamp}.csv')
        
        # Save MATLAB format for random forest analysis
        import scipy.io as sio
        sio.savemat(f'PA_Sensitivity_Results_{timestamp}.mat', {
            'allInputs': np.array(all_inputs),
            'allMetrics': np.array(all_metrics),
            'inputNames': np.array([
                'placeAttachmentMean',
                'placeAttachmentSD',
                'attachmentFormRate',
                'attachmentDecayRate',
                'initialAttachment',
                'pInitiateMean',
                'pInitiateSD',
                'pConsumat'
            ]),
            'metricNames': np.array([
                'totalMoves',
                'avgMoveDistance',
                'avgSatisfaction',
                'timeAwayFromHome',
                'fractionMovesHome',
                'utilityChange'
            ])
        })
        
        return pd.DataFrame(results)

    def run_single_experiment(self, params, ticks=60):
        """Run single experiment with given parameters."""
        try:
            # Initialize metrics with default values
            experiment_metrics = {
                'move_distances': [],
                'yearly_moves': 0,
                'income_improvement': 0,
                'qol_improvement': 0,
                'cyclical_moves': 0,
                'time_to_next_move': 0,
                'attachment_level': params.get('mean_attachment', 0),
                'mean_pInitiate': params.get('mean_pInitiate', 0),
                'pConsumat': params.get('pConsumat', 0),
                'avg_satisfaction': 0,
                'num_uncertain': 0,
                'avg_move_distance': 0,
                'total_moves': 0,
                'satisfaction_levels': [],
                'utility_changes': [],
                'total_satisfaction': 0,
                'total_uncertainty': 0,
                'TimeStepsAwayFromHome': 0,
                'AvgDistanceAwayFromHome': 0,
                'FractionMovesHome': 0,
                'TotalDistanceTraveled': 0,
                'NumReturnsToHome': 0,
                'avg_utility_change': 0  # Add default value
            }

            # Set parameters and initialize
            for param, value in params.items():
                self.netlogo.command(f'set {param} {value}')
            
            self.netlogo.command('setup')

            # Track metrics at each tick
            for t in range(ticks):
                self.netlogo.command('go')
                
                # Get metrics from NetLogo with error handling
                try:
                    moves = float(self.netlogo.report('moves'))
                    moves = moves if not np.isnan(moves) else 0
                    far_from_home = float(self.netlogo.report('far_from_home'))
                    far_from_home = far_from_home if not np.isnan(far_from_home) else 0
                    at_home = float(self.netlogo.report('at_home'))
                    at_home = at_home if not np.isnan(at_home) else 0
                    total_people = float(self.netlogo.report('count people'))
                    total_people = total_people if not np.isnan(total_people) else 1
                except Exception as e:
                    print(f"Error getting NetLogo metrics: {e}")
                    moves = far_from_home = at_home = 0
                    total_people = 1
                
                # Update metrics
                experiment_metrics['total_moves'] += moves
                experiment_metrics['TimeStepsAwayFromHome'] += far_from_home
                experiment_metrics['FractionMovesHome'] = at_home / total_people if total_people > 0 else 0
                
                # Get improvement metrics with error handling
                try:
                    improvements = self.netlogo.report('get-avg-improvement')
                    experiment_metrics['income_improvement'] += improvements[0]
                    experiment_metrics['qol_improvement'] += improvements[1]
                except Exception as e:
                    print(f"Error getting improvement metrics: {e}")
                
            # Get final metrics with error handling
            try:
                final_metrics = self.netlogo.report('get-experiment-metrics')
                if final_metrics is not None and len(final_metrics) >= 5:
                    # Convert any numpy arrays to floats and handle NaN values
                    total_moves = float(final_metrics[0]) if not np.isnan(final_metrics[0]) else 0
                    avg_move_dist = float(final_metrics[1]) if not np.isnan(final_metrics[1]) else 0
                    avg_satis = float(final_metrics[2]) if not np.isnan(final_metrics[2]) else 0
                    avg_util = float(final_metrics[3]) if not np.isnan(final_metrics[3]) else 0
                    num_uncert = float(final_metrics[4]) if not np.isnan(final_metrics[4]) else 0
                    
                    experiment_metrics.update({
                        'total_moves': total_moves,
                        'avg_move_distance': avg_move_dist,
                        'avg_satisfaction': avg_satis,
                        'avg_utility_change': avg_util,
                        'num_uncertain': num_uncert
                    })
            except Exception as e:
                print(f"Error getting final metrics: {e}")
            
            # Calculate yearly_moves
            experiment_metrics['yearly_moves'] = experiment_metrics['total_moves'] / (ticks / 12)

            return experiment_metrics

        except Exception as e:
            print(f"Error in experiment: {e}")
            return {k: 0 for k in experiment_metrics.keys()} | {'success': False, **params}

    def run_experiments(self, n_iterations=5):
        """Run experiments and generate all visualizations."""
        if 'mean_attachment' not in self.params:
            raise ValueError("No mean_attachment parameter found in experiment setup")
            
        results = []
        param_combinations = self._generate_param_combinations()
        
        for params in param_combinations:
            level_results = []
            for _ in range(n_iterations):
                result = self.run_single_experiment(params)
                # Add parameters to result
                result.update(params)
                level_results.append(result)
            
            # Average results across iterations
            avg_result = {}
            for k in level_results[0].keys():
                values = [r.get(k, 0) for r in level_results]
                if isinstance(values[0], (list, np.ndarray)):
                    # For list metrics, take the mean of each position
                    avg_result[k] = np.mean(values, axis=0) if values else []
                else:
                    # For numeric metrics, take simple mean
                    avg_result[k] = float(np.mean([v for v in values if v is not None]))
            
            results.append(avg_result)
        
        # Create DataFrame from results
        results_df = pd.DataFrame(results)
        
        # Generate visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'results_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Place attachment effects
        pa_fig = self.visualize_place_attachment_effects(results_df)
        pa_fig.savefig(os.path.join(output_dir, 'place_attachment_effects.png'))
        plt.close(pa_fig)
        
        # Migration patterns
        mig_fig = self.analyze_migration_patterns(results_df)
        mig_fig.savefig(os.path.join(output_dir, 'migration_patterns.png'))
        plt.close(mig_fig)
        
        # Random Forest analysis
        rf_results = self.analyze_random_forest(results_df)
        
        # Save results
        results_df.to_csv(os.path.join(output_dir, 'migration_results.csv'))
        
        return results_df, rf_results

    def _calculate_cyclical_moves(self, location_history):
        """Calculate the proportion of moves that are cyclical."""
        cyclical_count = 0
        total_moves = 0
        
        for agent_history in location_history.values():
            if len(agent_history) < 3:
                continue
                
            for i in range(len(agent_history) - 2):
                if agent_history[i] == agent_history[i + 2]:
                    cyclical_count += 1
                total_moves += 1
        
        return cyclical_count / total_moves
    
    def visualize_place_attachment_effects(self, results_df):
        """Create visualizations showing place attachment effects on various metrics."""
        # First verify columns exist
        required_cols = ['attachment_level', 'total_moves', 'avg_move_distance', 
                        'FractionMovesHome', 'avg_satisfaction']
        
        # Use attachment_level instead of mean_attachment
        available_cols = [col for col in required_cols if col in results_df.columns]
        if not available_cols:
            print("Warning: Required columns missing for visualization")
            return plt.figure()  # Return empty figure
            
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # Plot 1: Place Attachment vs Total Moves
        ax1 = fig.add_subplot(gs[0, 0])
        sns.regplot(data=results_df, x='attachment_level', y='total_moves', ax=ax1)
        ax1.set_title('Place Attachment vs Total Moves')
        
        # Plot 2: Place Attachment vs Distance
        ax2 = fig.add_subplot(gs[0, 1])
        sns.regplot(data=results_df, x='attachment_level', y='avg_move_distance', ax=ax2)
        ax2.set_title('Place Attachment vs Move Distance')
        
        # Plot 3: Place Attachment vs Returns Home
        ax3 = fig.add_subplot(gs[0, 2])
        sns.regplot(data=results_df, x='attachment_level', y='FractionMovesHome', ax=ax3)
        ax3.set_title('Place Attachment vs Returns Home')
        
        # Plot 4: Correlation Heatmap
        ax4 = fig.add_subplot(gs[1, :])
        correlation_matrix = results_df[available_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', center=0, ax=ax4)
        ax4.set_title('Correlation Heatmap of Key Metrics')
        
        plt.tight_layout()
        return fig

    def analyze_migration_patterns(self, results_df):
        """Analyze and visualize migration patterns."""
        # Check if required columns exist
        if 'attachment_level' not in results_df.columns:
            print("Warning: attachment_level column missing")
            return plt.figure()  # Return empty figure
            
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, figure=fig)
        
        # Plot 1: Migration Flow by Attachment Level
        ax1 = fig.add_subplot(gs[0, 0])
        sns.lineplot(data=results_df, x='attachment_level', y='total_moves', ax=ax1)
        ax1.set_title('Migration Flow by Attachment Level')
        ax1.set_xlabel('Place Attachment')
        ax1.set_ylabel('Total Moves')
        
        # Plot 2: Distribution of Move Distances
        ax2 = fig.add_subplot(gs[0, 1])
        if 'avg_move_distance' in results_df.columns:
            sns.histplot(data=results_df, x='avg_move_distance', kde=True, ax=ax2)
        ax2.set_title('Distribution of Move Distances')
        ax2.set_xlabel('Average Move Distance')
        ax2.set_ylabel('Count')
        
        # Plot 3: Attachment vs Migration Decision Type
        ax3 = fig.add_subplot(gs[1, 0])
        if all(col in results_df.columns for col in ['pConsumat', 'total_moves']):
            sns.boxplot(data=results_df, x='pConsumat', y='total_moves', 
                       hue='attachment_level', ax=ax3)
        ax3.set_title('Migration by Decision Type and Attachment')
        ax3.set_xlabel('Probability of Consumat Decision')
        ax3.set_ylabel('Total Moves')
        
        # Plot 4: Network Effects and Attachment
        ax4 = fig.add_subplot(gs[1, 1])
        if 'mean_pInitiate' in results_df.columns:
            scatter = sns.scatterplot(data=results_df, x='mean_pInitiate', y='total_moves', 
                                    size='attachment_level', hue='attachment_level',
                                    sizes=(50, 200), ax=ax4)
            if scatter.legend_ is not None:
                scatter.legend_.set_title('Place Attachment')
        ax4.set_title('Network Effects and Place Attachment')
        ax4.set_xlabel('Mean Initiate Probability')
        ax4.set_ylabel('Total Moves')
        
        plt.tight_layout()
        return fig

    def analyze_random_forest(self, results_df):
        """Generate Random Forest analysis plots similar to MATLAB implementation."""
        from sklearn.ensemble import RandomForestRegressor
        import matplotlib.pyplot as plt
        
        # Only use parameters that exist in the results
        available_params = [param for param in [
            'mean_attachment', 'sd_attachment', 'attachment_form_rate', 
            'attachment_decay_rate', 'initial_attachment', 'mean_pInitiate', 
            'sd_pInitiate', 'pConsumat'
        ] if param in results_df.columns]
        
        available_metrics = [metric for metric in [
            'total_moves', 'avg_move_distance', 'avg_satisfaction', 
            'timeStepsAwayFromHome', 'fractionMovesHome', 'avg_utility_change'
        ] if metric in results_df.columns]
        
        if not available_params or not available_metrics:
            print("Warning: Not enough parameters or metrics for random forest analysis")
            return None
        
        # Prepare data
        X = results_df[available_params].values
        importance_scores = {}
        rf_models = {}
        
        # Colors for parameter types
        pa_params = ['mean_attachment', 'sd_attachment', 'attachment_form_rate', 
                    'attachment_decay_rate', 'initial_attachment']
        param_colors = ['#3399CC' if param in pa_params else '#CC6644' 
                       for param in available_params]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate plots for each metric
        for metric in available_metrics:
            if metric not in results_df.columns:
                continue
                
            y = results_df[metric].values
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5)
            rf.fit(X, y)
            
            # Store model and importance scores
            rf_models[metric] = rf
            importance_scores[metric] = rf.feature_importances_
            
            # Plot variable importance
            plt.figure(figsize=(12, 8))
            sorted_idx = np.argsort(rf.feature_importances_)
            pos = np.arange(len(sorted_idx)) + .5
            
            # Create bar plot with color coding
            bars = plt.barh(pos, rf.feature_importances_[sorted_idx])
            for idx, bar in enumerate(bars):
                bar.set_color(param_colors[sorted_idx[idx]])
                
            plt.yticks(pos, np.array(available_params)[sorted_idx])
            plt.xlabel('Relative Importance')
            plt.title(f'Variable Importance for {metric}')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#3399CC', label='Place Attachment Parameters'),
                Patch(facecolor='#CC6644', label='Other Parameters')
            ]
            plt.legend(handles=legend_elements)
            
            plt.tight_layout()
            plt.savefig(f'rf_importance_{metric}_{timestamp}.png')
            plt.close()
        
        # Generate heatmap if we have data
        if importance_scores:
            plt.figure(figsize=(14, 8))
            importance_matrix = np.array([importance_scores[m] for m in available_metrics])
            sns.heatmap(importance_matrix, xticklabels=available_params, 
                       yticklabels=available_metrics,
                       cmap='YlOrRd', annot=True, fmt='.2f')
            plt.title('Parameter Importance Across Metrics')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'rf_importance_heatmap_{timestamp}.png')
            plt.close()
        
        return {
            'models': rf_models,
            'importance_scores': importance_scores,
            'input_params': available_params,
            'metrics': available_metrics
        }

    def generate_report(self):
        """Generate a summary report of the experiments."""
        report = pd.DataFrame(self.metrics)
        
        # Calculate correlations
        correlations = report.corr()['attachment_level'].sort_values(ascending=False)
        
        # Create summary statistics
        summary = report.describe()
        
        return {
            'correlations': correlations,
            'summary_stats': summary,
            'raw_data': report
        }
    
    def cleanup(self):
        """Close NetLogo connection and clean up resources."""
        try:
            if hasattr(self, 'netlogo'):
                self.netlogo.kill_workspace()
                self.netlogo = None
            plt.close('all')  # Close all matplotlib figures
        except Exception as e:
            print(f"Error during cleanup: {e}")
            
    def _generate_param_combinations(self):
        """Generate all parameter combinations for sweep."""
        param_names = list(self.params.keys())
        param_values = list(self.params.values())
        combinations = list(itertools.product(*param_values))
        return [dict(zip(param_names, combo)) for combo in combinations]
    
    def analyze_results(self, results_df):
        """Analyze parameter sweep results."""
        # Group by parameters and calculate statistics
        summary = results_df.groupby(list(self.params.keys())).agg({
            'avg_move_distance': ['mean', 'std'],
            'total_moves': ['mean', 'std'],
            'avg_income_change': ['mean', 'std'],
            'cyclical_moves': ['mean', 'std']
        }).reset_index()


# Example usage
if __name__ == "__main__":
    try:
        model_path = "./migration_place_attachment_model.nlogo"
        print("Running migration analysis...")
        analyzer = MigrationAnalyzer(model_path)
        
        print("Setup and run experiments")
        analyzer.setup_experiment()
        results_df, rf_results = analyzer.run_experiments(n_iterations=5)
        
        print(f"Analysis complete. Check results in the results_* directory.")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'analyzer' in locals():
            analyzer.cleanup()
