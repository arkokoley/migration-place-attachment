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
            places[place_name] = base_config
            return
            
        num_parts = max(1, int(np.random.normal(self.mean_parts, self.sd_parts)))
        for i in range(num_parts):
            new_prefix = prefix + [i]
            self._generate_places(new_prefix, level + 1, places, base_config)

class MigrationAnalyzer:
    def __init__(self, model_path):
        """Initialize the NetLogo connection and setup basic parameters."""
        self.netlogo = NetLogoLink(gui=True)
        self.netlogo.load_model(model_path)
        self.results = {}
        
    def setup_experiment(self, params=None):
        """Setup experiment parameters with defaults."""
        self.params = params or {
            'attachment_levels': np.linspace(0, 1, 5),
            'mean_pInitiate': np.linspace(0.1, 0.5, 3),
            'pConsumat': [0.2, 0.4, 0.6]
        }
        
        # Initialize metrics dictionary
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
            'num_uncertain': []
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
        """Run experiments for all parameter combinations."""
        results = []
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations()
        
        for params in param_combinations:
            for run in range(runs_per_combo):
                # Run single experiment with parameter combination
                run_metrics = self.run_single_experiment(
                    params=params,
                    ticks=num_ticks
                )
                
                # Add run identifier
                run_metrics.update({
                    'run_id': run,
                    **params  # Include parameter values
                })
                
                results.append(run_metrics)
                
        return pd.DataFrame(results)

    def run_single_experiment(self, params, ticks=60):
        """Run single experiment with given parameters."""
        try:
            # Set parameters
            for param, value in params.items():
                self.netlogo.command(f'set {param} {value}')
            
            # Setup and initialize
            self.netlogo.command('setup')
            self.netlogo.command('setup-experiment-metrics')
            
            # Run simulation
            for t in range(ticks):
                self.netlogo.command('go')
                
                # Reset metrics at start of sample interval
                if t % 12 == 0:
                    self.netlogo.command('reset-metrics')
            
            # Collect final metrics using new reporter
            experiment_data = self.netlogo.report('get-experiment-metrics')
            total_moves, mean_distances, mean_satisfaction, mean_utility_change, num_uncertain = experiment_data
            
            return {
                'total_moves': float(total_moves),
                'avg_move_distance': float(mean_distances),
                'avg_satisfaction': float(mean_satisfaction),
                'avg_utility_change': float(mean_utility_change),
                'num_uncertain': float(num_uncertain),
                'success': True,
                **params  # Include experiment parameters
            }
        
        except Exception as e:
            print(f"Error in experiment: {e}")
            return {
                'total_moves': 0,
                'avg_move_distance': 0,
                'avg_satisfaction': 0,
                'avg_utility_change': 0,
                'num_uncertain': 0,
                'success': False,
                **params
            }

    # def run_single_experiment(self, params, ticks=50):
    #     """Run single experiment with given parameters."""
    #     # Set model parameters
    #     for param, value in params.items():
    #         self.netlogo.command(f'set {param} {value}')
        
    #     self.netlogo.command('setup')
        
    #     # Initialize tracking variables
    #     agent_data = {}  # To store agent-specific data
    #     move_distances = []
    #     moves_per_year = []
    #     income_changes = []
    #     qol_changes = []
    #     last_move_times = {}
    #     location_history = {}
        
    #     # Run model and collect data
    #     for t in range(ticks):
    #         self.netlogo.command('go')
            
    #         # Collect metrics
    #         moves = self.netlogo.report('moves')
    #         avg_distance = self.netlogo.report('far_from_home')
    #         total_income_change = self.netlogo.report('total_income_change')
    #         total_qol_change = self.netlogo.report('total_qol_change')
    #         avg_income_change = total_income_change / self.netlogo.report('count people')
    #         avg_qol_change = total_qol_change / self.netlogo.report('count people')

    #         move_distances.append(avg_distance)
    #         # Track yearly moves
    #         if t % 12 == 0:  # Assuming each tick is a month
    #             moves_per_year.append(moves)
            
    #         # Track individual agent metrics
    #         self.netlogo.report('agent-data-csv')
    #         df = pd.read_csv('agent_data.csv')
    #         for index, row in df.iterrows():
    #             agent_id = row['who']
    #             location = row['resident_state_name']
    #             if agent_id not in location_history:
    #                 location_history[agent_id] = []
    #             location_history[agent_id].append(location)
                
    #             # Track time since last move
    #             if location != location_history[agent_id][-2] if len(location_history[agent_id]) > 1 else True:
    #                 last_move_times[agent_id] = t
    
    #         income_changes.append(avg_income_change)
    #         qol_changes.append(avg_qol_change)

        
    #     return {
    #         'move_distances': np.mean(move_distances),
    #         'yearly_moves': np.mean(moves_per_year),
    #         'income_improvement': np.mean(income_changes),
    #         'qol_improvement': np.mean(qol_changes),
    #         'cyclical_moves': self._calculate_cyclical_moves(location_history),
    #         'time_to_next_move': np.mean([t - v for v in last_move_times.values()])
    #     }
    
    def run_experiments(self, n_iterations=5):
        """Run multiple experiments varying attachment levels."""
        for level in self.attachment_levels:
            level_results = []
            for _ in range(n_iterations):
                result = self.run_single_experiment(level)
                level_results.append(result)
            
            # Average results across iterations
            avg_results = {k: np.mean([r[k] for r in level_results]) 
                         for k in level_results[0].keys()}
            print(avg_results)
            # Store results
            for metric in self.metrics:
                if metric != 'attachment_level':
                    self.metrics[metric].append(avg_results[metric])
                else:
                    self.metrics[metric].append(level)
    
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
    
    def visualize_results(self):
        """Create visualizations of experimental results."""
        results_df = pd.DataFrame(self.metrics)
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Migration Patterns vs Place Attachment')
        
        # Plot relationships
        metrics_to_plot = [
            ('move_distances', 'Average Move Distance'),
            ('yearly_moves', 'Yearly Moves'),
            ('cyclical_moves', 'Proportion of Cyclical Moves'),
            ('time_to_next_move', 'Average Time to Next Move'),
            ('income_improvement', 'Income Improvement'),
            ('qol_improvement', 'Quality of Life Improvement')
        ]
        
        for (metric, title), ax in zip(metrics_to_plot, axes.flat):
            sns.scatterplot(data=results_df, x='attachment_level', y=metric, ax=ax)
            sns.regplot(data=results_df, x='attachment_level', y=metric, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Place Attachment Level')
        
        plt.tight_layout()
        return fig
    
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
        """Close NetLogo connection."""
        try:
            self.netlogo.kill_workspace()
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
    model_path = "./migration_place_attachment_model.nlogo"
    print("Running migration analysis...")
    analyzer = MigrationAnalyzer(model_path)
    
    print("Setup and run experiments")
    analyzer.setup_experiment(attachment_levels=np.linspace(0, 1, 10))
    analyzer.run_experiments(n_iterations=10)
    
    print("Generate visualizations and report")
    fig = analyzer.visualize_results()
    report = analyzer.generate_report()
    
    print("Save results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(f'migration_patterns_{timestamp}.png')
    report['raw_data'].to_csv(f'migration_results_{timestamp}.csv')
    
    # Cleanup
    analyzer.cleanup()
