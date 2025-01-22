from dataclasses import dataclass
from multiprocessing import Pool
import numpy as np
import pandas as pd
from migration_analysis import MigrationAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

class ExperimentConfig:
    # Landscape parameters
    admin_levels: list = (2, 3)
    mean_parts: list = (2, 3, 4)
    sd_parts: list = (0.5, 1.0)
    
    # Agent parameters
    population_sizes: list = (100, 200, 500)
    attachment_levels: list = np.linspace(0, 1, 5)
    
    def get_parameter_combinations(self):
        return [(a, m, s, p) for a in self.admin_levels 
                            for m in self.mean_parts
                            for s in self.sd_parts
                            for p in self.population_sizes]

def initialize_worker(model_path):
    """Initialize NetLogo instance for each worker"""
    global analyzer
    analyzer = MigrationAnalyzer(model_path)

def run_single_experiment(params):
    """Run single experiment with given parameters"""
    try:
        return analyzer.run_single_experiment(params)
    except Exception as e:
        print(f"Error in experiment {params}: {e}")
        return None


class ComplexityExperiment:
    def __init__(self, model_path, config=None):
        # self.analyzer = MigrationAnalyzer(model_path)
        self.config = config or ExperimentConfig()
        self.results = []
        self.model_path = model_path
        
    def run_experiments(self, runs_per_combo=1):
        for admin, mean_p, sd_p, pop in self.config.get_parameter_combinations():
            for pa in self.config.attachment_levels:
                for run in range(runs_per_combo):
                    result = self.analyzer.run_single_experiment({
                        'admin_levels': admin,
                        'mean_parts': mean_p,
                        'sd_parts': sd_p,
                        'initial_population': pop,
                        'mean_attachment': pa
                    })
                    result.update({
                        'admin_levels': admin,
                        'mean_parts': mean_p,
                        'sd_parts': sd_p,
                        'population': pop,
                        'attachment': pa,
                        'run': run
                    })
                    self.results.append(result)
                    
    def analyze_results(self):
        """Analyze experiment results with proper reshaping"""
        # df = pd.DataFrame(self.results)
        # df.to_json('results.json')
        df = pd.read_json('results.json')
        
        # Ensure attachment levels are properly formatted
        df['attachment'] = pd.to_numeric(df['attachment'])
        
        # Group by complexity parameters
        grouped = df.groupby(['admin_levels', 'mean_parts', 'sd_parts'])
        
        # Calculate correlations between PA and outcomes
        correlations = grouped.apply(lambda x: pd.Series({
            'moves_corr': x['total_moves'].corr(x['attachment']),
            'distance_corr': x['avg_move_distance'].corr(x['attachment']),
            'satisfaction_corr': x['avg_satisfaction'].corr(x['attachment']),
            'utility_corr': x['avg_utility_change'].corr(x['attachment'])
        }))
        
        return correlations.reset_index()
    
    def plot_complexity_effects(self):
        """Visualize PA effects across complexity levels"""
        effects = self.analyze_results()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics = ['moves_corr', 'distance_corr', 'satisfaction_corr', 'utility_corr']
        titles = ['Moves', 'Move Distance', 'Satisfaction', 'Utility Change']
        
        # Create heatmap for each metric
        for idx, (ax, metric, title) in enumerate(zip(axes.flat, metrics, titles)):
            # Reshape data for heatmap
            pivot_data = effects.pivot_table(
                values=metric,
                index='mean_parts',
                columns='admin_levels',
                aggfunc='mean'
            )
            
            # Generate heatmap
            sns.heatmap(
                pivot_data, 
                ax=ax,
                center=0,
                cmap='RdBu_r',
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation'}
            )
            
            ax.set_title(f'PA Effect on {title}')
            ax.set_xlabel('Admin Levels')
            ax.set_ylabel('Mean Parts')
        
        plt.tight_layout()
        return fig

    def run_parallel_experiments(self, num_processes=4, runs_per_combo=3):
        """Run experiments in parallel"""
        # Generate all parameter combinations
        param_combinations = []
        for admin, mean_p, sd_p, pop in self.config.get_parameter_combinations():
            for pa in self.config.attachment_levels:
                for run in range(runs_per_combo):
                    param_combinations.append({
                        'admin_levels': admin,
                        'mean_parts': mean_p,
                        'sd_parts': sd_p,
                        'initial_population': pop,
                        'mean_attachment': pa
                    })

        # Run experiments in parallel
        with Pool(num_processes, initializer=initialize_worker, 
                 initargs=(self.model_path,)) as pool:
            results = pool.map(run_single_experiment, param_combinations)
            
        self.results = [r for r in results if r is not None]
        return pd.DataFrame(self.results)

    def analyze_parameter_effects(self):
        """Find parameter combinations with strongest PA effects"""
        df = pd.DataFrame(self.results)
        
        # Group by parameter combinations
        params = ['admin_levels', 'mean_parts', 'sd_parts', 'initial_population']
        effects = []
        
        for _, group in df.groupby(params):
            # Calculate effect sizes for each metric
            pa_effects = {
                'moves': np.corrcoef(group['total_moves'], group['attachment'])[0,1],
                'distance': np.corrcoef(group['avg_move_distance'], group['attachment'])[0,1],
                'satisfaction': np.corrcoef(group['avg_satisfaction'], group['attachment'])[0,1],
                'utility': np.corrcoef(group['avg_utility_change'], group['attachment'])[0,1]
            }
            
            # Calculate composite effect size
            effect_size = np.mean([abs(v) for v in pa_effects.values()])
            
            # Calculate consistency (std dev of effects)
            effect_consistency = np.std([v for v in pa_effects.values()])
            
            effects.append({
                **dict(zip(params, group[params].iloc[0])),
                'composite_effect': effect_size,
                'effect_consistency': effect_consistency,
                **pa_effects
            })
        
        effects_df = pd.DataFrame(effects)
        
        # Find optimal parameter combinations
        best_params = effects_df.nlargest(5, 'composite_effect')
        
        # Visualize results
        plt.figure(figsize=(12, 6))
        
        # Plot composite effects
        sns.scatterplot(
            data=effects_df,
            x='mean_parts',
            y='composite_effect',
            size='effect_consistency',
            hue='admin_levels',
            style='initial_population',
            alpha=0.7
        )
        
        plt.title('Parameter Effects on Place Attachment Impact')
        plt.xlabel('Mean Parts')
        plt.ylabel('Composite Effect Size')
        
        return best_params, plt.gcf()

if __name__ == "__main__":
  # Initialize with your NetLogo model path
  model_path = "./migration_place_attachment_model.nlogo"
  exp = ComplexityExperiment(model_path)

  # Run experiments with desired number of repeats
#   exp.run_parallel_experiments(num_processes=8, runs_per_combo=1)
  exp.results = pd.read_json('results.json')
  # Analyze the data
  summary = exp.analyze_results()
  print(summary)

  # Plot and display
  exp.plot_complexity_effects()
  # Find optimal parameters
  best_params, fig = exp.analyze_parameter_effects()
  print("\nTop Parameter Combinations:")
  print(best_params)
  plt.show()