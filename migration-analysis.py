import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pynetlogo import NetLogoLink
import os
from datetime import datetime
import networkx as nx

class MigrationAnalyzer:
    def __init__(self, model_path):
        """Initialize the NetLogo connection and setup basic parameters."""
        self.netlogo = NetLogoLink(gui=False)
        self.netlogo.load_model(model_path)
        self.results = {}
        
    def setup_experiment(self, attachment_levels=np.linspace(0, 1, 5)):
        """Setup experiment parameters."""
        self.attachment_levels = attachment_levels
        self.metrics = {
            'move_distances': [],
            'yearly_moves': [],
            'income_improvement': [],
            'qol_improvement': [],
            'cyclical_moves': [],
            'time_to_next_move': [],
            'attachment_level': []
        }
        
    def run_single_experiment(self, attachment_level, ticks=50):
        """Run a single experiment with given attachment level."""
        # Set model parameters
        self.netlogo.command(f'set mean_attachment {attachment_level}')
        self.netlogo.command('setup')
        
        # Initialize tracking variables
        agent_data = {}  # To store agent-specific data
        move_distances = []
        moves_per_year = []
        income_changes = []
        qol_changes = []
        last_move_times = {}
        location_history = {}
        
        # Run model and collect data
        for t in range(ticks):
            self.netlogo.command('go')
            
            # Collect metrics
            moves = self.netlogo.report('moves')
            avg_distance = self.netlogo.report('far_from_home')
            total_income_change = self.netlogo.report('total_income_change')
            total_qol_change = self.netlogo.report('total_qol_change')
            avg_income_change = total_income_change / self.netlogo.report('count people')
            avg_qol_change = total_qol_change / self.netlogo.report('count people')

            move_distances.append(avg_distance)
            # Track yearly moves
            if t % 12 == 0:  # Assuming each tick is a month
                moves_per_year.append(moves)
            
            # Track individual agent metrics
            self.netlogo.report('agent-data-csv')
            df = pd.read_csv('agent_data.csv')
            for index, row in df.iterrows():
                agent_id = row['who']
                location = row['resident_state_name']
                if agent_id not in location_history:
                    location_history[agent_id] = []
                location_history[agent_id].append(location)
                
                # Track time since last move
                if location != location_history[agent_id][-2] if len(location_history[agent_id]) > 1 else True:
                    last_move_times[agent_id] = t
    
            income_changes.append(avg_income_change)
            qol_changes.append(avg_qol_change)

        
        return {
            'move_distances': np.mean(move_distances),
            'yearly_moves': np.mean(moves_per_year),
            'income_improvement': np.mean(income_changes),
            'qol_improvement': np.mean(qol_changes),
            'cyclical_moves': self._calculate_cyclical_moves(location_history),
            'time_to_next_move': np.mean([t - v for v in last_move_times.values()])
        }
    
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
        
        return cyclical_count / max(1, total_moves)
    
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
        self.netlogo.kill_workspace()

# Example usage
if __name__ == "__main__":
    model_path = "./migration_place_attachment_model.nlogo"
    analyzer = MigrationAnalyzer(model_path)
    
    # Setup and run experiments
    analyzer.setup_experiment(attachment_levels=np.linspace(0, 1, 10))
    analyzer.run_experiments(n_iterations=1)
    
    # Generate visualizations and report
    fig = analyzer.visualize_results()
    report = analyzer.generate_report()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(f'migration_patterns_{timestamp}.png')
    report['raw_data'].to_csv(f'migration_results_{timestamp}.csv')
    
    # Cleanup
    analyzer.cleanup()
