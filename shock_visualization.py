import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import json
import numpy as np

def visualize_shock_effects(data_path: str, output_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Create visualizations comparing individual and place shock effects across attachment levels.
    
    Args:
        data_path: Path to JSON/CSV file containing shock effect data
        output_path: Optional path to save the visualization. If None, returns the figure object
    
    Expected data format (JSON):
    {
        "experiments": [
            {
                "shock_type": "individual"|"place",
                "mean_attachment": float,
                "avg_satisfaction_shocked": float,
                "avg_satisfaction_not_shocked": float,
                "avg_moves_shocked": float,
                "avg_moves_not_shocked": float
            },
            ...
        ]
    }
    """
    # Load data
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data['experiments'])
    else:
        df = pd.read_csv(data_path)

    # Validate required columns
    required_cols = [
        'shock_type', 'mean_attachment',
        'avg_satisfaction_shocked', 'avg_satisfaction_not_shocked',
        'avg_moves_shocked', 'avg_moves_not_shocked'
    ]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Need: {required_cols}")

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Calculate effects for each shock type
    for shock_type in ['individual', 'place']:
        shock_data = df[df['shock_type'] == shock_type].copy()
        if shock_data.empty:
            continue

        # Calculate effect sizes
        shock_data['satisfaction_effect'] = shock_data['avg_satisfaction_shocked'] - shock_data['avg_satisfaction_not_shocked']
        shock_data['moves_effect'] = shock_data['avg_moves_shocked'] - shock_data['avg_moves_not_shocked']

        # Group by attachment level and calculate means and standard errors
        grouped = shock_data.groupby('mean_attachment', as_index=False).agg({
            'satisfaction_effect': ['mean', 'std'],
            'moves_effect': ['mean', 'std']
        })

        # Plot satisfaction effects
        ax1.errorbar(
            grouped['mean_attachment'], 
            grouped[('satisfaction_effect', 'mean')],
            yerr=grouped[('satisfaction_effect', 'std')],
            label=f'{shock_type.title()} Shock',
            marker='o',
            capsize=5
        )

        # Plot movement effects
        ax2.errorbar(
            grouped['mean_attachment'], 
            grouped[('moves_effect', 'mean')],
            yerr=grouped[('moves_effect', 'std')],
            label=f'{shock_type.title()} Shock',
            marker='o',
            capsize=5
        )

    # Customize plots
    ax1.set_title('Shock Effects on Satisfaction by Attachment Level')
    ax1.set_xlabel('Mean Attachment Level')
    ax1.set_ylabel('Satisfaction Effect\n(Shocked - Not Shocked)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Shock Effects on Movement by Attachment Level')
    ax2.set_xlabel('Mean Attachment Level')
    ax2.set_ylabel('Movement Effect\n(Shocked - Not Shocked)')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
        return None
    return fig

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize shock effects from experimental data')
    parser.add_argument('data_path', help='Path to input JSON/CSV file')
    parser.add_argument('--output', '-o', help='Path to save visualization (optional)')
    args = parser.parse_args()

    visualize_shock_effects(args.data_path, args.output)
