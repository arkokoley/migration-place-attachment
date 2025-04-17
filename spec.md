# Migration Place Attachment Model Specification
Version: 1.0.0
Last Updated: 2024-01-27

## Project Overview

This project simulates migration patterns with place attachment using NetLogo and Python. It explores how psychological attachment to places influences migration decisions.

## Project Components

### 1. NetLogo Model (`migration_place_attachment_model.nlogo`)
- Agent-based simulation of migration patterns
- Models place attachment dynamics
- Configurable parameters for experiments
- Exports metrics for analysis

### 2. Migration Analysis (`migration_analysis.py`) 
- Main Python wrapper for running experiments
- Parameter sweeps and visualization
- Random forest analysis integration
- Metrics collection and export

### 3. Complexity Experiment (`ex1.py`)
- Specialized complexity analysis
- Parallel experiment execution
- Configuration:
  - Landscape parameters (admin_levels, mean_parts, sd_parts)
  - Population parameters (population_sizes, attachment_levels)
- Key Features:
  - Parallel processing with multiprocessing Pool
  - Parameter effect analysis
  - Effect size calculations for:
    - Movement patterns
    - Distance metrics
    - Satisfaction levels
    - Utility changes
  - Visualization of complexity effects
  - CSV export of results

### 4. Random Forest Analysis (`random_forrest.m`)
- Advanced statistical analysis in MATLAB
- Variable importance scoring
- Visualization generation

## Analysis Pipeline

1. Configure experiment parameters
2. Run parallel simulations (ex1.py)
3. Collect and process metrics
4. Generate visualizations
5. Export results for random forest analysis
6. Analyze parameter importance

## Technical Details

### Parameters

1. Place Attachment Parameters:
   - mean_attachment (0-1)
   - sd_attachment (0.1-0.5)
   - attachment_form_rate (0.01-0.05)
   - attachment_decay_rate (0.01-0.05)
   - initial_attachment (0.2-0.8)

2. Network Parameters:
   - mean_pInitiate (0.1-0.5)
   - sd_pInitiate (0.05-0.2)
   - pConsumat (0.2-0.6)

### Metrics Tracked

1. Movement Metrics:
   - total_moves
   - avg_move_distance
   - yearly_moves
   - cyclical_moves
   - FractionMovesHome
   - TimeStepsAwayFromHome

2. Satisfaction Metrics:
   - avg_satisfaction
   - qol_improvement
   - income_improvement
   - utility_changes

### Working Solutions

1. Memory Management:
   ```python
   torch.cuda.empty_cache()
   compressed_model = compressed_model.cpu()
   ```

2. Parameter Space Reduction:
   ```python
   'mean_attachment': np.linspace(0, 1, 3),  # Reduced from 5 to 3
   'sd_attachment': np.linspace(0.1, 0.5, 2),  # Reduced from 3 to 2
   ```

3. Error Handling:
   ```python
   try:
       final_metrics = self.netlogo.report('get-experiment-metrics')
       if final_metrics is not None and len(final_metrics) >= 5:
           # Handle metrics
   except Exception as e:
       print(f"Error getting final metrics: {e}")
   ```

### Known Issues

1. Array Truth Value Ambiguity:
   - Problem: NumPy arrays in boolean context
   - Solution: Use explicit NaN checking and proper array handling

2. Memory Usage:
   - Problem: Large parameter combinations
   - Solution: Reduced parameter space and iterations

3. Data Type Consistency:
   - Problem: Mixed types in metrics
   - Solution: Explicit type conversion and validation

4. NetLogo Communication:
   - Problem: Empty lists and null values
   - Solution: Default values and error handling

## Development Notes

### Testing Protocol

1. Parameter Sweep Testing:
   - Verify parameter combinations
   - Check metric collection
   - Validate visualization generation

2. Memory Testing:
   - Monitor RAM usage
   - Track VRAM allocation
   - Check for memory leaks

3. Data Validation:
   - Verify metric calculations
   - Ensure data type consistency
   - Validate statistical analysis

### Best Practices

1. Always include error handling for NetLogo interactions
2. Clear memory after each experiment
3. Use consistent data types throughout analysis
4. Validate metrics before visualization
5. Save intermediate results regularly

## Environment Setup

```requirements
python>=3.9
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
seaborn>=0.11
pynetlogo>=0.5
scikit-learn>=1.0
```

### NetLogo Requirements
- NetLogo 6.4.0
- Minimum 4GB RAM
- Java 8 or higher

## Future Improvements

1. Optimization:
   - Batch processing for large parameter sweeps
   - Improved memory management
   - Parallel processing support

2. Analysis:
   - Additional visualization types
   - More statistical measures
   - Enhanced error reporting

3. Features:
   - Multi-GPU support
   - Real-time visualization
   - Advanced network analysis

## For LLMs

When working with this project:

1. Code Structure:
   - NetLogo model handles simulation
   - Python wrapper manages experiments
   - MATLAB provides advanced analysis

2. Common Issues:
   - Handle empty lists from NetLogo
   - Convert NumPy arrays properly
   - Manage memory for large experiments

3. Modification Guidelines:
   - Preserve error handling
   - Maintain type consistency
   - Keep visualization functions modular

4. Tips:
   - Test with small parameter sets first
   - Validate metric calculations
   - Check memory usage regularly

[End of Specification]
