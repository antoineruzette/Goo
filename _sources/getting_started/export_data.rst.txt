.. _export_data:

Export data
=================

The data structure that is exported is a HDF5 file, and is structured as follows:

.. code-block:: text

   frame_001/
   ├── time (float)
   ├── cells/
   │   ├── cell_001/
   │   │   ├── name (attribute)
   │   │   ├── loc (tuple)
   │   │   ├── volume (float)
   │   │   ├── pressure (float)
   │   │   ├── division_frame (float)
   │   │   ├── force_loc (tuple)
   │   │   ├── aspect_ratio (float)
   │   │   ├── sphericity (float)
   │   │   ├── compactness (float)
   │   │   ├── sav_ratio (float)
   │   │   ├── gene_x_conc (float)
   │   │   ├── gene_y_conc (float)
   │   │   ├── ...
   │   │   ├── mol_A_conc (float)
   │   │   ├── mol_B_conc (float)
   │   │   └── ...
   │   └── cell_002/
   │       └── ...
   └── concentration_grid/
      ├── mol_A/
      │   ├── dimensions (dataset)
      │   └── values (dataset)
      └── mol_B/
         ├── dimensions (dataset)
         └── values (dataset)
   frame_002/
   ├── cells/
   │   └── ...
.. _export_data_plot:

Plotting exported data
----------------------

Here's a step-by-step guide to analyze and visualize data from Goo simulations. We'll break down the process into reusable functions and demonstrate their use with cell volume analysis.

First, let's import the required libraries:

.. code-block:: python

   import h5py
   import pandas as pd
   import matplotlib.pyplot as plt
   from collections import defaultdict
   import numpy as np
   from typing import Dict, List, Tuple

Step 1: Reading Data from HDF5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we'll create functions to read and organize data from the HDF5 file:

.. code-block:: python

   def get_sorted_frames(h5_file: h5py.File) -> List[str]:
       """Get frame names from HDF5 file and sort them numerically."""
       frames = [key for key in h5_file.keys() if key.startswith("frame_")]
       return sorted(frames, key=lambda x: int(x.split("_")[1]))

   def extract_cell_property(h5_file: h5py.File, property_name: str) -> pd.DataFrame:
       """
       Extract a specific property for all cells across all frames.
       
       Args:
           h5_file: Open HDF5 file
           property_name: Name of the property to extract (e.g., 'volume', 'pressure')
           
       Returns:
           DataFrame with frames as index and cells as columns
       """
       data_over_time = defaultdict(list)
       frames = get_sorted_frames(h5_file)
       
       for frame in frames:
           cells_group = h5_file[frame]["cells"]
           
           for cell_name in cells_group.keys():
               try:
                   value = cells_group[cell_name][property_name][()]
                   data_over_time[cell_name].append(value)
               except KeyError:
                   # Handle missing properties gracefully
                   data_over_time[cell_name].append(np.nan)
       
       # Convert to DataFrame
       frame_indices = [int(f.split("_")[1]) for f in frames]
       return pd.DataFrame(data_over_time, index=frame_indices)

Step 2: Analysis Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, let's create functions for analyzing the data:

.. code-block:: python

   def calculate_theoretical_growth(
       initial_value: float,
       time_points: np.ndarray,
       growth_rate: float = 1.0,
       max_value: float = 100.0
   ) -> np.ndarray:
       """
       Calculate theoretical linear growth curve.
       
       Args:
           initial_value: Starting value
           time_points: Array of time points
           growth_rate: Growth rate (default: 1.0 µm³/min for volume)
           max_value: Maximum allowed value
           
       Returns:
           Array of theoretical values
       """
       return np.minimum(initial_value + time_points * growth_rate, max_value)

   def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
       """Calculate Root Mean Square Error between actual and predicted values."""
       return np.sqrt(np.mean((actual - predicted)**2))

Step 3: Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, let's create functions for plotting:

.. code-block:: python

   def setup_plot(figsize: Tuple[int, int] = (12, 8)) -> None:
       """Set up the plot with standard formatting."""
       plt.figure(figsize=figsize)
       plt.grid(True, linestyle='--', alpha=0.5)

   def plot_cell_property(
       df: pd.DataFrame,
       property_name: str,
       units: str,
       show_theoretical: bool = True,
       growth_rate: float = 1.0,
       max_value: float = 100.0
   ) -> None:
       """
       Create a plot of cell property over time.
       
       Args:
           df: DataFrame with property values
           property_name: Name of the property being plotted
           units: Units for the y-axis
           show_theoretical: Whether to show theoretical growth curve
           growth_rate: Growth rate for theoretical curve
           max_value: Maximum value for theoretical curve
       """
       setup_plot()
       
       # Plot each cell's data
       for cell in df.columns:
           plt.plot(df.index, df[cell], label=cell)
       
       # Add theoretical curve if requested
       if show_theoretical:
           theoretical = calculate_theoretical_growth(
               df.iloc[0,0], df.index, growth_rate, max_value
           )
           plt.plot(df.index, theoretical,
                   label="theoretical linear growth",
                   linestyle="--", color="black", alpha=0.5)
           
           # Calculate and display RMSE
           rmses = [calculate_rmse(df[col], theoretical) for col in df.columns]
           avg_rmse = np.mean(rmses)
           plt.text(0.02, 0.98, f'Avg RMSE: {avg_rmse:.2f}',
                   transform=plt.gca().transAxes,
                   verticalalignment='top', fontsize=12)
       
       # Customize plot
       plt.xlabel("Time (min)", fontsize=16)
       plt.ylabel(f"{property_name} ({units})", fontsize=16)
       plt.legend(title="", fontsize=10, loc="lower right")
       plt.tight_layout()

Step 4: Putting It All Together
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's how to use these functions to analyze cell volumes:

.. code-block:: python

   # Path to your HDF5 data file
   h5_path = "path/to/your/data.h5"

   # Read and plot cell volumes
   with h5py.File(h5_path, "r") as f:
       # Extract volume data
       volume_df = extract_cell_property(f, "volume")
       
       # Create the plot
       plot_cell_property(
           df=volume_df,
           property_name="Volume",
           units="µm³",
           show_theoretical=True,
           growth_rate=1.0,
           max_value=100.0
       )
       plt.show()

This modular approach makes it easy to:

1. Extract different cell properties by changing the property name:

   .. code-block:: python

      pressure_df = extract_cell_property(f, "pressure")
      plot_cell_property(pressure_df, "Pressure", "Pa")

2. Customize plots with different parameters:

   .. code-block:: python

      plot_cell_property(
          volume_df, "Volume", "µm³",
          show_theoretical=False  # Skip theoretical comparison
      )

3. Analyze multiple properties in sequence:

   .. code-block:: python

      properties = ["volume", "pressure", "aspect_ratio"]
      units = ["µm³", "Pa", "ratio"]
      
      for prop, unit in zip(properties, units):
          df = extract_cell_property(f, prop)
          plot_cell_property(df, prop.replace("_", " ").title(), unit)
          plt.show()

The functions handle common issues like:
- Missing data points
- Proper scientific notation
- Consistent plot formatting
- Error calculation
- Theoretical model comparison

