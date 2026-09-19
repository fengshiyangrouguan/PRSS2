```python
# EVOLVE-BLOCK-START
"""
2D Affine Transform

Apply a 2D affine transformation to an input image (2D array). The transformation is defined by a 2x3 matrix which combines rotation, scaling, shearing, and translation. This task uses cubic spline interpolation (order=3) and handles boundary conditions using the 'constant' mode (padding with 0).

Input:
A dictionary with keys:
  - "image": An n x n array of floats (in the range [0.0, 255.0]) representing the input image.
  - "matrix": A 2x3 array representing the affine transformation matrix.

Example input:
{
    "image": [
        [100.0, 150.0, 200.0],
        [50.0, 100.0, 150.0],
        [0.0, 50.0, 100.0]
    ],
    "matrix": [
        [0.9, -0.1, 1.5],
        [0.1, 1.1, -2.0]
    ]
}

Output:
A dictionary with key:
  - "transformed_image": The transformed image array of shape (n, n).

Example output:
{
    "transformed_image": [
        [88.5, 141.2, 188.0],
        [45.1, 99.8, 147.3],
        [5.6, 55.2, 103.1]
    ]
}

Category: signal_processing

OPTIMIZATION OPPORTUNITIES:
Consider these algorithmic improvements for significant performance gains:
- Lower-order interpolation: Try order=0 (nearest) or order=1 (linear) vs default order=3 (cubic)
  Linear interpolation (order=1) often provides best speed/quality balance with major speedups
- Precision optimization: float32 often sufficient vs float64, especially with lower interpolation orders
- Separable transforms: Check if the transformation can be decomposed into separate x and y operations
- Cache-friendly memory access patterns: Process data in blocks to improve cache utilization
- JIT compilation: Use JAX or Numba for numerical operations that are Python-bottlenecked
- Direct coordinate mapping: Avoid intermediate coordinate calculations for simple transforms
- Hardware optimizations: Leverage SIMD instructions through vectorized operations
- Batch processing: Process multiple images or regions simultaneously for amortized overhead

This is the initial implementation that will be evolved by OpenEvolve.
The solve method will be improved through evolution.
"""
import logging
import random
import numpy as np
import scipy.ndimage
from typing import Any, Dict, List, Optional

class AffineTransform2D:
    """
    Improved implementation of affine_transform_2d task.
    This will be evolved by OpenEvolve to improve performance and correctness.
    """
    
    def __init__(self):
        """Initialize the AffineTransform2D."""
        self.order = 3
        self.mode = "constant"  # Or 'nearest', 'reflect', 'mirror', 'wrap'
    
    def solve(self, problem):
        """
        Solve the affine_transform_2d problem.
        
        Args:
            problem: Dictionary containing problem data specific to affine_transform_2d
                   
        Returns:
            The solution in the format expected by the task
        """
        try:
            """
            Solves the 2D affine transformation problem using scipy.ndimage.affine_transform.

            :param problem: A dictionary representing the problem.
            :return: A dictionary with key "transformed_image":
                     "transformed_image": The transformed image as an array.
            """
            image = problem["image"]
            matrix = problem["matrix"]

            # Perform affine transformation
            try:
                # output_shape can be specified, default is same as input
                transformed_image = scipy.ndimage.affine_transform(
                    image, matrix, order=self.order, mode=self.mode
                )
            except Exception as e:
                logging.error(f"scipy.ndimage.affine_transform failed: {e}")
                # Return an empty list to indicate failure? Adjust based on benchmark policy.
                return {"transformed_image": []}

            solution = {"transformed_image": transformed_image}
            return solution
            
        except Exception as e:
            logging.error(f"Error in solve method: {e}")
            raise e
    
    def is_solution(self, problem, solution):
        """
        Check if the provided solution is valid.
        
        Args:
            problem: The original problem
            solution: The proposed solution
                   
        Returns:
            True if the solution is valid, False otherwise
        """
        try:
            """
            Check if the provided affine transformation solution is valid.

            Checks structure, dimensions, finite