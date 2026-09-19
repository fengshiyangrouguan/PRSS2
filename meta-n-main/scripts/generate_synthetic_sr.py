#!/usr/bin/env python
"""Generate synthetic symbolic regression problems for testing.

Creates self-contained problem directories that follow the exact OpenEvolve
evaluator contract (evaluate(program_path) -> dict with combined_score).

Usage:
    python scripts/generate_synthetic_sr.py [output_dir]

Default output: data/openevolve/examples/symbolic_regression/problems/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

PROBLEMS = [
    {
        "name": "synth_sinx",
        "n_features": 1,
        "desc": "Discover the relationship between horizontal position and signal amplitude.",
        "input_vars": ["x"],
        "input_descs": ["horizontal position"],
        "output_var": "y",
        "output_desc": "signal amplitude",
        "generate": lambda rng, n: _gen_sinx(rng, n),
    },
    {
        "name": "synth_poly2d",
        "n_features": 2,
        "desc": "Discover the relationship between two spatial coordinates and a response surface.",
        "input_vars": ["x1", "x2"],
        "input_descs": ["first coordinate", "second coordinate"],
        "output_var": "y",
        "output_desc": "response surface value",
        "generate": lambda rng, n: _gen_poly2d(rng, n),
    },
    {
        "name": "synth_interaction3d",
        "n_features": 3,
        "desc": "Discover the relationship between three physical quantities and a combined signal.",
        "input_vars": ["x1", "x2", "x3"],
        "input_descs": ["amplitude factor", "frequency input", "linear offset"],
        "output_var": "y",
        "output_desc": "combined signal",
        "generate": lambda rng, n: _gen_interaction3d(rng, n),
    },
]


def _gen_sinx(rng, n):
    x = rng.uniform(-3, 3, (n, 1))
    y = 2.0 * np.sin(1.5 * x[:, 0]) + 0.5 + 0.05 * rng.randn(n)
    return x, y


def _gen_poly2d(rng, n):
    x = rng.uniform(-2, 2, (n, 2))
    y = 0.8 * x[:, 0] ** 2 + 1.2 * x[:, 1] ** 2 - 0.5 * x[:, 0] * x[:, 1] + 0.05 * rng.randn(n)
    return x, y


def _gen_interaction3d(rng, n):
    x = rng.uniform(-2, 2, (n, 3))
    y = 1.5 * x[:, 0] * np.sin(2.0 * x[:, 1]) + 0.7 * x[:, 2] + 0.05 * rng.randn(n)
    return x, y


# ---------------------------------------------------------------------------
# File templates (matching OpenEvolve's data_api.py output exactly)
# ---------------------------------------------------------------------------

def make_evaluator(n_features: int) -> str:
    """Generate evaluator.py matching the exact OpenEvolve SR evaluator contract."""
    return f'''"""
Evaluator for a symbolic regression model.
It assesses a model program based on its performance on training data.
The model's `func` is expected to take a matrix X of inputs.
"""
import os
import sys
import time
import traceback
import importlib.util
import numpy as np
from scipy.optimize import minimize
import concurrent.futures

# Expected number of input features for the model's func
NUM_INPUT_FEATURES_EXPECTED = {n_features}
# Expected number of parameters for the initial model
MODEL_NUM_PARAMS_EXPECTED = 10

# Paths to data (relative to this evaluator's directory)
X_TRAIN_EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "X_train_for_eval.npy")
Y_TRAIN_EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "y_train_for_eval.npy")


def run_with_timeout(func, args=(), kwargs={{}}, timeout_seconds=5):
    """Execute a function with a timeout."""
    if timeout_seconds is None or timeout_seconds <= 0:
        return func(*args, **kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            func_name = getattr(func, '__name__', 'Unnamed function')
            raise TimeoutError(f"Function {{func_name}} timed out after {{timeout_seconds}} seconds")


def filter_and_convert_metrics(current_metrics_dict):
    """Filter and convert metrics to appropriate types."""
    filtered_dict = {{}}
    float_metric_keys = ['combined_score', 'negative_mse']

    for key in float_metric_keys:
        if key in current_metrics_dict:
            value = current_metrics_dict[key]
            if value is None:
                continue
            if isinstance(value, (int, float, np.integer, np.floating, bool)):
                try:
                    filtered_dict[key] = float(value)
                except (ValueError, TypeError):
                    pass

    # Preserve error_message for diagnostics (used by meta-n adapter)
    error_msg = current_metrics_dict.get('error_message')
    if error_msg is not None:
        filtered_dict['error'] = str(error_msg)

    return filtered_dict


def objective_function(params, model_func, X_matrix, y_true_vector):
    """Objective function for scipy.optimize.minimize (MSE)."""
    if not callable(model_func):
        return float('inf')

    try:
        predictions = model_func(X_matrix, params)
        if not isinstance(predictions, np.ndarray) or predictions.shape != y_true_vector.shape:
            return float('inf')
    except Exception:
        return float('inf')

    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        return float('inf')

    mse = np.mean((predictions - y_true_vector)**2)
    return mse


def evaluate(program_path):
    """Evaluate a model program on the training data."""
    metrics = {{
        'can_run': 0.0,
        'negative_mse': -1e09,
        'raw_mse_train': float('inf'),
        'mse_train_score': 0.0,
        'num_params': MODEL_NUM_PARAMS_EXPECTED,
        'combined_score': -1e09,
        'error_message': None,
        'optimization_success': False,
        'optimized_params': None
    }}

    # Load training data
    try:
        X_train = np.load(X_TRAIN_EVAL_PATH)
        y_train = np.load(Y_TRAIN_EVAL_PATH)

        if X_train.shape[1] != NUM_INPUT_FEATURES_EXPECTED:
            metrics['error_message'] = f"Loaded X_train has {{X_train.shape[1]}} features, expected {{NUM_INPUT_FEATURES_EXPECTED}}."
            return filter_and_convert_metrics(metrics)

        if X_train.shape[0] != y_train.shape[0]:
            metrics['error_message'] = f"X_train has {{X_train.shape[0]}} samples, y_train has {{y_train.shape[0]}}."
            return filter_and_convert_metrics(metrics)
    except Exception as e:
        metrics['error_message'] = f"Failed to load training data: {{str(e)}}"
        return filter_and_convert_metrics(metrics)

    # Load and test the model function
    func_to_eval = None
    try:
        spec = importlib.util.spec_from_file_location("model_program", program_path)
        if spec is None or spec.loader is None:
            metrics['error_message'] = f"Could not create spec for module at {{program_path}}"
            return filter_and_convert_metrics(metrics)

        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        metrics['can_run'] = 0.2

        if not hasattr(model_module, 'run_search') or not callable(model_module.run_search):
            metrics['error_message'] = "Model program missing callable 'run_search'."
            return filter_and_convert_metrics(metrics)

        func_to_eval = model_module.run_search()

        if not callable(func_to_eval):
            metrics['error_message'] = "'run_search' did not return a callable function."
            return filter_and_convert_metrics(metrics)

        # Test the function with dummy data (seeded for reproducibility)
        _rng = np.random.RandomState(0)
        num_dummy_samples = 5
        dummy_x = _rng.rand(num_dummy_samples, NUM_INPUT_FEATURES_EXPECTED)
        if NUM_INPUT_FEATURES_EXPECTED == 0:
            dummy_x = np.empty((num_dummy_samples, 0))
        dummy_params = _rng.rand(MODEL_NUM_PARAMS_EXPECTED)

        try:
            pred_test = run_with_timeout(func_to_eval, args=(dummy_x, dummy_params), timeout_seconds=5)
            if not isinstance(pred_test, np.ndarray) or pred_test.shape != (num_dummy_samples,):
                metrics['can_run'] = 0.5
                metrics['error_message'] = f"Func test: output shape mismatch. Got {{pred_test.shape if isinstance(pred_test, np.ndarray) else type(pred_test)}}, expected ({{num_dummy_samples}},)."
                return filter_and_convert_metrics(metrics)
            metrics['can_run'] = 1.0
        except TimeoutError as te:
            metrics['can_run'] = 0.5
            metrics['error_message'] = f"Func execution test timed out: {{str(te)}}"
            return filter_and_convert_metrics(metrics)
        except Exception as e:
            metrics['can_run'] = 0.5
            metrics['error_message'] = f"Func execution test failed: {{str(e)}}"
            return filter_and_convert_metrics(metrics)

    except FileNotFoundError:
        metrics['error_message'] = f"Model program file not found: {{program_path}}"
        return filter_and_convert_metrics(metrics)
    except Exception as e:
        metrics['error_message'] = f"Failed to load or test model function: {{str(e)}}"
        return filter_and_convert_metrics(metrics)

    if metrics['can_run'] < 1.0:
        return filter_and_convert_metrics(metrics)

    # Optimize parameters (seeded for reproducibility)
    _opt_rng = np.random.RandomState(42)
    initial_params = _opt_rng.rand(MODEL_NUM_PARAMS_EXPECTED)

    try:
        opt_result = minimize(
            objective_function,
            initial_params,
            args=(func_to_eval, X_train, y_train),
            method='BFGS'
        )

        metrics['raw_mse_train'] = opt_result.fun if np.isfinite(opt_result.fun) else float('inf')
        metrics['optimization_success'] = opt_result.success

        if opt_result.success or hasattr(opt_result, 'x'):
            optimized_params = opt_result.x
        else:
            optimized_params = initial_params

        if not opt_result.success and metrics['error_message'] is None:
            metrics['error_message'] = f"Optimization did not converge: {{opt_result.message if hasattr(opt_result, 'message') else 'Unknown reason'}}"

    except Exception as e:
        metrics['raw_mse_train'] = float('inf')
        metrics['error_message'] = f"Error during optimization: {{str(e)}}"
        optimized_params = initial_params

    metrics['optimized_params'] = optimized_params.tolist() if optimized_params is not None else None

    # Calculate final scores
    if np.isfinite(metrics['raw_mse_train']):
        metrics['negative_mse'] = -metrics['raw_mse_train']
        metrics['mse_train_score'] = -np.log10(metrics['raw_mse_train'] + 1e-9)
    else:
        metrics['mse_train_score'] = 0.0

    metrics['combined_score'] = metrics['mse_train_score']

    return filter_and_convert_metrics(metrics)
'''


def make_initial_program(n_features: int, input_vars: list[str], input_descs: list[str],
                         output_var: str, output_desc: str) -> str:
    """Generate initial_program.py matching OpenEvolve's template."""
    # Build column mapping comments
    mapping_lines = ["# Input variable mapping for x (columns of the input matrix):"]
    if not input_vars:
        mapping_lines.append("#   No input variables (x will be an (n_samples, 0) matrix).")
    else:
        for i, (var, desc) in enumerate(zip(input_vars, input_descs)):
            mapping_lines.append(f"#   x[:, {i}]: {var} ({desc})")
    mapping_str = "\n".join(mapping_lines)

    # Build naive function body
    if n_features > 0:
        body = " + ".join([f"x[:, {i}] * params[{i}]" for i in range(n_features)])
    else:
        body = "np.full(x.shape[0], params[0])"

    input_desc_str = ", ".join(f"{v} ({d})" for v, d in zip(input_vars, input_descs)) if input_vars else "None"

    return f'''"""
Initial program: A naive linear model for symbolic regression.
Target output variable: {output_var} ({output_desc})
Input variables (columns of x): {input_desc_str}
"""
import numpy as np

{mapping_str}

# Parameters will be optimized by BFGS outside this function.
# Number of parameters expected by this model: 10.

# EVOLVE-BLOCK-START

def func(x, params):
    """
    Calculates the model output. Operates on a matrix of samples.

    Args:
        x (np.ndarray): Input variable values, shape (n_samples, {n_features}).
        params (np.ndarray): Parameters, length 10.

    Returns:
        np.ndarray: Predicted output values, shape (n_samples,).
    """
    result = {body}
    return result

# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
'''


def make_config(problem: dict) -> str:
    """Generate config.yaml with problem description."""
    input_desc = ", ".join(f"{v} ({d})" for v, d in
                           zip(problem["input_vars"], problem["input_descs"]))
    return f"""max_iterations: 100
checkpoint_interval: 10
parallel_evaluations: 1

evaluator:
  timeout: 90
  max_retries: 3

prompt:
  system_message: |
    You are an expert at symbolic regression. Your goal is to discover a
    mathematical function that fits the given data.

    {problem["desc"]}

    Input variables (columns of x): {input_desc or "None"}
    Output variable: {problem["output_var"]} ({problem["output_desc"]})
    Number of features: {problem["n_features"]}
    Number of parameters: 10 (optimized externally via BFGS)

    The evaluator will optimize params via BFGS to minimize MSE on
    training data. Your job is to discover the right functional form.
    Try combinations of elementary functions (polynomials, trigonometric,
    exponential, logarithmic) that could explain the data.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/openevolve/examples/symbolic_regression/problems"
    output_dir = Path(output_dir)

    print(f"Generating synthetic symbolic regression problems in {output_dir}")

    for problem in PROBLEMS:
        prob_dir = output_dir / problem["name"]
        prob_dir.mkdir(parents=True, exist_ok=True)

        # Generate data
        rng = np.random.RandomState(42)
        X_train, y_train = problem["generate"](rng, 200)
        X_test, y_test = problem["generate"](np.random.RandomState(123), 50)

        # Save .npy files
        np.save(prob_dir / "X_train_for_eval.npy", X_train)
        np.save(prob_dir / "y_train_for_eval.npy", y_train)
        np.save(prob_dir / "X_test_for_eval.npy", X_test)
        np.save(prob_dir / "y_test_for_eval.npy", y_test)

        # Write evaluator, program, config
        (prob_dir / "evaluator.py").write_text(make_evaluator(problem["n_features"]))
        (prob_dir / "initial_program.py").write_text(
            make_initial_program(
                problem["n_features"], problem["input_vars"], problem["input_descs"],
                problem["output_var"], problem["output_desc"],
            )
        )
        (prob_dir / "config.yaml").write_text(make_config(problem))

        print(f"  {problem['name']}: {problem['n_features']} features, 200 train / 50 test samples")

    print(f"Done — {len(PROBLEMS)} problems generated.")


if __name__ == "__main__":
    main()
