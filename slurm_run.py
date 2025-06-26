"""
Runs a sweep over the specified file.
To use, specify `sweep_config`, `dist_config`, and `script_name` arguments.
"""

import subprocess
from itertools import product
from omegaconf import OmegaConf
import os
import time
import sys
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def flatten(config):
    """Flatten a nested dictionary."""
    flat_config = {}
    for k, v in config.items():
        if isinstance(v, dict) or OmegaConf.is_dict(v):
            for k2, v2 in flatten(v).items():
                flat_config[f"{k}.{k2}"] = v2
        else:
            flat_config[k] = v
    return flat_config


def grid_to_list(grid):
    """Convert a grid to a list of configs."""
    flat_grid = flatten(grid)
    iter_overwrites = {}
    flat_overwrites = {}
    for k, v in flat_grid.items():
        if isinstance(v, list) or OmegaConf.is_list(v):
            iter_overwrites[k] = v
        else:
            flat_overwrites[k] = v

    product_values = list(product(*iter_overwrites.values()))
    grid_list = []
    for values in product_values:
        overwrite_dict = dict(zip(iter_overwrites.keys(), values))
        overwrite_dict.update(flat_overwrites)
        grid_list.append(overwrite_dict)
    return grid_list

def run(cli_args):
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    slurm_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))

    base_sweep = OmegaConf.load(cli_args.sweep_config)
    list_of_sweeps = base_sweep.pop("sweep")
    config_list = []
    for sweep in list_of_sweeps:
        sweep_config = OmegaConf.merge(base_sweep, sweep)
        config_list += grid_to_list(sweep_config)

    if slurm_task_id >= len(config_list):
        logger.error(f"Task ID {slurm_task_id} exceeds the number of configurations {len(config_list)}")
        return
    overrides = config_list[slurm_task_id]
    launch_args = [
        f"torchrun --nproc_per_node 1",
        f"--nnodes 1",
        "run.py",
        f"--slurm_run_name {slurm_job_id}_{slurm_task_id}"
    ]
    
    for k, v in overrides.items():
        if isinstance(v, bool):
            if v == True:
                launch_args.append(f"--{k}")
        else:
            launch_args.append(f"--{k}={v}")
    print(" ".join(launch_args), flush=True)
    result = subprocess.run([
        "bash",
        "-c", ' '.join(launch_args)
    ], capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    print(cli_args, flush=True)
    run(cli_args)