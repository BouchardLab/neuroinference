import argparse
import numpy as np


def main(args):
    batch_path = args.batch_path
    save_tag = args.save_tag
    script_path = args.script_path

    # Experiment settings
    n_datasets = args.n_datasets
    model_rng = args.model_rng
    data_rng = np.random.default_rng(args.data_rng)
    data_rngs = data_rng.integers(low=0, high=2**32-1, size=n_datasets)

    # Handle NERSC settings
    n_nodes = args.n_nodes
    n_tasks = args.n_tasks
    time = args.time
    qos = args.qos

    # Header of batch script
    batch_script = (
        "#!/bin/bash\n"
        f"#SBATCH -N {n_nodes}\n"
        "#SBATCH -C haswell\n"
        f"#SBATCH -q {qos}\n"
        "#SBATCH -J nb\n"
        "#SBATCH --output=/global/homes/s/sachdeva/out/uoineuro/poisson.o\n"
        "#SBATCH --error=/global/homes/s/sachdeva/error/uoineuro/poisson.o\n"
        "#SBATCH --mail-user=pratik.sachdeva@berkeley.edu\n"
        "#SBATCH --mail-type=ALL\n"
        f"#SBATCH -t {time}\n"
        f"#SBATCH --image=docker:pssachdeva/neuro:latest\n\n"
        "export OMP_NUM_THREADS=1\n\n"
    )

    for idx in range(n_datasets):
        batch_script += f"echo 'Job {idx+1}/{n_datasets}'\n"
        batch_script += (
            f"srun -n {n_tasks} -c 1 shifter python -u {script_path} "
            f"--save_path={save_tag}_idx.h5"
            f"--n_features={args.n_features} "
            f"--n_nz_features={args.n_nz_features} "
            f"--n_samples={args.n_samples} "
            f"--high={args.high} "
            f"--scale={args.scale} "
            f"--model_rng={model_rng} "
            f"--data_rng={data_rngs[idx]} "
            f"--n_lambdas={args.n_lambdas} "
            f"--n_boots_sel={args.n_boots_sel} "
            f"--n_boots_est={args.n_boots_est} "
            f"--selection_frac={args.selection_frac} "
            f"--estimation_frac={args.estimation_frac} "
            f"--stability_selection={args.stability_selection} "
            f"--estimation_score={args.estimation_score} "
            f"--max_iter={args.max_iter}\n"
        )

    with open(batch_path, 'w') as batch:
        batch.write(batch_script)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_path', type=str)
    parser.add_argument('--save_tag', type=str)
    parser.add_argument('--script_path', type=str)
    # NERSC options
    parser.add_argument('--n_nodes', type=int, default=30)
    parser.add_argument('--n_tasks', type=int, default=32)
    parser.add_argument('--time', default='00:30:00')
    parser.add_argument('--qos', default='debug')
    # Random number generator arguments
    parser.add_argument('--model_rng', type=int, default=2332)
    parser.add_argument('--data_rng', type=int, default=48119)
    # Experiment arguments
    parser.add_argument('--n_datasets', type=int, default=30)
    parser.add_argument('--n_features', type=int, default=300)
    parser.add_argument('--n_nz_features', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=1200)
    parser.add_argument('--high', type=int, default=1)
    parser.add_argument('--scale', type=int, default=1)
    # UoI objects
    parser.add_argument('--standardize', action='store_true')
    parser.add_argument('--n_lambdas', type=int, default=50)
    parser.add_argument('--n_boots_sel', type=int, default=30)
    parser.add_argument('--n_boots_est', type=int, default=30)
    parser.add_argument('--selection_frac', type=float, default=0.8)
    parser.add_argument('--estimation_frac', type=float, default=0.8)
    parser.add_argument('--stability_selection', type=float, default=0.95)
    parser.add_argument('--estimation_score', default='BIC')
    parser.add_argument('--max_iter', type=int, default=10000)

    args = parser.parse_args()

    main(args)
