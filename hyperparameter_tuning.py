import sys
import os
import stat
import subprocess
from cluster_script import run as run_csgrid
from ccv_script import run as run_ccv

# Tuning hyperparameters

SEED_START = 0
SEEDS_PER_RUN = 3

# Program args
default_args = [
   "--env", "PongNoFrameskip-v4",
#    "--env", "SeaquestNoFrameskip-v4",
#    "--env", "BreakoutNoFrameskip-v4",
#    "--env", "QbertNoFrameskip-v4",
#    "--env", "MsPacmanNoFrameskip-v4",
    "--enable-cudnn", 
#   Common args
    "--evaluation-interval", "10000",
    "--target-update", "2000",
    "--noisy-std", "0.1",
    "--atoms", "51",
    "--replay-frequency", "4",
    "--priority-exponent", "0.5",
    "--priority-weight", "0.4",
    "--reward-clip", "1",
    "--batch-size", "32",
    "--adam-eps", "1.5e-4",
#   Data efficient args
    "--learn-start", "1600", 
    "--replay-frequency", "1",
    "--multi-step", "20", 
    "--hidden-size", "256",
    "--architecture", "data-efficient",
    "--T-max", "400000",
    "--memory-capacity", "400000",
#   Other args
    "--checkpoint-interval", "100000",
    "--memory", "memory_store",
]

# Values to tune
tuning_values = {
    "--learning-rate": ["0.0001"]
}


if __name__ == "__main__": 
    if len(sys.argv) != 3:
        raise RuntimeError("""Usage:
python hyperparameter_tuning.py /path/to/env/ [ccv | csgrid | no_grid]""")
    ENV_PATH = sys.argv[1]
    grid_type = sys.argv[2]
    seed = SEED_START
    # Cluster args
    if grid_type == "ccv":
        cluster_args = [
            "--cpus", "4",
            "--gpus", "1",
            "--mem", "5",
            "--env", ENV_PATH,
            "--duration", "medium",
        ]
    elif grid_type == "csgrid":
        cluster_args = [
            "--jobtype", "gpu",
            "--mem", "5",
            "--nresources", "1",
            "--env", ENV_PATH,
            # "--duration", "vlong",
        ]
    elif grid_type == "no_grid":
        pass                                           
    else:
        raise RuntimeError("""Usage:
python hyperparameter_tuning.py /path/to/env/ [ccv | csgrid | no_grid]""")

    for _ in range(SEEDS_PER_RUN):
        for i, (arg, value_range) in enumerate(tuning_values.items()):
            for value in value_range:
                run_args = default_args + [arg, value]
                clean_arg_name = arg.strip('-').replace('-', '_')
                run_tag = f"{clean_arg_name}_{value}"
                run_args += ["--uuid", run_tag]
                run_args += ["--seed", str(seed)]
                cmd = "python main.py " + ' '.join(run_args)
                jobname = f"{default_args[1].replace('-', '_')}_{run_tag.replace('-','_')}_seed_{str(seed)}"
                if grid_type != "no_grid":
                    cmd = "unbuffer " + cmd
                    cluster_args += ["--command", cmd]
                    cluster_args += ["--jobname", jobname]
                if grid_type == "ccv":
                    run_ccv(custom_args=cluster_args)
                elif grid_type == "csgrid":
                    run_csgrid(custom_args=cluster_args)
                elif grid_type == "no_grid":
                    os.makedirs("./jobs/logs", exist_ok=True)
                    os.makedirs("./jobs/scripts", exist_ok=True)
                    print(cmd)
                    with open(f"./jobs/scripts/{jobname}", "w+") as sc:
                        sc.write(f"""
#!/usr/bin/env bash
{cmd}""")
                    os.chmod(f"./jobs/scripts/{jobname}", stat.S_IRWXU)
                    script_cmd = f"CUDA_VISIBLE_DEVICES={seed % 4} ./jobs/scripts/{jobname}"
                    subprocess.Popen(script_cmd, shell=True)
                else:
                    raise RuntimeError("""Usage:
python hyperparameter_tuning.py /path/to/env/ [ccv | csgrid]""")

                seed += 1
