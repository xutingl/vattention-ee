# DREX experiments

This is the home for **new** DREX experiments. From now on, run experiments here and
keep each run's results next to the script that produced them — so a run is easy to
find, reproduce, and compare.

> Historical results live elsewhere and are **not** reorganized: `outputs/`,
> `outputs_13b/`, `outputs_70b/`, `outputs_70b_llama3/`, `outputs_13b_tuned/`,
> `outputs_7b_daniel/`, `outputs_qwen_14b/`, and the legacy `experiments/e2e_dynamic_eval/`.
> Those predate this convention and are left as-is.

## Layout

```
experiments/
  <experiment_name>/
    README.md                  # what this experiment is, what it sweeps, the metric
    <environment_model>/       # one runnable instance for a given GPU + model
      README.md                # env-specific notes + exact launch command
      run.sh                   # direct-run script (or run.sbatch for login-node submit)
      summarize.py             # optional: aggregate results/ -> a table
      results/                 # CSVs + per-run logs + summary (created at runtime)
```

- **`<experiment_name>/`** — one logical experiment (e.g. `sanity_check`).
- **`<environment_model>/`** — `<gpu>_<model>`, e.g. `b200_llama_13b`. This pins the
  hardware + model an instance was run on, so the same experiment can have several
  instances side by side (`b200_llama_13b/`, `b200_llama_70b/`, ...).

See **[`sanity_check/`](sanity_check/)** as the reference example/template.

## How to run an experiment

Two ways, depending on where your shell is (mirrors the root
[`CLAUDE.md`](../CLAUDE.md) "Running experiments"):

### Directly on a compute node — current setup (use `nohup`)

When your shell is already on a `dgx-b200` node with a GPU:

```bash
cd experiments/<experiment_name>/<environment_model>
export CUDA_VISIBLE_DEVICES=0       # pick a free GPU on the shared node
nohup bash run.sh > run.log 2>&1 &  # GPU job — detach with nohup
tail -f run.log
```

`run.sh` is self-contained: it `source`s `scripts/drex_env.sh` (venv + local CUDA
toolkit), sets `RAY_ADDRESS=local`, and runs `ray stop --force` before each run.

### From a login node (use `sbatch`)

Login nodes have no GPU and will kill model-loading jobs. Submit a `run.sbatch` that
requests the `dgx-b200` partition so the job lands on a compute node:

```bash
sbatch experiments/<experiment_name>/<environment_model>/run.sbatch
```

## Where results go

Each run of `scripts/run_ee.py` writes one CSV, named by
`benchmark_runner.py` as
`req_<nreq>_batch_<bs>_layer_<layer>_conf_<conf>_<policy>_<kv>.csv`. We point its
`--csv_path` at the instance's `results/` dir, so every run of a sweep lands there with
a distinct, self-describing filename (no collisions). Per-run stdout is teed to
`results/log_<...>.txt`, and an experiment's `summarize.py` writes `results/summary.txt`.

## Adding a new experiment

1. Copy the skeleton: `cp -r sanity_check/b200_llama_13b <experiment_name>/<environment_model>`
   (or start a fresh `<environment_model>/` under an existing experiment).
2. Edit the **config block** at the top of `run.sh` (model, layer, conf, batch sizes,
   policies, num_requests).
3. Write the experiment `README.md` (what/why/metric) and the instance `README.md`
   (env + launch command).
4. Run it as above; commit the scripts/READMEs (and the results you want to keep).

See also: root [`README.md`](../README.md) and [`CLAUDE.md`](../CLAUDE.md) for the full
setup, the `run_ee.py` flag reference, and the B200 environment.
