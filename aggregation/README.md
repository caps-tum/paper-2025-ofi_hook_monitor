# ofi_hook_monitor Benchmarking

The scripts in this folder perform benchmarks using the OSU Microbenchmark suite.

The folder `launcher` contains a bash launcher script, which runs a given binary for N iterations and tests three "profiles":

- (0) baseline, without `ofi_hook_monitor`
- (1) with `ofi_hook_monitor` and `FI_HOOK_MONITOR_TICK_MAX=20000`
- (2) with `ofi_hook_monitor` and `FI_HOOK_MONITOR_TICK_MAX=100`

The output is stored relative to a given `$basedir` at subfolder `data`. These folders are:

- `job`: output of SLURM jobscript, if present (see below)
- `measurements`: output of OSU benchmarks. Each measurement contains three folders `profile_[0,1,2]`, storing the actual outputs, and a `meta.md` file, storing metadata of that run.
- `monitoring`: output of the telegraf sampler, zipped in a zstd file.

Usage: 

```shell
bash launcher_multi.sh \
	--basedir "$BASEDIR" \
	--mpirun "$MPIRUN" \
    --iterations 25 \
    --mpi-args "--mca mtl ofi --mca btl ofi --mca pml cm"\
    --binary "$OSU_BASE/collective/blocking/osu_bcast"\
    --binary-args "-m 1:262144"
```

Note: Make sure that the sampler is running prior to launching the measurement script!

## Slurm

Should you want to test this using SLURM (recommended), you may use the `slurm/launcher.job` jobsript.

This script requires several tweaks to properly run on a given HPC site:

- the SBATCH parameters need to be updated, especially the output and error paths. Ideally point these to `$PROJECT_ROOT/aggregation/data/job`
- adjust the `module [un]load` calls to load proper modules (& remove those modules you don't want! This could be e.g. an existing libfabric)
- the `LAUNCHER`, `BASEDIR`, `MON_OUTDIR`, `MPIRUN`, `OSU_BASE`, and `LAST_CORE` variables need to be updated
	- `BASEDIR` should point to `$PROJECT_ROOT/aggregation/data/measurements`
	- `MON_OUTDIR` should point to `$PROJECT_ROOT/aggregation/data/monitoring/$SLURM_JOB_ID`
		- Note: Uncomment the last `srun` call if your telegraf instance does not write to a shared filesystem! This could be the case for the `telegraf_1s_shm.conf`, where it writes to shared memory
	- `LAST_CORE` set to the last physical (i.e. no SMT) CPU core ID. For a 48-core system, this would be 47
- the `--config` file for the telegraf sampler needs to be verified
	- check the `files = ` directive and point its stem to `$MON_OUTDIR`

> Note: This SLURM script assumes that you can ssh into all nodes of an allocation.

### Benchmarks

The `launcher.job` script currently runs the following OSU benchmarks 25 times:
- one Rank per Node (hybrid OpenMP/MPI simulation)
	- `osu_bcast`
	- `osu_ibcast`
- LAST_CORE ranks per node
	- `osu_bcast`
	- `osu_ibcast`


Finally, before launching, it might make sense to verify that the overlap feature works. For this, you may use the `test.job` file.
This launches one `osu_bcast` call for the number of nodes specified via `--nodes`, and overlapps it with the previously launched sampler. 

#### Launching

To launch the jobscript, make sure to pass the following variables:
```bash
sbatch --nodes=4 --ntasks-per-node=47 launcher.job
```
where `--nodes` is the number of nodes and `--ntasks-per-node` should be the number of physical cores _except_ the last one (reserved for telegraf!).
Make sure to first specify the parameters and then the jobfile! Otherwise the parameters will get silently dropped.


Please note that for larger node counts (>= 64), the osu_ibcast calls may take a _long_ time to finish. If the runtime is prohibitive, then try to decrease the number of `--iterations`.

### Slurm, interactive

Should batch jobs not work, then the contents of the `launcher.job` scripts can be copy&pasted piece by piece into an interactive allocation.

## Verifying the core pinning
While running, SSH into one of the machine and run `top`. Press `f`, scroll down to `P = Last Used CPU (SMP)`, tick it `space`, then exit `q`. You should see the column `P`. Verify that telegraf is running on `$LAST_CORE` and that each `--binary` is running on a distinct core.