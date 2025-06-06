# OFI Monitoring

This document describes how to build, deploy, and run the OFI Monitoring Stack.

Note: As of [`b690555`](https://github.com/ofiwg/libfabric/commit/b690555e2998fa9cfe6cebc874f42c9281887610), an updated version of the the `ofi_hook_monitor` provider has been upstreamed into libfabric. 
The version used in the paper is still kept in this repository for reproducibility purposes.

## Installing the OFI Monitoring stack

### Preparing Libfabric

The OFI Monitoring implementation has been tested using libfabric v1.22.0 (https://github.com/ofiwg/libfabric/tree/v1.22.0), but should - barring major changes - work on future versions as well.

Clone libfabric v1.22.0 (or download a release-version) and apply the patch file `0001-ofi_hook_monitor_v5.patch` using `git apply < $file`. 

Next, run `autogen.sh` if you checked out from an upstream git tree.
Next, run `./configure` with the parameter `--enable-monitor=dl`, plus all other components you require (such as `--enable-cxi` for Slingshot or `--enable-verbs` for ibverbs/InfiniBand).

Finally run `make` and `make install`. 

### Preparing OpenMPI

The monitoring stack has been tested on OpenMPI v5.0.5 (https://github.com/open-mpi/ompi/tree/v5.0.5), but should work with other versions as well. The only requirement is that OpenMPI supports & is configured to run with libfabric. No patches are required.

During build time, `./configure` OpenMPI with `--with-ofi=/path/to/above_built/libfabric_install`.


### Preparing Telegraf

The stack has been tested with telegraf v1.32.0 (https://github.com/influxdata/telegraf/tree/v1.32.0). Clone it and apply the patch file `0001-ofi_telegraf_v3.patch`.

Next, run `go mod tidy`.
Finally, run `make`. This might take a while.

Note: This requires an existing Go installation. Telegraf v1.32.0 expects Go v1.23+.


## Configuring the components

### Telegraf
In order to run the monitoring stack, for each involved node, a telegraf instance needs to be running with the `[[inputs.ofi]]` input configured. Refer to the `telegraf.conf` attached to this document. 

Note: Make sure that the `basepath` points to a directory running on a tmpfs! For almost all systems, `/dev/shm` suffices. 

Note: To support multi-user separation, each `ofi_hook_monitor` output directory should have a shared prefix, e.g. `/dev/shm/ofi_$USER`. Set `folder_prefix` to this prefix (e.g. `ofi_`).

Note: If you use the provided `telegraf.conf` file, make sure that the output plugin `[[outputs.file]]` points to a vaild output file via the parameter `files`. This file will contain the output metrics. There will be one file per host!


### Libfabric
The libfabric `ofi_hook_monitor` has to be explicitly enabled. You can achieve this by setting the following environment variable: `FI_HOOK=ofi_hook_monitor`.

These variables are also available:

- `FI_HOOK_MONITOR_BASEPATH`: set this to a directory stored on a tmpfs. To support multi-user separation, set this to e.g. `/dev/shm/ofi_$USER`. Note that both the basepath `/dev/shm` and the prefix `ofi_` need to match the telegraf configuration! (default: `/dev/shm/ofi`)
  - Note: the OFI monitor provider heavily assumes a tmpfs! If you choose a storage location other than `/dev/shm`, make sure it is a tmpfs!
- `FI_HOOK_MONITOR_LINGER`: set this to 1 if the output files should linger on even after termination of the libfabric application. This is useful to allow the sampler (telegraf) to fetch the last counter values even after termination. The sampler must then take care of deleting the output files. (default: 0)
- `FI_HOOK_MONITOR_TICK_MAX`: set this to the amount of ticks (i.e. calls to any libfabric API like `fi_send`) before the counters are flushed to file (default: 1024)

## Running

### Running plain MPI Applications

For the above-configured OpenMPI 5, you can run an MPI application as follows:

```bash
/path/to/mpirun \
        --host $host1,$host2,... -np $N \
        --mca mtl ofi \
        --mca blt ofi \
        --mca pml cm\
        -x FI_HOOK=ofi_hook_monitor\
        -x FI_HOOK_MONITOR_BASEPATH=/dev/shm/ofi_$USER \
        -x FI_HOOK_MONITOR_TICK_MAX=1024\
        -x FI_HOOK_MONITOR_LINGER=1\
        $mpi_app
```

Note: Setting the `mtl`, `btl`, and `pml` as shown above is required to force OpenMPI to use libfabric for all communication and hence trigger the monitoring system.

### Running via Slurm

One possible slurm script could be as follows:

```bash
#!/bin/bash

#SBATCH --job-name=ofi_hook_monitor
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=$partition
#SBATCH --time=00:01:00
#SBATCH --output=/path/to/%x_%j.out
#SBATCH --error=/path/to/%x_%j.err

srun --overlap -N 2 --ntasks-per-node=1 \
	/path/to/telegraf \
    --config /path/to/telegraf.conf &

SLURM_OVERLAP=1 /path/to/mpirun \
	--mca mtl ofi \
	--mca blt ofi \
	--mca pml cm\
	-x FI_HOOK=ofi_hook_monitor\
	-x FI_HOOK_MONITOR_BASEPATH=/dev/shm/ofi_$USER \
	-x FI_HOOK_MONITOR_TICK_MAX=1024\
	-x FI_HOOK_MONITOR_LINGER=1\
	-x FI_HOOK_MONITOR_SYNC_THRESHOLD_SEC=10\
	$mpi_app
```

Note: This script runs two tasks simultaneously (using `--overlap`): First, the telegraf sampler, second the actual MPI job. Make sure to set the number of nodes both in `SBATCH --nodes` as well as in the first `srun` call to the same number!

Note: Verify that the overlap feature is working correctly! Some sites disable it. For a more comprehensive solution, please refer to `aggregation/launcher.job`.
