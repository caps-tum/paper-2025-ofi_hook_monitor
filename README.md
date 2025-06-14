# ofi_hook_monitor

This repository contains the data to build, deploy, and benchmark the `ofi_hook_monitor` OFI plugin,
as presented in the paper "Application-focused HPC Network Monitoring", published at ISC-HPC 2025.

The repo is structured as follows:

- `deployment` contains all files for installing & deploying the plugin
  - Note: As of [`b690555`](https://github.com/ofiwg/libfabric/commit/b690555e2998fa9cfe6cebc874f42c9281887610), an updated version of the the `ofi_hook_monitor` provider has been upstreamed into libfabric. The version used in the paper is still kept in this repository for reproducibility purposes.
- `aggregation` contains all files for benchmarking the plugin
- `visualization` contains visualization scripts
- `data` contains the data gathered during benchmarking

Please refer to the respective `README.md` per folder.

## License
The license is stated in the LICENSE file.
> Note: This does not apply to the result of applying the patch files in `$ROOT/deployment/patches` to the respective projects. 
> Please refer to the respective projects for information about their license.
