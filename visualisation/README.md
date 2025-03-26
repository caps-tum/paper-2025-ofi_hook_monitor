# Visualisations
This directory contains all files for reproducing all data-related figures in the paper.

The associated Python scripts have been developed and run on a Linux machine using Python 3.12.
The package requirements are listed in `requirements.txt`. Please install these, ideally in a virtual environment, prior to executing the scripts.

The following list lists all Figures and the associated script to run. Output data is stored in `$ROOT/visualisation/{figures,monitoring}`.

- Figure  3: `benchmarks/plotter.py`, files `figures/sng/2_osu_{latency,mbw_mr}.pdf`
- Figure  4: `benchmarks/plotter.py`, files `figures/sng/{4,4_47c}_osu_bcast.pdf`
- Figure  5: `benchmarks/plotter.py`, files `figures/sng/{4,4_47c}_osu_ibcast.pdf`
- Figure  6: `benchmarks/plotter.py`, file  `figures/atos/atos_collective_4.pdf`
- Figure  7: `benchmarks/plotter.py`, file  `figures/atos/scaling_atos_osu_ibcast.pdf`
- Figure  8: `benchmarks/plotter.py`, file  `figures/lumi/scaling_lumi_osu_ibcast.pdf`

- Figure  9: `monitoring.py`, files `monitoring/{api_calls_ofi,api_calls_ofi_sum}.pdf`
- Figure 10: `monitoring.py`, file `monitoring/tsenddata.pdf`
- Figure 11: `monitoring.py`, file `monitoring/tsenddata_zoom.pdf`
- Figure 12: `monitoring.py`, file `monitoring/trecv_node_focus.pdf`
- Figure 13: `monitoring.py`, file `monitoring/tsenddata_node_focus.pdf`
- Figure 14: `monitoring.py`, file `monitoring/tsenddata_node_bucket.pdf`

Please call the script files `$ROOT/visualisation/benchmarks/plotter.py` and `$ROOT/visualisation/monitoring.py` 
from the `$ROOT` directory as follows:

```bash
PYTHONPATH=$(pwd) python3 visualisation/benchmarks/plotter.py
PYTHONPATH=$(pwd) python3 visualisation/monitoring.py
```

Figures 1 and 2 have been crafted using Affinity Design 2, the SVG files are placed in `$ROOT/visualisation/plots`.

The accompanying median and percentile data for Figures 3-5 is written to the folder `$ROOT/visualisation/meta/$SYSTEM`.
The format is:
```
Run $TYPE: $AVERAGE ($PERCENTILE_10, $PERCENTILE_90)
[...]
```

# License
All images are licensed under CC-BY-SA 4.0.