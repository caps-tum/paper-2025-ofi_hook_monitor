import functools
import itertools
import typing
from typing import Optional, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D

from visualisation.benchmarks.loaders import RunType, load
from visualisation.classes import Run
from visualisation.utils import sizeof_fmt

matplotlib.rc('font', **{
    'family': 'sans',
    'size': 24})
# matplotlib.use('QtAgg')  # or can use 'TkAgg', whatever you have/prefer


np.set_printoptions(linewidth=240)


def lineplot_osu(run_baseline: Run,
                 run_b: [Run],
                 basepath: Path,
                 metric: Union[Union[str, typing.Callable], list[Union[str, typing.Callable]]],
                 metric_name: str = "",
                 mode: str = "",
                 error_q: float = 0.1,
                 save: bool = False,
                 title: bool = True,
                 figsize: tuple[int, int] = (12, 8),
                 y_max: Optional[float] = None,
                 fname: Optional[str] = None,
                 out_dir: str = "benchmark"):
    """
    Visualise benchmark Runs as linesplot.

    :param run_baseline: Baseline Run
    :param run_b: List of measurement Runs, compared against baseline Run
    :param metric: which metric to plot (str for simple key, depends on importer, or callable on a DataFrame for more complex lookups), can also be list of metrics
    :param metric_name: metric name to plot on yaxis
    :param mode: evaluation modes - "speedup", "delta", None
    :param error_q: percentile for error bars, [0,1]
    :param basepath: basepath to store figure output to
    :param out_dir: output directory
    :param save: whether to save plot to file (will plt.show() on false)
    :param title: whether to add title
    :param figsize: matplotlib figure size
    :param fname: optional figure name suffix
    """
    if not isinstance(metric, list):
        metric = [metric]

    fig, ax = plt.subplots(ncols=1, figsize=figsize)

    fig: plt.Figure
    _metric_name = metric_name

    orig_shape = run_baseline.df.loc[1, :].shape
    data_baseline = np.array(run_baseline.df[metric[0]]).reshape((-1, orig_shape[0])).T
    data_baseline_mean = np.mean(data_baseline, axis=1)
    xlabels = list(run_baseline.df.loc[1, :].index)

    overall_min = 0
    overall_max = 0
    plot_metadata = ""
    if mode == "delta":
        for r_i, run in enumerate(run_b + [run_baseline]):
            data_b = np.array(run.df[metric[0]]).reshape((-1, orig_shape[0])).T

            diff = np.mean(data_b, axis=1) - data_baseline_mean

            diff_min = np.quantile(data_b, q=error_q, axis=1) - data_baseline_mean
            diff_max = np.quantile(data_b, q=1 - error_q, axis=1) - data_baseline_mean

            ax.plot(diff, label=f"{run.tag}")
            ax.fill_between(np.arange(len(diff)), diff_min, diff_max, alpha=.25)
            _metric_name = f"Δ"

    elif mode == "speedup" or mode == "speedup_inverse":
        inverse = mode == "speedup_inverse"
        for r_i, run in enumerate(run_b):
            data_b = np.array(run.df[metric[0]]).reshape((-1, orig_shape[0])).T

            data_b_mean = np.mean(data_b, axis=1)
            if inverse:
                speedup = (data_baseline_mean / data_b_mean) - 1
                speedup_min = (data_baseline_mean / np.quantile(data_b, q=error_q, axis=1)) - 1
                speedup_max = (data_baseline_mean / np.quantile(data_b, q=1 - error_q, axis=1)) - 1
            else:
                speedup = (data_b_mean / data_baseline_mean) - 1
                speedup_min = (np.quantile(data_b, q=error_q, axis=1) / data_baseline_mean) - 1
                speedup_max = (np.quantile(data_b, q=1 - error_q, axis=1) / data_baseline_mean) - 1

            ax.plot(speedup, label=f"{run.tag}")
            _metric_name = f"overhead"
            ax.fill_between(np.arange(len(speedup)), speedup_min, speedup_max, alpha=.25)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))

            plot_metadata += f"Run {run.tag}: {np.average(speedup) * 100:.2f}% " \
                             f"({np.quantile(speedup, error_q) * 100:.2f}%, {np.quantile(speedup, 1 - error_q) * 100:.2f}%)\n"
            overall_min = min(overall_min, min(speedup_min))
            overall_max = max(overall_max, max(speedup_max))

    else:
        for r_i, r in enumerate([run_baseline] + run_b):
            data = np.array(r.df[metric[0]]).reshape((-1, orig_shape[0])).T
            data_mean = np.mean(data, axis=1)
            ax.errorbar(x=np.arange(len(data_mean)),
                        y=data_mean,
                        yerr=[np.clip(data_mean - np.min(data, axis=1), a_min=0, a_max=None),
                              np.clip(np.max(data, axis=1) - data_mean, a_min=0, a_max=None)],
                        label=f"{r.tag}")
            ax.set_yscale("log")

    xlabels_fmt = list(map(sizeof_fmt, xlabels))
    ax.set_xticks(np.arange(orig_shape[0]), xlabels_fmt, rotation=45, ha="right")
    if overall_min >= -0.2 and overall_max < 0.2:
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))

    if title:
        ax.set_title(run_baseline.name.split(" ")[0])
    if metric_name != "":
        ax.set_ylabel(metric_name)
    if y_max:
        ax.set_ylim(top=y_max)
    ax.yaxis.grid(True)
    ax.yaxis.grid(True, which="minor", alpha=.5)
    ax.legend(ncols=3)

    fig.tight_layout(pad=0)

    if save:
        prefix = f"{fname}_" if isinstance(fname, str) else ""
        output_dir = basepath / "figures" / out_dir
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)
        plt.savefig(output_dir / f"{prefix}{run_baseline.name.split(' ')[0]}.pdf")

        meta_output_dir = basepath / "meta" / out_dir
        if not meta_output_dir.is_dir():
            meta_output_dir.mkdir(parents=True)
        with open(meta_output_dir / f"{prefix}{run_baseline.name.split(' ')[0]}.meta", "w") as out:
            out.write(plot_metadata)
    else:
        plt.show()


def boxplots(run_set: dict[str, typing.Any],
             system: str,
             size: str,
             basepath: Path,
             basepath_data: Path,
             save: bool = False,
             whis_q: int = 10,
             locator_base: float = 0.01,
             figsize: tuple[int, int] = (12, 8),
             y_decimals: int = 2,
             fname: Optional[str] = None,
             out_dir: Optional[str] = None):
    """
    Draw boxplots for given runs
    :param run_set: dictionary of runs
    :param system: system name
    :param size: node allocation size
    :param basepath: basepath for output images
    :param basepath_data: basepath for input data
    :param save: whether to save plots (True) or show (False)
    :param whis_q: percentile for boxpot whiskers, range [0,100]
    :param locator_base: Base for axis tick locator - Locator is percentage-based, with maximum 1 (0.01 = 1%)
    :param figsize: size of figure
    :param y_decimals: number of decimals for y-axis
    :param fname: output file name
    :param out_dir: output file directory
    """

    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    fig: plt.Figure

    def _calculate_overhead(df_baseline, df_run, _metric: str):
        orig_shape = df_baseline.loc[1, :].shape
        data_baseline_mean = np.mean(np.array(df_baseline[_metric]).reshape((-1, orig_shape[0])).T, axis=1)
        data_b = np.array(df_run[_metric]).reshape((-1, orig_shape[0]))
        speedup = np.mean((data_b / data_baseline_mean) - 1, axis=1)
        return speedup

    overheads = []
    for call, metric in [("osu_bcast", "Avg Latency(us)"), ("osu_ibcast", "Pure Comm.(us)")]:
        run_1_baseline = load(folder=run_set[system][size][call], run_type=RunType.OSU, profile="0", tag="baseline",
                              basepath_data=basepath_data / system / size)
        run_1_100 = load(folder=run_set[system][size][call], run_type=RunType.OSU, profile="2", tag="tick=100",
                         basepath_data=basepath_data / system / size)
        run_full_baseline = load(folder=run_set[system][f"{size}_full"][call], run_type=RunType.OSU, profile="0",
                                 tag="baseline", basepath_data=basepath_data / system / size)
        run_full_100 = load(folder=run_set[system][f"{size}_full"][call], run_type=RunType.OSU, profile="2",
                            tag="tick=100", basepath_data=basepath_data / system / size)

        overheads.append(_calculate_overhead(run_1_baseline.df, run_1_100.df, _metric=metric))
        overheads.append(_calculate_overhead(run_1_baseline.df, run_1_baseline.df, _metric=metric))
        overheads.append(_calculate_overhead(run_full_baseline.df, run_full_100.df, _metric=metric))
        overheads.append(_calculate_overhead(run_full_baseline.df, run_full_baseline.df, _metric=metric))

    # "bundle" every even-odd pair of boxplots as they belong together
    group_interspacing = 0.35
    positions = [
        (2 * i - group_interspacing, 2 * i + group_interspacing)
        for i in range(len(overheads) // 2)
    ]
    positions = list(itertools.chain(*positions))

    bplots = ax.boxplot(overheads, showfliers=False,
                        widths=0.6,
                        meanline=True,
                        showmeans=True,
                        positions=positions,
                        whis=(whis_q, 100 - whis_q))

    # add primary group labels
    x_axis_primary = [(0, "one process"), (2, "full"),
                      (4, "one process"), (6, "full")]
    ax.set_xticks([tup[0] for tup in x_axis_primary],
                  labels=[f"{tup[1]}" for tup in x_axis_primary])

    # add secondary axis, offset downward, make spine and ticks invisible
    #  use this axis to add labels for a second group
    sec = ax.secondary_xaxis(-0.14)
    x_axis_secondary = [(1, "osu_bcast"), (5, "osu_ibcast")]
    sec.spines["bottom"].set_visible(False)
    sec.set_xticks([tup[0] for tup in x_axis_secondary],
                   labels=[f"{tup[1]}" for tup in x_axis_secondary])
    sec.tick_params("x", length=0)

    # add secondary group separators
    locs = [tup[0] for tup in x_axis_secondary]
    if len(locs) > 1:
        for loc_right, loc_left in zip(locs[:-1], locs[1:]):
            loc = np.mean([loc_right, loc_left])
            ax.axvline(x=loc, color="black", alpha=.5, linestyle=":")

    # make boxplot median blue/red depending on group
    for i, bplot in enumerate(bplots["medians"]):
        if i % 2 == 0:
            bplot.set_color("tab:orange")
        else:
            bplot.set_color("tab:green")
        bplot.set_linewidth(2)
    for i, bplot in enumerate(bplots["means"]):
        if i % 2 == 0:
            bplot.set_color("tab:orange")
        else:
            bplot.set_color("tab:green")
        bplot.set_linewidth(2)

    for bplot in bplots["boxes"]:
        bplot.set_linewidth(1.5)
    for bplot in bplots["whiskers"]:
        bplot.set_linewidth(1.5)

    custom_lines = [Line2D([0], [0], color="tab:orange", lw=2),
                    Line2D([0], [0], color="tab:green", lw=2)]

    ax.legend(custom_lines, ['tick=100', 'baseline'], ncols=2)

    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=y_decimals))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=locator_base))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=locator_base / 2))
    # ax.set_ylabel("Overhead")
    ax.yaxis.grid(True)
    ax.yaxis.grid(True, which="minor", alpha=.5)

    fig.tight_layout(pad=0)

    if save:
        suffix = f"_{fname}" if isinstance(fname, str) else ""
        output_dir = basepath / "figures" / out_dir
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)
        plt.savefig(output_dir / f"{system}{suffix}.pdf")
    else:
        plt.show()


def scaling(run_set: dict[str, typing.Any],
            system: str,
            call: str,
            metric: str,
            sizes: [str],
            basepath: Path,
            basepath_data: Path,
            save: bool = False,
            whis_q: int = 10,
            locator_base: float = 0.01,
            y_decimals: int = 2,
            figsize: tuple[int, int] = (12, 8),
            fname: Optional[str] = None,
            out_dir: Optional[str] = None):
    """
    Draw scaling boxplots across multiple node allocation sizes
    :param run_set: run set
    :param system: system name
    :param call: benchmark call name
    :param metric: metric name
    :param sizes: list of node allocation sizes
    :param basepath: basepath for output plots
    :param basepath_data: basepath for input data
    :param save: whether to save (True) or show (False) plots
    :param whis_q: percentile for boxpot whiskers, range [0,100]
    :param y_decimals: number of decimal places for y-axis
    :param locator_base: Base for axis tick locator - Locator is percentage-based, with maximum 1 (0.01 = 1%)
    :param figsize: size of figure
    :param fname: output file name
    :param out_dir: output file directory
    """

    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    fig: plt.Figure

    def _calculate_overhead(df_baseline, df_run, _metric: str):
        orig_shape = df_baseline.loc[1, :].shape
        data_baseline_mean = np.mean(np.array(df_baseline[_metric]).reshape((-1, orig_shape[0])).T, axis=1)
        data_b = np.array(df_run[_metric]).reshape((-1, orig_shape[0]))
        speedup = np.mean((data_b / data_baseline_mean) - 1, axis=1)
        return speedup

    overheads = []
    for size in sizes:
        run_baseline = load(folder=run_set[system][size][call], run_type=RunType.OSU, profile="0", tag="baseline",
                            basepath_data=basepath_data / system / size)
        run_overhead = load(folder=run_set[system][size][call], run_type=RunType.OSU, profile="2", tag="tick=100",
                            basepath_data=basepath_data / system / size)

        overheads.append(_calculate_overhead(run_baseline.df, run_overhead.df, _metric=metric))
        overheads.append(_calculate_overhead(run_baseline.df, run_baseline.df, _metric=metric))

    # "bundle" every even-odd pair of boxplots as they belong together
    group_interspacing = 0.35
    positions = [
        (2 * i - group_interspacing, 2 * i + group_interspacing)
        for i in range(len(overheads) // 2)
    ]
    positions = list(itertools.chain(*positions))

    bplots = ax.boxplot(overheads, showfliers=False,
                        widths=0.6,
                        positions=positions,
                        meanline=True,
                        showmeans=True,
                        whis=(whis_q, 100 - whis_q))

    # add primary group labels
    x_axis_primary = [(2 * i, size)
                      for i, size in enumerate(sizes)]
    ax.set_xticks([tup[0] for tup in x_axis_primary],
                  labels=[f"{tup[1]}" for tup in x_axis_primary])

    # make boxplot median blue/red depending on group
    for i, bplot in enumerate(bplots["medians"]):
        if i % 2 == 0:
            bplot.set_color("tab:orange")
        else:
            bplot.set_color("tab:green")
        bplot.set_linewidth(2)
    for i, bplot in enumerate(bplots["means"]):
        if i % 2 == 0:
            bplot.set_color("tab:orange")
        else:
            bplot.set_color("tab:green")
        bplot.set_linewidth(2)

    for bplot in bplots["boxes"]:
        bplot.set_linewidth(1.5)
    for bplot in bplots["whiskers"]:
        bplot.set_linewidth(1.5)

    custom_lines = [Line2D([0], [0], color="tab:orange", lw=2),
                    Line2D([0], [0], color="tab:green", lw=2)]

    ax.legend(custom_lines, ['tick=100', 'baseline'], ncols=2)

    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=y_decimals))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=locator_base))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=locator_base / 2))
    # ax.set_ylabel("Overhead")
    ax.yaxis.grid(True)
    ax.yaxis.grid(True, which="minor", alpha=.5)

    fig.tight_layout(pad=0)

    if save:
        suffix = f"_{fname}" if isinstance(fname, str) else ""
        output_dir = basepath / "figures" / out_dir
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)
        plt.savefig(output_dir / f"scaling_{system}{suffix}.pdf")
    else:
        plt.show()


def main_supermuc(nodes: int):
    basepath = Path(__file__).parent.parent  # double .parent to get out of benchmark/
    # basepath_data = Path("/mnt/supermuc_hppfs/ofi_hook_monitor/data/measurements")
    basepath_data = basepath.parent / "data" / "measurements" / "sng"

    save = False
    figsize = (12, 4)

    if nodes == 2:
        save = True
        _load = functools.partial(load, basepath_data=basepath_data / "2")
        # osu_latency, 2 nodes
        # old measurements_i03r11c05s01_osu_latency_24-09-27T1103
        lineplot_osu(
            run_baseline=_load(folder="measurements_3993785_i01r04c03s07_osu_latency_24-10-03T1908",
                               run_type=RunType.OSU, profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_3993785_i01r04c03s07_osu_latency_24-10-03T1908", run_type=RunType.OSU,
                      profile="1", tag="tick=20000"),
                _load(folder="measurements_3993785_i01r04c03s07_osu_latency_24-10-03T1908", run_type=RunType.OSU,
                      profile="2", tag="tick=100"),
                _load(folder="measurements_3993785_i01r04c03s07_osu_latency_24-10-03T1908", run_type=RunType.OSU,
                      profile="0",
                      tag="baseline")
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Avg Latency(us)",
            # metric_name="Overhead",
            out_dir="sng", fname="2"
        )
        #
        # osu_mbw_mr, 2 nodes
        lineplot_osu(
            run_baseline=_load(folder="measurements_i03r11c05s01_osu_mbw_mr_24-09-27T1214", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_i03r11c05s01_osu_mbw_mr_24-09-27T1214", run_type=RunType.OSU, profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_i03r11c05s01_osu_mbw_mr_24-09-27T1214", run_type=RunType.OSU, profile="2",
                      tag="tick=100"),
                _load(folder="measurements_i03r11c05s01_osu_mbw_mr_24-09-27T1214", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup_inverse",
            metric="Messages/s",
            # metric_name="Overhead",
            out_dir="sng", fname="2"
        )



    elif nodes == 4:
        _load = functools.partial(load, basepath_data=basepath_data / "4")
        save = True

        # osu_bcast, 4 nodes, 1 core
        lineplot_osu(
            run_baseline=_load(folder="measurements_3982881_i05r11c03s12_osu_bcast_24-10-02T0923", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_3982881_i05r11c03s12_osu_bcast_24-10-02T0923", run_type=RunType.OSU,
                      profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_3982881_i05r11c03s12_osu_bcast_24-10-02T0923", run_type=RunType.OSU,
                      profile="2",
                      tag="tick=100"),
                _load(folder="measurements_3982881_i05r11c03s12_osu_bcast_24-10-02T0923", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Avg Latency(us)",
            # metric_name="Overhead",
            y_max=0.1,
            fname="4", out_dir="sng"
        )
        # osu_ibcast, 4 nodes, 1 core
        lineplot_osu(
            run_baseline=_load(folder="measurements_i03r11c05s01_osu_ibcast_24-09-27T1033", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_i03r11c05s01_osu_ibcast_24-09-27T1033", run_type=RunType.OSU, profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_i03r11c05s01_osu_ibcast_24-09-27T1033", run_type=RunType.OSU, profile="2",
                      tag="tick=100"),
                _load(folder="measurements_i03r11c05s01_osu_ibcast_24-09-27T1033", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Pure Comm.(us)",
            # metric_name="Overhead",
            y_max=0.1,

            fname="4", out_dir="sng"
        )

        # osu_bcast, 4 nodes, 47 cores
        lineplot_osu(
            run_baseline=_load(folder="measurements_i01r05c05s10_osu_bcast_24-09-29T1919", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_i01r05c05s10_osu_bcast_24-09-29T1919", run_type=RunType.OSU,
                      profile="1", tag="tick=20000"),
                _load(folder="measurements_i01r05c05s10_osu_bcast_24-09-29T1919", run_type=RunType.OSU,
                      profile="2", tag="tick=100"),
                _load(folder="measurements_i01r05c05s10_osu_bcast_24-09-29T1919", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Avg Latency(us)",
            # metric_name="Overhead",
            fname="4_47c", out_dir="sng"
        )

        # osu_ibcast, 4 nodes, 47 cores
        lineplot_osu(
            run_baseline=_load(folder="measurements_i01r05c03s01_osu_ibcast_24-09-29T1920", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_i01r05c03s01_osu_ibcast_24-09-29T1920", run_type=RunType.OSU,
                      profile="1", tag="tick=20000"),
                _load(folder="measurements_i01r05c03s01_osu_ibcast_24-09-29T1920", run_type=RunType.OSU,
                      profile="2", tag="tick=100"),
                _load(folder="measurements_i01r05c03s01_osu_ibcast_24-09-29T1920", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Pure Comm.(us)",
            # metric_name="Overhead",
            fname="4_47c", out_dir="sng"
        )


    elif nodes == 16:
        _load = functools.partial(load, basepath_data=basepath_data / "16")
        save = True

        # osu_bcast, 16 nodes
        lineplot_osu(
            run_baseline=_load(folder="measurements_3991243_i05r09c04s12_osu_bcast_24-10-03T0953", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_3991243_i05r09c04s12_osu_bcast_24-10-03T0953", run_type=RunType.OSU,
                      profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_3991243_i05r09c04s12_osu_bcast_24-10-03T0953", run_type=RunType.OSU,
                      profile="2",
                      tag="tick=100"),
                _load(folder="measurements_3991243_i05r09c04s12_osu_bcast_24-10-03T0953", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Avg Latency(us)",
            # metric_name="Overhead",
            fname="16", out_dir="sng"
        )

        # osu_bcast, 16 nodes full
        lineplot_osu(
            run_baseline=_load(folder="measurements_3984691_i02r11c03s09_osu_bcast_24-10-03T0110", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_3984691_i02r11c03s09_osu_bcast_24-10-03T0110", run_type=RunType.OSU,
                      profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_3984691_i02r11c03s09_osu_bcast_24-10-03T0110", run_type=RunType.OSU,
                      profile="2",
                      tag="tick=100"),
                _load(folder="measurements_3984691_i02r11c03s09_osu_bcast_24-10-03T0110", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Avg Latency(us)",
            # metric_name="Overhead",
            fname="16_47c", out_dir="sng"
        )

        # osu_ibcast, 16 nodes
        lineplot_osu(
            run_baseline=_load(folder="measurements_3984691_i02r11c03s09_osu_ibcast_24-10-03T0127",
                               run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_3984691_i02r11c03s09_osu_ibcast_24-10-03T0127", run_type=RunType.OSU,
                      profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_3984691_i02r11c03s09_osu_ibcast_24-10-03T0127", run_type=RunType.OSU,
                      profile="2",
                      tag="tick=100"),
                _load(folder="measurements_3984691_i02r11c03s09_osu_ibcast_24-10-03T0127", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Pure Comm.(us)",
            # metric_name="Overhead",
            fname="16", out_dir="sng"
        )


    elif nodes == 64:
        save = True
        _load = functools.partial(load, basepath_data=basepath_data / "64")

        # osu_bcast, 64 nodes
        lineplot_osu(
            run_baseline=_load(folder="measurements_i07r05c04s09_osu_bcast_24-09-30T1134", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_i07r05c04s09_osu_bcast_24-09-30T1134", run_type=RunType.OSU, profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_i07r05c04s09_osu_bcast_24-09-30T1134", run_type=RunType.OSU, profile="2",
                      tag="tick=100"),
                _load(folder="measurements_i07r05c04s09_osu_bcast_24-09-30T1134", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Avg Latency(us)",
            # metric_name="Overhead",
            fname="64", out_dir="sng"
        )

        # osu_ibcast, 64 nodes
        lineplot_osu(
            run_baseline=_load(folder="measurements_i06r05c04s09_osu_ibcast_24-09-27T1233", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_i06r05c04s09_osu_ibcast_24-09-27T1233", run_type=RunType.OSU, profile="1",
                      tag="tick=1024"),
                _load(folder="measurements_i06r05c04s09_osu_ibcast_24-09-27T1233", run_type=RunType.OSU, profile="2",
                      tag="tick=1"),
                _load(folder="measurements_i06r05c04s09_osu_ibcast_24-09-27T1233", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Pure Comm.(us)",
            # metric_name="Overhead",
            fname="64", out_dir="sng"
        )


def main_atos(nodes: int):
    basepath = Path(__file__).parent.parent  # double .parent to get out of benchmark/
    basepath_data = basepath.parent / "data" / "measurements" / "atos"

    save = False
    figsize = (12, 4)

    if nodes == 2:
        _load = functools.partial(load, basepath_data=basepath_data / "2")
        save = True
        # osu_latency, 2 nodes
        lineplot_osu(
            run_baseline=_load(folder="measurements_aa3-4025.bullx_osu_latency_24-10-07T2046", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_aa3-4025.bullx_osu_latency_24-10-07T2046", run_type=RunType.OSU, profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_aa3-4025.bullx_osu_latency_24-10-07T2046", run_type=RunType.OSU, profile="2",
                      tag="tick=100"),
                _load(folder="measurements_aa3-4025.bullx_osu_latency_24-10-07T2046", run_type=RunType.OSU, profile="0",
                      tag="baseline")
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Latency (us)", metric_name="Latency Δ (μs)",
            out_dir="atos", fname="2"
        )

    elif nodes == 4:
        _load = functools.partial(load, basepath_data=basepath_data / "4")
        save = True

        # osu_bcast, 4 nodes, 1 core
        lineplot_osu(
            run_baseline=_load(folder="measurements_aa3-2002.bullx_osu_bcast_24-10-07T1434", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_aa3-2002.bullx_osu_bcast_24-10-07T1434", run_type=RunType.OSU, profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_aa3-2002.bullx_osu_bcast_24-10-07T1434", run_type=RunType.OSU, profile="2",
                      tag="tick=100"),
                _load(folder="measurements_aa3-2002.bullx_osu_bcast_24-10-07T1434", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Avg Latency(us)", metric_name="Latency Δ (μs)",
            fname="4", out_dir="atos"
        )

        # osu_ibcast, 4 nodes, 1 core
        lineplot_osu(
            run_baseline=_load(folder="measurements_aa3-2002.bullx_osu_ibcast_24-10-07T1446", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_aa3-2002.bullx_osu_ibcast_24-10-07T1446", run_type=RunType.OSU, profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_aa3-2002.bullx_osu_ibcast_24-10-07T1446", run_type=RunType.OSU, profile="2",
                      tag="tick=100"),
                _load(folder="measurements_aa3-2002.bullx_osu_ibcast_24-10-07T1446", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Pure Comm.(us)", metric_name="Latency Δ (μs)",
            fname="4", out_dir="atos"
        )

        # osu_bcast, 4 nodes, 127 cores
        lineplot_osu(
            run_baseline=_load(folder="measurements_aa3-2002.bullx_osu_bcast_24-10-07T1452", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_aa3-2002.bullx_osu_bcast_24-10-07T1452", run_type=RunType.OSU,
                      profile="1", tag="tick=20000"),
                _load(folder="measurements_aa3-2002.bullx_osu_bcast_24-10-07T1452", run_type=RunType.OSU,
                      profile="2", tag="tick=100"),
                _load(folder="measurements_aa3-2002.bullx_osu_bcast_24-10-07T1452", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Avg Latency(us)", metric_name="Latency Δ (μs)",
            fname="4_127c", out_dir="atos"
        )

        # osu_ibcast, 4 nodes, 127 cores
        lineplot_osu(
            run_baseline=_load(folder="measurements_aa3-2002.bullx_osu_ibcast_24-10-07T1611", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_aa3-2002.bullx_osu_ibcast_24-10-07T1611", run_type=RunType.OSU,
                      profile="1", tag="tick=20000"),
                _load(folder="measurements_aa3-2002.bullx_osu_ibcast_24-10-07T1611", run_type=RunType.OSU,
                      profile="2", tag="tick=100"),
                _load(folder="measurements_aa3-2002.bullx_osu_ibcast_24-10-07T1611", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Pure Comm.(us)", metric_name="Latency Δ (μs)",
            fname="4_127c", out_dir="atos"
        )


    elif nodes == 16:
        _load = functools.partial(load, basepath_data=basepath_data / "16")
        save = True
        # osu_bcast, 16 nodes
        lineplot_osu(
            run_baseline=_load(folder="measurements_aa1-4008.bullx_osu_bcast_24-10-08T1942", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_aa1-4008.bullx_osu_bcast_24-10-08T1942", run_type=RunType.OSU, profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_aa1-4008.bullx_osu_bcast_24-10-08T1942", run_type=RunType.OSU, profile="2",
                      tag="tick=100"),
                _load(folder="measurements_aa1-4008.bullx_osu_bcast_24-10-08T1942", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Avg Latency(us)", metric_name="Latency Δ (μs)",
            fname="16", out_dir="atos"
        )

        # osu_ibcast, 16 nodes
        lineplot_osu(
            run_baseline=_load(folder="measurements_aa1-4008.bullx_osu_ibcast_24-10-08T2246", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_aa1-4008.bullx_osu_ibcast_24-10-08T2246", run_type=RunType.OSU, profile="1",
                      tag="tick=1024"),
                _load(folder="measurements_aa1-4008.bullx_osu_ibcast_24-10-08T2246", run_type=RunType.OSU, profile="2",
                      tag="tick=1"),
                _load(folder="measurements_aa1-4008.bullx_osu_ibcast_24-10-08T2246", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Pure Comm.(us)", metric_name="Latency Δ (μs)",
            fname="16", out_dir="atos"
        )

    elif nodes == 64:
        _load = functools.partial(load, basepath_data=basepath_data / "64")
        save = True
        # osu_bcast, 64 nodes
        lineplot_osu(
            run_baseline=_load(folder="measurements_aa1-1053.bullx_osu_bcast_24-10-09T0255", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_aa1-1053.bullx_osu_bcast_24-10-09T0255", run_type=RunType.OSU, profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_aa1-1053.bullx_osu_bcast_24-10-09T0255", run_type=RunType.OSU, profile="2",
                      tag="tick=100"),
                _load(folder="measurements_aa1-1053.bullx_osu_bcast_24-10-09T0255", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Avg Latency(us)", metric_name="Latency Δ (μs)",
            fname="64", out_dir="atos"
        )

        # osu_ibcast, 64 nodes
        lineplot_osu(
            run_baseline=_load(folder="measurements_aa1-1053.bullx_osu_ibcast_24-10-09T0003", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_aa1-1053.bullx_osu_ibcast_24-10-09T0003", run_type=RunType.OSU, profile="1",
                      tag="tick=1024"),
                _load(folder="measurements_aa1-1053.bullx_osu_ibcast_24-10-09T0003", run_type=RunType.OSU, profile="2",
                      tag="tick=1"),
                _load(folder="measurements_aa1-1053.bullx_osu_ibcast_24-10-09T0003", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Pure Comm.(us)", metric_name="Latency Δ (μs)",
            fname="64", out_dir="atos"
        )

        # osu_bcast, 64 nodes full
        lineplot_osu(
            run_baseline=_load(folder="measurements_aa1-1053.bullx_osu_bcast_24-10-08T2111", run_type=RunType.OSU,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_aa1-1053.bullx_osu_bcast_24-10-08T2111", run_type=RunType.OSU, profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_aa1-1053.bullx_osu_bcast_24-10-08T2111", run_type=RunType.OSU, profile="2",
                      tag="tick=100"),
                _load(folder="measurements_aa1-1053.bullx_osu_bcast_24-10-08T2111", run_type=RunType.OSU,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Avg Latency(us)", metric_name="Latency Δ (μs)",
            fname="64_full", out_dir="atos"
        )


def main_lumi(nodes: int):
    basepath = Path(__file__).parent.parent  # double .parent to get out of benchmark/
    basepath_data = basepath.parent / "data" / "measurements" / "lumi"

    save = False
    figsize = (12, 4)

    if nodes == 4:
        _load = functools.partial(load, basepath_data=basepath_data / "4")
        save = True

        # osu_bcast, 4 nodes, 1 core
        lineplot_osu(
            run_baseline=_load(folder="measurements_nid001439_osu_bcast_24-10-10T2114", run_type=RunType.OSU, pid=False,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_nid001439_osu_bcast_24-10-10T2114", run_type=RunType.OSU, pid=False,
                      profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_nid001439_osu_bcast_24-10-10T2114", run_type=RunType.OSU, pid=False,
                      profile="2",
                      tag="tick=100"),
                _load(folder="measurements_nid001439_osu_bcast_24-10-10T2114", run_type=RunType.OSU, pid=False,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Avg Latency(us)", metric_name="Latency Δ (μs)",
            fname="4", out_dir="lumi"
        )

        # osu_ibcast, 4 nodes, 1 core
        lineplot_osu(
            run_baseline=_load(folder="measurements_nid001439_osu_ibcast_24-10-10T2118", run_type=RunType.OSU,
                               pid=False,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_nid001439_osu_ibcast_24-10-10T2118", run_type=RunType.OSU, pid=False,
                      profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_nid001439_osu_ibcast_24-10-10T2118", run_type=RunType.OSU, pid=False,
                      profile="2",
                      tag="tick=100"),
                _load(folder="measurements_nid001439_osu_ibcast_24-10-10T2118", run_type=RunType.OSU, pid=False,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Pure Comm.(us)", metric_name="Latency Δ (μs)",
            fname="4", out_dir="lumi"
        )

    elif nodes == 64:
        _load = functools.partial(load, basepath_data=basepath_data / "64")
        save = True

        # osu_bcast, 64 nodes, 1 core
        lineplot_osu(
            run_baseline=_load(folder="measurements_nid001000_osu_bcast_24-10-10T2310", run_type=RunType.OSU,
                               pid=False,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_nid001000_osu_bcast_24-10-10T2310", run_type=RunType.OSU, pid=False,
                      profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_nid001000_osu_bcast_24-10-10T2310", run_type=RunType.OSU, pid=False,
                      profile="2",
                      tag="tick=100"),
                _load(folder="measurements_nid001000_osu_bcast_24-10-10T2310", run_type=RunType.OSU, pid=False,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Avg Latency(us)", metric_name="Latency Δ (μs)",
            fname="64", out_dir="lumi"
        )

        # osu_ibcast, 64 nodes, 1 core
        lineplot_osu(
            run_baseline=_load(folder="measurements_nid001000_osu_ibcast_24-10-10T2323", run_type=RunType.OSU,
                               pid=False,
                               profile="0", tag="baseline"),
            run_b=[
                _load(folder="measurements_nid001000_osu_ibcast_24-10-10T2323", run_type=RunType.OSU, pid=False,
                      profile="1",
                      tag="tick=20000"),
                _load(folder="measurements_nid001000_osu_ibcast_24-10-10T2323", run_type=RunType.OSU, pid=False,
                      profile="2",
                      tag="tick=100"),
                _load(folder="measurements_nid001000_osu_ibcast_24-10-10T2323", run_type=RunType.OSU, pid=False,
                      profile="0", tag="baseline"),
            ],
            basepath=basepath, save=save, title=False, figsize=figsize,
            mode="speedup",
            metric="Pure Comm.(us)", metric_name="Latency Δ (μs)",
            fname="64", out_dir="lumi"
        )


def main_boxplots(run_set):
    basepath = Path(__file__).parent.parent  # double .parent to get out of benchmark/
    basepath_data = basepath.parent / "data" / "measurements"
    save = False
    figsize = (12, 4)
    save = True

    boxplots(run_set=run_set,
             system="atos",
             size="4",
             basepath_data=basepath_data,
             y_decimals=0,
             basepath=basepath, save=save, figsize=figsize,
             fname="colletive_4", out_dir="atos")

    boxplots(run_set=run_set,
             system="sng",
             size="4",
             basepath_data=basepath_data,
             basepath=basepath, save=save, figsize=figsize,
             locator_base=0.05,
             y_decimals=0,
             fname="colletive_4", out_dir="sng")


def main_scaling(run_set):
    basepath = Path(__file__).parent.parent  # double .parent to get out of benchmark/
    basepath_data = basepath.parent / "data" / "measurements"
    save = False
    figsize = (12, 4)
    save = True

    scaling(run_set=run_set,
            system="atos",
            sizes=["4", "16", "64"],
            call="osu_bcast",
            metric="Avg Latency(us)",
            locator_base=0.05,
            y_decimals=0,
            basepath_data=basepath_data,
            basepath=basepath, save=save, figsize=figsize,
            fname="osu_bcast", out_dir="atos")
    scaling(run_set=run_set,
            system="atos",
            sizes=["4", "16", "64"],
            call="osu_ibcast",
            metric="Pure Comm.(us)",
            locator_base=0.10,
            y_decimals=0,
            basepath_data=basepath_data,
            basepath=basepath, save=save, figsize=figsize,
            fname="osu_ibcast", out_dir="atos")

    scaling(run_set=run_set,
            system="sng",
            sizes=["4", "16", "64"],
            call="osu_bcast",
            metric="Avg Latency(us)",
            locator_base=0.10,
            y_decimals=0,
            basepath_data=basepath_data,
            basepath=basepath, save=save, figsize=figsize,
            fname="osu_bcast", out_dir="sng")
    scaling(run_set=run_set,
            system="sng",
            sizes=["4", "16", "64"],
            call="osu_ibcast",
            metric="Pure Comm.(us)",
            locator_base=0.10,
            y_decimals=0,
            basepath_data=basepath_data,
            basepath=basepath, save=save, figsize=figsize,
            fname="osu_ibcast", out_dir="sng")

    scaling(run_set=run_set,
            system="lumi",
            sizes=["4", "16", "64"],
            call="osu_bcast",
            metric="Avg Latency(us)",
            locator_base=0.01,
            y_decimals=0,
            basepath_data=basepath_data,
            basepath=basepath, save=save, figsize=figsize,
            fname="osu_bcast", out_dir="lumi")
    scaling(run_set=run_set,
            system="lumi",
            sizes=["4", "16", "64"],
            call="osu_ibcast",
            metric="Pure Comm.(us)",
            locator_base=0.01,
            y_decimals=0,
            basepath_data=basepath_data,
            basepath=basepath, save=save, figsize=figsize,
            fname="osu_ibcast", out_dir="lumi")


def main():
    run_set = {
        "sng": {
            "4": {
                "osu_bcast": "measurements_3982881_i05r11c03s12_osu_bcast_24-10-02T0923",
                "osu_ibcast": "measurements_i03r11c05s01_osu_ibcast_24-09-27T1033"
            },
            "16": {
                "osu_bcast": "measurements_3991243_i05r09c04s12_osu_bcast_24-10-03T0953",
                "osu_ibcast": "measurements_3984691_i02r11c03s09_osu_ibcast_24-10-03T0127"
            },
            "64": {
                "osu_bcast": "measurements_i07r05c04s09_osu_bcast_24-09-30T1134",
                "osu_ibcast": "measurements_i06r05c04s09_osu_ibcast_24-09-27T1233",
            },
            "4_full": {
                "osu_bcast": "measurements_i01r05c05s10_osu_bcast_24-09-29T1919",
                "osu_ibcast": "measurements_i01r05c03s01_osu_ibcast_24-09-29T1920"
            },

        },
        "atos": {
            "4": {
                "osu_bcast": "measurements_aa3-2002.bullx_osu_bcast_24-10-07T1434",
                "osu_ibcast": "measurements_aa3-2002.bullx_osu_ibcast_24-10-07T1446"
            },
            "16": {
                "osu_bcast": "measurements_aa1-4008.bullx_osu_bcast_24-10-08T1942",
                "osu_ibcast": "measurements_aa1-4008.bullx_osu_ibcast_24-10-08T2246"
            },
            "64": {
                "osu_bcast": "measurements_aa1-1053.bullx_osu_bcast_24-10-09T0255",
                "osu_ibcast": "measurements_aa1-1053.bullx_osu_ibcast_24-10-09T0003",
            },
            "4_full": {
                "osu_bcast": "measurements_aa3-2002.bullx_osu_bcast_24-10-07T1452",
                "osu_ibcast": "measurements_aa3-2002.bullx_osu_ibcast_24-10-07T1611"
            },
        },

        "lumi": {
            "4": {
                "osu_bcast": "measurements_nid001439_osu_bcast_24-10-10T2114",
                "osu_ibcast": "measurements_nid001439_osu_ibcast_24-10-10T2118"
            },
            "16": {
                "osu_bcast": "measurements_nid001768_osu_bcast_24-10-10T2130",
                "osu_ibcast": "measurements_nid001768_osu_ibcast_24-10-10T2137"
            },
            "64": {
                "osu_bcast": "measurements_nid001000_osu_bcast_24-10-10T2310",
                "osu_ibcast": "measurements_nid001000_osu_ibcast_24-10-10T2323",
            },
        }
    }

    main_supermuc(2)
    main_supermuc(4)
    # main_atos(16)
    # main_lumi(64)
    main_boxplots(run_set)
    main_scaling(run_set)

if __name__ == "__main__":
    main()