import datetime
import functools

import pytz

import pandas as pd

from typing import Optional, Callable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.dates
from pathlib import Path

from influxdb_client import InfluxDBClient

from functionCacher.Cacher import Cacher
from mpl_toolkits.axes_grid1 import make_axes_locatable

from visualisation.utils import sizeof_fmt

import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore", MissingPivotFunction)

cacher = Cacher()

matplotlib.rc('font', **{
    'family' : 'sans',
    'size'   : 24})
#matplotlib.use('QtAgg')  # or can use 'TkAgg', whatever you have/prefer


pd.options.display.width = 1920
pd.options.display.max_columns = 99

@cacher.cache(exclude_args=[0])
def load(client, query):
    return client.query_api().query_data_frame(org="caps", query=query )

def query_all(fn: str, measurement: str, start: int, stop: int):
    return f"""
from(bucket: "ofi")
    |> range(start: {start}, stop: {stop})
    |> filter(fn: (r) =>r["_measurement"] == "{measurement}" )
    |> filter(fn: (r) =>r["_field"] == "{fn}" )
    |> group(columns: ["host","pid","provider"])
    |> aggregateWindow(every: 1s, fn: sum, createEmpty: false)
    |> group(columns: ["host","_time", "provider"])
    |> sum(column: "_value")
    |> group(columns: ["host","provider"])
"""

def query_api_calls(measurement: str, start: int, stop: int, node: Optional[str] = None):
    return f"""
from(bucket: "ofi")
    |> range(start: {start}, stop: {stop})
    |> filter(fn: (r) =>r["_measurement"] == "{measurement}" )
 {f'|> filter(fn: (r) =>r["host"] == "{node}" )' if node is not None else ''}
    |> group(columns:["provider", "_field"])
    |> sum(column: "_value")
    |> group()
    |> filter(fn: (r) => r["_value"] > 0)
"""

def query_bucket(measurement: str, field: str, start: int, stop: int, node: Optional[str] = None):
    return f"""
from(bucket: "ofi")
    |> range(start: {start}, stop: {stop})
    |> filter(fn: (r) =>r["_measurement"] == "{measurement}" )
    |> filter(fn: (r) =>r["_field"] == "{field}" )
 {f'|> filter(fn: (r) =>r["host"] == "{node}" )' if node is not None else ''}
    |> group(columns:["provider", "bucket"])
    |> sum(column: "_value")
    |> group()
"""

def query_bucket_ts(measurement: str, field: str, start: int, stop: int, node: Optional[str] = None):
    return f"""
    import "strings"
import "regexp"

suffix_to_value = (v) => {{
  out = if strings.hasSuffix(v:v, suffix: "K") then 
      int(v:strings.trimSuffix(v:v, suffix:"K"))*1024
    else if strings.hasSuffix(v:v, suffix: "M") then 
      int(v:strings.trimSuffix(v:v, suffix:"M"))*1024*1024
  else int(v:v)
  return out
}}

from(bucket: "ofi")
    |> range(start: {start}, stop: {stop})
    |> filter(fn: (r) =>r["_measurement"] == "{measurement}" )
    |> filter(fn: (r) =>r["_field"] == "{field}" )
 {f'|> filter(fn: (r) =>r["host"] == "{node}" )' if node is not None else ''}
    |> group(columns:["_time","provider", "bucket"])
    |> sum(column: "_value")
    |> map(fn: (r) => ({{ r with sort_key: suffix_to_value(v:strings.split(v: r["bucket"], t: "_")[0]) }}))
    |> group(columns: ["provider","sort_key"])

"""

def query_bucket_pid(measurement: str, field: str, start: int, stop: int, node: Optional[str] = None):
    return f"""
    import "strings"
import "regexp"

suffix_to_value = (v) => {{
  out = if strings.hasSuffix(v:v, suffix: "K") then 
      int(v:strings.trimSuffix(v:v, suffix:"K"))*1024
    else if strings.hasSuffix(v:v, suffix: "M") then 
      int(v:strings.trimSuffix(v:v, suffix:"M"))*1024*1024
  else int(v:v)
  return out
}}

from(bucket: "ofi")
    |> range(start: {start}, stop: {stop})
    |> filter(fn: (r) =>r["_measurement"] == "{measurement}" )
    |> filter(fn: (r) =>r["_field"] == "{field}" )
 {f'|> filter(fn: (r) =>r["host"] == "{node}" )' if node is not None else ''}
    |> group(columns:["pid","provider", "bucket"])
    |> sum(column: "_value")
    |> map(fn: (r) => ({{ r with sort_key: suffix_to_value(v:strings.split(v: r["bucket"], t: "_")[0]) }}))
    |> group(columns: ["pid","provider","sort_key"])

"""

def query_node(node: str, fn: str, measurement: str, start: int, stop: int):
    return f"""
from(bucket: "ofi")
    |> range(start: {start}, stop: {stop})
    |> filter(fn: (r) =>r["_measurement"] == "{measurement}" )
    |> filter(fn: (r) =>r["_field"] == "{fn}" )
    |> filter(fn: (r) =>r["host"] == "{node}" )
    |> group(columns: ["pid","provider"])
    |> aggregateWindow(every: 1s, fn: sum, createEmpty: false)
    |> group(columns: ["pid","_time", "provider"])
    |> sum(column: "_value")
    |> group(columns: ["pid","provider"])
    """

def plot(df: pd.DataFrame,
         basepath: Path,
         groupby: Optional[list[str]] = None,
         groupby_idxs: Optional[list[int]] = None,
         fmt: Callable[..., str] = sizeof_fmt,
         locator_base: float = 1074000000.,
         date_format: str = "%H:%M",
         t_from: Optional[datetime.datetime] = None, t_to: Optional[datetime.datetime] = None,
         t_buffer: datetime.timedelta = datetime.timedelta(seconds=20),
         x_locator: matplotlib.ticker.Locator = matplotlib.dates.AutoDateLocator(),
         x_locator_minor: matplotlib.ticker.Locator = matplotlib.dates.AutoDateLocator(),
         colorscheme: str = "tab10",
         figsize: tuple[int,int] = (12, 4),
         legend: bool = False,
         lw_legend: float = 1,
         annotations: Optional[list[datetime.datetime]] = None,
         save: bool = False,
         fname: str = ""):
    fig, ax = plt.subplots(figsize=figsize)
    ax: plt.Axes

    if groupby is None:
        groupby = ["host","provider"]
    if t_from is not None and t_to is not None:
        df = df[df["_time"].between(t_from, t_to)]

    ax.set_prop_cycle(plt.cycler("color", matplotlib.colormaps[colorscheme].colors))
    for i, (idx, group) in enumerate(df.groupby(groupby)):
        ts = group["_time"]
        vals = group["_value"]
        plot_idxs = idx
        if groupby_idxs is not None:
            plot_idxs = []
            for j in groupby_idxs:
                plot_idxs.append(idx[j])

        ax.plot(ts, vals, label=" ".join(list(map(str, plot_idxs))),
                lw=0.75)

    ax.yaxis.grid(True)
    ax.yaxis.grid(True, which="minor", alpha=.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, y: f"{fmt(x, digits=0)}/s"))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=locator_base))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=locator_base/2))

    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(date_format, tz=pytz.timezone("Europe/Berlin")))
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_minor_locator(x_locator_minor)

    ax.tick_params(axis="x", which="major", direction="inout", length=7)

    time = df["_time"]
    ax.set_xlim(left=min(time) - t_buffer,
               right=max(time) + t_buffer)
    ymax = ax.get_ylim()
    if annotations is not None:
        for annot in annotations:
            ax.axvline(x=annot, color="tab:red",
                       linestyle=":", linewidth=4)

    if legend:
        legend = ax.legend(ncols=3, loc="upper right",
                   columnspacing=1,
                   borderaxespad=0.2,
                   handletextpad=0.3,
                   labelspacing=0.25)
        legend: plt.Legend
        for handle in legend.legend_handles:
            handle.set(linewidth=lw_legend)

    fig.tight_layout(pad=0)

    if save:
        plt.savefig(basepath / f"{fname}.pdf")
    else:
        plt.show()

def barplot(df: pd.DataFrame,
            basepath: Path,
            fmt: Callable[..., str] = sizeof_fmt,
            locator_base: float = 1024**2*10,
            save: bool = False,
            figsize: tuple[int,int]=(12, 4),
            fname: str = ""):
    fig, ax = plt.subplots(figsize=figsize)
    ax: plt.Axes


    df.sort_values(by="_value", inplace=True, ascending=False)
    x = list(map(lambda s: s.replace("mon_", ""), df["_field"].values))
    y = df["_value"].values
    ax.barh(x,y, zorder=10)

    ax.xaxis.grid(True, zorder=1)
    ax.xaxis.grid(True, which="minor", alpha=.5, zorder=1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=locator_base))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=locator_base/4))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda i, _: f"{fmt(i, digits=0, space="")}"))
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_xlim(left=0)
    fig.tight_layout(pad=0)

    if save:
        plt.savefig(basepath / f"{fname}.pdf")
    else:
        plt.show()


def heatmap(df: pd.DataFrame,
            basepath: Path,
            groupby: (str,str),
            groupby_labels: (str,str),
            fmt: Callable[..., str] = sizeof_fmt,
            save: bool = False,
            fname: str = ""):
    fig, ax = plt.subplots(figsize=(7,7))
    ax: plt.Axes

    data = []
    for i, (idx, group) in enumerate(df.groupby(groupby[0])):
        dat = group.sort_values(by=groupby[1])["_value"].values
        data.append(dat)

    x_label = df[groupby_labels[1]].unique()
    y_label = df[groupby_labels[0]].unique()

    norm = matplotlib.colors.LogNorm()
    data = np.array(data)
    im = ax.imshow(data, norm=norm)
    ax.set_xticks(np.arange(len(x_label)), labels=x_label, rotation=90)
    ax.set_yticks(np.arange(len(y_label)), labels=y_label)
    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="5%", pad=0.15)
    ax.figure.colorbar(im, cax=cax,
                              format=matplotlib.ticker.FuncFormatter(fmt),
                              ticks=matplotlib.ticker.LogLocator(numticks=10),
                              # ticks=matplotlib.ticker.MultipleLocator(base=locator_base)
                              )


    fig.tight_layout(pad=0)

    if save:
        plt.savefig(basepath / f"{fname}.pdf")
    else:
        plt.show()

def main():
    save = False
    basepath = Path(__file__).parent / "monitoring"
    basepath_data = Path(__file__).parent.parent / "data" / "dfs"
    tz = pytz.timezone("Europe/Berlin")
    load_from_database = False


    if load_from_database:
        client = InfluxDBClient(url="http://localhost:8086",
                                token="ALXSY6dKa3Z1eoP5SeC0SWykvAaQnAmfeP36eyc2LjCGRjo1Rr53kZZIM72QLdZ8d5qEvN2176yh4oWE-MqS5w==",
                                org="caps",
                                timeout=None,
                                enable_gzip=True)
        start = tz.localize(datetime.datetime.strptime("2024-10-04 15:56:49", "%Y-%m-%d %H:%M:%S"))
        stop  = tz.localize(datetime.datetime.strptime("2024-10-04 16:19:30", "%Y-%m-%d %H:%M:%S"))
        df_api_calls_ofi = load(client,
                                query_api_calls("ofi", int(start.timestamp()), int(stop.timestamp())))
        df_api_calls_ofi_sum = load(client,
                                    query_api_calls("ofi_sum", int(start.timestamp()), int(stop.timestamp())))

        df_tsenddata_all = load(client,
                                query_all("mon_tsenddata", "ofi_sum", int(start.timestamp()), int(stop.timestamp())))

        df_node_tsenddata = load(client,
                                 query_node("aa3-1016.bullx", "mon_tsenddata", "ofi_sum", int(start.timestamp()), int(stop.timestamp())))
        df_node_trecv = load(client,
                             query_node("aa3-1016.bullx", "mon_trecv", "ofi", int(start.timestamp()), int(stop.timestamp())))

        df_bucket_tsenddata_ts = load(client,
                                      query_bucket_ts("ofi_sum", "mon_tsenddata",
                                                      int(start.timestamp()), int(stop.timestamp()),
                                                      node="aa3-1016.bullx"))
        df_bucket_tsenddata_pid = load(client,
                                       query_bucket_pid("ofi_sum", "mon_tsenddata",
                                                        int(start.timestamp()), int(stop.timestamp()),
                                                        node="aa3-1016.bullx"))
    else:
        df_api_calls_ofi = pd.read_csv(basepath_data / "df_api_calls_ofi.csv.xz", compression="xz", parse_dates=[1,2], date_format="ISO8601")
        df_api_calls_ofi_sum = pd.read_csv(basepath_data / "df_api_calls_ofi_sum.csv.xz", compression="xz", parse_dates=[1,2], date_format="ISO8601")
        df_tsenddata_all = pd.read_csv(basepath_data / "df_tsenddata_all.csv.xz", compression="xz", parse_dates=[1,2], date_format="ISO8601")
        df_node_tsenddata = pd.read_csv(basepath_data / "df_node_tsenddata.csv.xz", compression="xz", parse_dates=[1,2], date_format="ISO8601")
        df_node_trecv = pd.read_csv(basepath_data / "df_node_trecv.csv.xz", compression="xz", parse_dates=[1,2], date_format="ISO8601")
        df_bucket_tsenddata_ts = pd.read_csv(basepath_data / "df_bucket_tsenddata_ts.csv.xz", compression="xz", parse_dates=[1,2], date_format="ISO8601")
        df_bucket_tsenddata_pid = pd.read_csv(basepath_data / "df_bucket_tsenddata_pid.csv.xz", compression="xz", parse_dates=[1,2], date_format="ISO8601")


    save = True
    # heatmap(df=df_bucket_tsenddata_pid,
    #         groupby=["pid", "sort_key"],
    #         groupby_labels=["pid","bucket"],
    #         basepath=basepath,
    #         locator_base=1074000000*8,
    #         fmt=lambda x, y: sizeof_fmt(x, suffix="B", digits=0),
    #         save=save)

    barplot(df_api_calls_ofi,
            basepath=basepath,
            save=save,
            locator_base=1024**2*25,
            fmt=functools.partial(sizeof_fmt, suffix=""),
            fname="api_calls_ofi",
            figsize=(5, 4),
            )

    barplot(df_api_calls_ofi_sum,
            basepath=basepath,
            save=save,
            locator_base=1024**4*10,
            fmt=functools.partial(sizeof_fmt, suffix="B"),
            fname="api_calls_ofi_sum",
            figsize=(5, 4),
            )


    plot(df_tsenddata_all,
         basepath=basepath,
         save=save,
         fname="tsenddata",
         locator_base=1074000000 * 2,
         annotations=[
             tz.localize(datetime.datetime.strptime("2024-10-04 16:05:30", "%Y-%m-%d %H:%M:%S"))
         ]
         )
    plot(df_tsenddata_all,
         basepath=basepath,
         save=save,
         fname="tsenddata_zoom",
         locator_base=1074000000 ,
         t_buffer=datetime.timedelta(seconds=1),
         date_format="%H:%M:%S",
         x_locator=matplotlib.dates.SecondLocator(bysecond=[10*x for x in range(0,6)]),
         x_locator_minor=matplotlib.dates.SecondLocator(interval=1),
         t_from=tz.localize(datetime.datetime.strptime("2024-10-04 16:07:38", "%Y-%m-%d %H:%M:%S")),
         t_to=tz.localize(datetime.datetime.strptime("2024-10-04 16:08:15", "%Y-%m-%d %H:%M:%S")),
         annotations=[
             tz.localize(datetime.datetime.strptime("2024-10-04 16:07:57", "%Y-%m-%d %H:%M:%S"))
         ]

         )

    plot(df=df_node_tsenddata,
         groupby=["pid","provider"],
         fmt=functools.partial(sizeof_fmt, suffix="B"),
         locator_base=1074000000,
         t_from=tz.localize(datetime.datetime.strptime("2024-10-04 16:04:33", "%Y-%m-%d %H:%M:%S")),
         t_to=tz.localize(datetime.datetime.strptime("2024-10-04 16:05:43", "%Y-%m-%d %H:%M:%S")),
         t_buffer=datetime.timedelta(seconds=1),
         date_format="%H:%M:%S",
         x_locator=matplotlib.dates.SecondLocator(bysecond=[0,15,30,45],interval=1),
         x_locator_minor=matplotlib.dates.SecondLocator(bysecond=[5 * i for i in range(int(60 / 5))], interval=1),
         fname="tsenddata_node_focus",
         save=save,
         basepath=basepath
         )

    plot(df=df_node_trecv,
         groupby=["pid","provider"],
         fmt=functools.partial(sizeof_fmt, suffix="calls"),
         locator_base=512,
         t_from=tz.localize(datetime.datetime.strptime("2024-10-04 16:06:40", "%Y-%m-%d %H:%M:%S")),
         t_to=tz.localize(datetime.datetime.strptime("2024-10-04 16:09:15", "%Y-%m-%d %H:%M:%S")),
         t_buffer=datetime.timedelta(seconds=1),
         date_format="%H:%M:%S",
         x_locator=matplotlib.dates.SecondLocator(bysecond=[0,30],interval=1),
         x_locator_minor=matplotlib.dates.SecondLocator(bysecond=[5*i for i in range(int(60/5))], interval=1),
         fname="trecv_node_focus",
         save=save,
         basepath=basepath
         )

    plot(df=df_bucket_tsenddata_ts,
         groupby=["sort_key","bucket",],
         groupby_idxs=[1],
         legend=True,
         fmt=functools.partial(sizeof_fmt, suffix="B"),
         locator_base=1074000000 /2,
         t_from=tz.localize(datetime.datetime.strptime("2024-10-04 16:06:40", "%Y-%m-%d %H:%M:%S")),
         t_to=tz.localize(datetime.datetime.strptime("2024-10-04 16:09:15", "%Y-%m-%d %H:%M:%S")),
         t_buffer=datetime.timedelta(seconds=1),
         date_format="%H:%M:%S",
         x_locator=matplotlib.dates.SecondLocator(bysecond=[0, 30], interval=1),
         x_locator_minor=matplotlib.dates.SecondLocator(bysecond=[0,10,20,30,40,50],interval=1),
         fname="tsenddata_node_bucket",
         colorscheme="tab10_r",
         lw_legend=2,
         save=save,
         basepath=basepath
         )

if __name__ == '__main__': main()