import enum
import re
import io
from pathlib import Path

import pandas as pd
from functionCacher.Cacher import Cacher

from visualisation.classes import Run
cacher = Cacher()

class RunType(enum.Enum):
    LULESH=1
    NPB_FT=4
    CXI=5
    OSU=6
    AMG=7

@cacher.cache
def load_lulesh(folder: str, basepath_data: Path) -> Run:
    pattern_fom = re.compile(r"FOM\s+=\s+?([\d\.]+) \(z\/s\)", flags=re.MULTILINE)
    data = []
    with open(basepath_data / folder / "meta.md", "r") as infile:
        content = infile.read()

        cores = re.compile("Cores: `(.*)`").search(content).group(1)
        # if len(cores) > 15:
        #     cores = len(cores.split(","))
        name = "LULESH " + f"{cores}"
    for file in list(filter(lambda x: x.suffix == ".dat", (basepath_data / folder).iterdir())):
        with open(file, "r") as infile:
            content = infile.read()
            datum = {"i": int(file.stem.split("_")[1])}

            match = pattern_fom.search(content)
            datum["fom"] = float(match.group(1))

            data.append(datum)
    df = pd.DataFrame(data).set_index("i").sort_index()

    return Run(df=df, name=name)


@cacher.cache
def load_npb_ft(folder: str, basepath_data: Path, **kwargs) -> Run:
    data = []
    with open(basepath_data / folder / "meta.md", "r") as infile:
        content = infile.read()
        benchmark_class = re.compile(r"Binary:.*?\/bin\/(.*?)\.x",
                                     flags=re.MULTILINE).search(content).group(1)
        name = f"NPB {benchmark_class} "
        if run_id := kwargs.get("run_id"):
            name += run_id
        else:
            name += re.compile("Cores: `(.*)`").search(content).group(1)

    for file in list(filter(lambda x: x.suffix == ".dat", (basepath_data / folder).iterdir())):
        with open(file, "r") as infile:
            pattern_copy = re.compile(r"Mop/s/thread\s+=\s+([\d\.]+).+",
                                      flags=re.MULTILINE)
            content = infile.read()
            datum = {"i": int(file.stem.split("_")[1]),
                     "mop/s/thread": float(pattern_copy.search(content).group(1))}

            data.append(datum)
    df = pd.DataFrame(data).set_index("i").sort_index()

    return Run(df=df, name=name)



@cacher.cache
def load_cxi(folder: str, basepath_data: Path, **kwargs) -> Run:
    dfs = []
    with open(basepath_data / folder / "meta.md", "r") as infile:
        content = infile.read()
        name = re.compile("`(cxi_.*) ").search(content).group(1) + " "
        if run_id := kwargs.get("run_id"):
            name += run_id
        else:
            name += re.compile("Cores: `(.*)`").search(content).group(1)

    search_field = kwargs.get("search_field", "RDMA")
    for file in list(filter(lambda x: x.suffix == ".dat", (basepath_data / folder).iterdir())):
        with open(file, "r") as infile:
            lines = infile.readlines()
            rdma_idx = -1
            for idx, line in enumerate(lines):
                if line.startswith(search_field):
                    rdma_idx = idx
                    break
            if rdma_idx == -1:
                raise IndexError(f"Could not find {search_field} in {file}")
            measurement_lines = lines[rdma_idx:-1]
            sio = io.StringIO()
            pattern = re.compile(r"(\s{2,})")
            for line in measurement_lines:
                sio.write(re.sub(pattern, "\\t", line).lstrip())
            sio.seek(0)
            df = pd.read_csv(sio, sep="\t")
            df.insert(loc=len(df.columns)-1, column="i", value=int(file.stem.split("_")[1]))
            dfs.append(df)
    out_df = pd.concat(dfs).set_index(["i", df.columns[0]]).sort_index()

    return Run(df=out_df, name=name)

@cacher.cache
def load_osu(folder: str, basepath_data: Path, **kwargs) -> Run:
    dfs = []
    name = ""
    with open(basepath_data / folder / "meta.md", "r") as infile:
        content = infile.read()
        name = re.compile(r"Binary: `.*\/(.*?)`").search(content).group(1) + " "

    file_dir = basepath_data / folder
    if kwargs.get("profile"):
        file_dir = file_dir / f'profile_{kwargs.get("profile")}'

    for file in list(filter(lambda x: x.suffix == ".dat", file_dir.iterdir())):
        with open(file, "r") as infile:
            # if int(file.stem.split("_")[1]) > 4: continue
            lines = infile.readlines()
            rdma_idx = -1
            if kwargs.get("pid", False):
                pid = int(lines[0])
            
            lines = lines[1:]
            for idx, line in enumerate(lines):
                if line.startswith("# Size"):
                    rdma_idx = idx
                    break
            if rdma_idx == -1:
                raise IndexError(f"Could not find '# Size' in {file}")
            measurement_lines = lines[rdma_idx:]
            if measurement_lines[1].startswith("#"):
                measurement_lines = [measurement_lines[0]] + measurement_lines[2:]
            sio = io.StringIO()
            pattern = re.compile(r"(\s{2,})")
            for line in measurement_lines:
                sio.write(re.sub(pattern, "\\t", line).lstrip())
            sio.seek(0)
            df = pd.read_csv(sio, sep="\t")
            df.insert(loc=len(df.columns)-1, column="i", value=int(file.stem.split("_")[1]))
            dfs.append(df)
    out_df = pd.concat(dfs).set_index(["i", df.columns[0]]).sort_index()

    return Run(df=out_df, name=name)

@cacher.cache
def load_amg(folder: str, basepath_data: Path, **kwargs) -> Run:
    data = []
    with open(basepath_data / folder / "meta.md", "r") as infile:
        content = infile.read()
        name = f"AMG2013"
        if run_id := kwargs.get("run_id"):
            name += run_id
        else:
            name += re.compile("Cores: `(.*)`").search(content).group(1)

    for file in list(filter(lambda x: x.suffix == ".dat", (basepath_data / folder).iterdir())):
        with open(file, "r") as infile:
            pattern_fom = re.compile(r"System Size \* Iterations / Solve Phase Time:\s+(.*?)$",
                                      flags=re.MULTILINE)
            content = infile.read()
            datum = {"i": int(file.stem.split("_")[1]),
                     "FOM": float(pattern_fom.search(content).group(1))}

            data.append(datum)
    df = pd.DataFrame(data).set_index("i").sort_index()

    return Run(df=df, name=name)



def load(folder: str, basepath_data: Path, run_type: RunType, tag: str = "", **kwargs) -> Run:
    """
    Return Run instance as loaded depending on the RunType / custom loader
    """
    match run_type:
        case RunType.LULESH:
            fn = load_lulesh
        case RunType.NPB_FT:
            fn = load_npb_ft
        case RunType.CXI:
            fn = load_cxi
        case RunType.OSU:
            fn = load_osu
        case RunType.AMG:
            fn = load_amg
        case _:
            raise Exception()

    run = fn(folder, basepath_data, **kwargs)
    run.tag = tag
    return run