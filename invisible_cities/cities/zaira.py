"""
-----------------------------------------------------------------------
                               Zaira
-----------------------------------------------------------------------
Zaira is the "City of Memories", a place defined entirely by its past,
where every street, wall, and object holds the memories and stories of
its people, like lines in a hand

This city reads Sophronia hits and applies the new relevant tooling
before extracting topology information.
"""

import numpy as np
import tables as tb
import pandas as pd

from os.path import expandvars
from scipy.stats import multivariate_normal
from numpy import nan_to_num

from .components import city
from .components import collect
from .components import copy_mc_info
from .components import print_every
from .components import hits_corrector
from .components import hits_thresholder
from .components import hits_and_kdst_from_files
from .components import compute_and_write_tracks_info
from .components import identity

from ..types.symbols import HitEnergy

from ..core.configure import EventRangeType
from ..core.configure import OneOrManyFiles
from ..core import tbl_functions as tbl
from ..evm import event_model as evm
from ..dataflow import dataflow as fl

from ..dataflow.dataflow import push
from ..dataflow.dataflow import pipe

from ..reco.hits_functions import cut_over_Q
from ..reco.hits_functions import drop_isolated

from ..io.run_and_event_io import run_and_event_writer
from ..io.hits_io import hits_writer
from ..io.hits_io import hits_from_df
from ..io.hits_io import hitc_from_df
from ..io.kdst_io import kdst_from_df_writer

from typing import List
from typing import Optional


# Temporary. The removal of the event model will fix this.
def hitc_to_df_(hitc):
    # Si ya es DF, solo aseguramos schema
    if isinstance(hitc, pd.DataFrame):
        df = hitc.copy()
    else:
        rows = []
        for hit in hitc.hits:
            rows.append({
                "event"   : hitc.event,
                "time"    : hitc.time,
                "npeak"   : hit.npeak,
                "Xpeak"   : hit.Xpeak,
                "Ypeak"   : hit.Ypeak,
                "nsipm"   : hit.nsipm,
                "X"       : hit.X,
                "Y"       : hit.Y,
                "Xrms"    : hit.Xrms,
                "Yrms"    : hit.Yrms,
                "Z"       : hit.Z,
                "Q"       : hit.Q,
                "E"       : hit.E,
                "Qc"      : hit.Qc,
                "Ec"      : hit.Ec,
                "track_id": hit.track_id,
                "Ep"      : hit.Ep,
            })
        df = pd.DataFrame.from_records(rows)

    # CAST al schema exacto de /CHITS/highTh
    df["event"]    = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype(np.int64)
    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce").fillna(0).astype(np.int64)
    df["npeak"]    = pd.to_numeric(df["npeak"], errors="coerce").fillna(0).astype(np.uint16)
    df["nsipm"]    = pd.to_numeric(df["nsipm"], errors="coerce").fillna(0).astype(np.uint16)

    float_cols = ["time","Xpeak","Ypeak","X","Y","Xrms","Yrms","Z","Q","E","Qc","Ec","Ep"]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64)

    return df



@city
def zaira(
    files_in: OneOrManyFiles,
    file_out: str,
    compression: str,
    event_range: EventRangeType,
    print_mod: int,
    detector_db: str,
    run_number: int,
    threshold: float,
    drop_distance: List[float],
    drop_minimum: int,
    paolina_params: dict,
    corrections: Optional[dict] = None,
):

    # mapping functionals
    hitc_to_df = fl.map(hitc_to_df_, item="hits")
    df_to_hitc = fl.map(hitc_from_df, item="hits")
    correct_hits = fl.map(
        hits_corrector(**corrections) if corrections is not None else identity,
        item="hits",
    )

    cut_sensors = fl.map(cut_over_Q(threshold, ["E", "Ec"]), item="hits")
    drop_sensors = fl.map(
        drop_isolated(drop_distance, ["E", "Ec"], drop_minimum), item="hits"
    )

    # spy components
    event_count_in = fl.spy_count()
    event_count_post_cuts = fl.spy_count()
    event_count_post_topology = fl.count()

    filter_out_none = fl.filter(lambda x: x is not None, args="kdst")
    event_number_collector = collect()
    collect_evts = "event_number", fl.fork(
        event_number_collector.sink,
        event_count_post_topology.sink,
    )

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:
        hits_writer_effect = hits_writer(h5out, group_name="CHITS", table_name="highTh")

        write_event_info = fl.sink(
            run_and_event_writer(h5out),
            args=("run_number", "event_number", "timestamp"),
        )

        # Write hits as DF
        write_hits_df = (
            fl.map(hitc_to_df_, item="hits"),
            fl.sink(hits_writer_effect, args="hits"),
        )

        write_kdst_table = fl.sink(kdst_from_df_writer(h5out), args="kdst")

        compute_tracks = compute_and_write_tracks_info(
            paolina_params,
            h5out,
            hit_type=HitEnergy.Ec,
            filter_hits_table_name="high_th_select",
            hits_writer=hits_writer_effect,
        )

        result = push(
            source=hits_and_kdst_from_files(files_in, "RECO", "Events"),
            pipe=pipe(
                fl.slice(*event_range, close_all=True),
                print_every(print_mod),
                event_count_in.spy,
                correct_hits,
                hitc_to_df,
                cut_sensors,
                drop_sensors,
                df_to_hitc,              # back to HitCollection for topology
                event_count_post_cuts.spy,
                fl.fork(
                    compute_tracks,                     # needs HitCollection
                    write_hits_df,                      # converts to DF only for writing
                    (filter_out_none, write_kdst_table),
                    write_event_info,
                    collect_evts,
                ),
            ),
            result=dict(
                events_in=event_count_in.future,
                events_out=event_count_post_cuts.future,
                evtnum_list=event_number_collector.future,
            ),
        )

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list, detector_db, run_number)

        return result
