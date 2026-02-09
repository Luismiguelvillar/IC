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
from .components import Efield_copier
from .components import identity

from ..types.symbols import HitEnergy

from ..core.configure import EventRangeType
from ..core.configure import OneOrManyFiles
from ..core import system_of_units as units
from ..core import tbl_functions as tbl
from ..evm import event_model as evm
from ..evm.nh5 import KrTable
from ..evm.nh5 import HitsTable
from ..dataflow import dataflow as fl

from ..dataflow.dataflow import push
from ..dataflow.dataflow import pipe

from ..reco.hits_functions import cut_over_Q
from ..reco.hits_functions import drop_isolated
from ..reco.hits_functions import drop_hits_satellites_xy_z_variable
from ..reco.hits_functions import drop_satellite_clusters

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

    # CAST to HitsTable schema (robustly force exact dtypes)
    hits_dtype = tb.dtype_from_descr(HitsTable)
    for name in hits_dtype.names:
        if name not in df.columns:
            continue
        col = pd.to_numeric(df[name], errors="coerce")
        if np.issubdtype(hits_dtype[name], np.integer):
            df[name] = col.fillna(0).astype(hits_dtype[name])
        else:
            df[name] = col.astype(hits_dtype[name])

    df = df[list(hits_dtype.names)]
    df = pd.DataFrame.from_records(df.to_records(index=False).astype(hits_dtype))
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
    correct_hits = fl.map(hits_corrector(**corrections) if corrections is not None else identity,item="hits")

    cut_sensors = fl.map(cut_over_Q(threshold, ["E", "Ec"]), item="hits")
    drop_sensors = fl.map(drop_isolated(drop_distance, ["E", "Ec"], drop_minimum), item="hits")

    copy_ep = fl.map(Efield_copier(HitEnergy.Ec), item="hits")
    drop_satellites_cluster = fl.map(lambda h: drop_satellite_clusters(h,
                                                                        r_iso=22 * units.mm,
                                                                        method="top_n",
                                                                        keep_top_n=2,
                                                                        frac_min=0.0,
                                                                        n_hits_min=5,
                                                                        e_min=0.0,
                                                                        redistribute_all=True,
                                                                        redistribute_weighted=True,),item="hits")
    
    drop_satellites = fl.map(lambda h: drop_hits_satellites_xy_z_variable(h,
                                                                        5*units.pes,
                                                                        22*units.mm,
                                                                        thr_percentile=40,
                                                                        n_neigh_thr=5,
                                                                        redistribute_weighted=True,
                                                                        redistribute_all=True), item='hits')
    
    # spy components
    event_count_in = fl.spy_count()
    event_count_post_cuts = fl.spy_count()
    event_count_post_topology = fl.count()

    filter_out_none = fl.filter(lambda x: x is not None, args="kdst")
    kr_dtype = tb.dtype_from_descr(KrTable)
    def _coerce_kdst_types(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for name in kr_dtype.names:
            if name not in out.columns:
                continue
            col = pd.to_numeric(out[name], errors="coerce")
            if np.issubdtype(kr_dtype[name], np.integer):
                out[name] = col.fillna(0).astype(kr_dtype[name])
            else:
                out[name] = col.astype(kr_dtype[name])
        return out[list(kr_dtype.names)]

    coerce_kdst_types = fl.map(_coerce_kdst_types, item="kdst")
    event_number_collector = collect()
    collect_evts = "event_number", fl.fork(event_number_collector.sink,event_count_post_topology.sink)

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:
        hits_dtype = tb.dtype_from_descr(HitsTable)
        _hits_writer = hits_writer(h5out, group_name="CHITS", table_name="highTh")
        def hits_writer_effect(hits_df):
            # Ensure consistent dtypes for table creation and subsequent appends
            arr = hits_df.to_records(index=False).astype(hits_dtype)
            hits_df = pd.DataFrame.from_records(arr)
            return _hits_writer(hits_df)

        write_event_info = fl.sink(
            run_and_event_writer(h5out),
            args=("run_number", "event_number", "timestamp"),
        )

        # Write hits as DF
        write_hits_df = (
            fl.map(hitc_to_df_, item="hits"),
            fl.sink(hits_writer_effect, args="hits"),
        )

        _kdst_writer = kdst_from_df_writer(h5out)
        def kdst_writer_effect(kdst_df):
            if not hasattr(kdst_writer_effect, "_printed"):
                kdst_writer_effect._printed = True
                if "DST" in h5out.root and "Events" in h5out.root.DST:
                    print("DST/Events table dtype:", h5out.root.DST.Events.dtype)
                print("DST/Events df dtypes:", kdst_df.dtypes.to_dict())
            return _kdst_writer(kdst_df)
        write_kdst_table = fl.sink(kdst_writer_effect, args="kdst")

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
                #cut_sensors, # NOTE
                #drop_sensors, # NOTE
                df_to_hitc,              # back to HitCollection for topology
                copy_ep,
                drop_satellites_cluster,
                drop_satellites, # NOTE este es el mio!
                event_count_post_cuts.spy,
                fl.fork(
                    compute_tracks,                     # needs HitCollection
                    write_hits_df,                      # converts to DF only for writing
                    (filter_out_none, coerce_kdst_types, write_kdst_table),
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
