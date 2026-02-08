import numpy  as np
import pandas as pd
import networkx as nx

from numba                  import njit
from scipy.spatial          import cKDTree
from scipy.spatial.distance import cdist

from functools import reduce

from .. types.ic_types      import NN
from .. evm.event_model     import HitCollection

from typing                 import Optional, Callable, List 

EPSILON = np.finfo(np.float64).eps

def cut_and_redistribute_df(cut_condition : str,
                            variables     : List[str]=[]) -> Callable:
    '''
    Apply a cut condition to a dataframe and redistribute the cut out values
    of a given variable.

    Parameters
    ----------
    df      : dataframe to be cut

    Initialization parameters:
        cut_condition : String with the cut condition (example "Q > 10")
        variables     : List with variables to be redistributed.

    Returns
    ----------
    pass_df : dataframe after applying the cut and redistribution.
    '''
    def cut_and_redistribute(df : pd.DataFrame) -> pd.DataFrame:
        pass_df = df.query(cut_condition).copy()
        if not len(pass_df): return pass_df

        with np.errstate(divide='ignore'):
            columns  =      pass_df.loc[:, variables]
            columns *= np.divide(df.loc[:, variables].sum().values, columns.sum())
            pass_df.loc[:, variables] = columns

        return pass_df

    return cut_and_redistribute

def cut_over_Q(q_cut, redist_var):
    """
    Apply a cut over the SiPM charge condition to hits and redistribute the
    energy variables.

    Parameters
    ----------
    q_cut      : Charge value over which to cut.
    redist_var : List with variables to be redistributed.

    Returns
    ----------
    cut_over_Q : Function that will cut the dataframe and redistribute
    values.
    """
    cut = cut_and_redistribute_df(f"Q > {q_cut}", redist_var)

    def cut_over_Q(df):  # df shall be an event cdst
        cdst = df.groupby(['event', 'npeak']).apply(cut).reset_index(drop=True)

        return cdst

    return cut_over_Q

def drop_isolated( distance   : List[float],
                   redist_var : Optional[List] = [],
                   nhits      : Optional[int] = 3):
    """
    Master function deciding whether to drop isolated
    hits or clusters, dependent on list provided.

    # Comment -> I think this should be changed, the logic
    for how you choose clusters or sensors is silly.

    Parameters
    ----------
    distance   : Sensor pitch in 2 or 3 dimensions
    redist_var : List with variables to be redistributed.
    nhits      : Number of hits 
    Returns
    ----------
    drop_isolated_sensors : Function that will drop the isolated sensors.
    """
    
    # distance is XY -> N
    if   len(distance) == 2:
        drop = drop_isolated_sensors(distance, redist_var)
    elif len(distance) == 3:
        drop = drop_isolated_clusters(distance, nhits, redist_var)
    else:
        raise ValueError(f"Invalid drop_dist parameter: expected 2 or 3 entries, but got {len(distance)}.")


    def drop_isolated(df): # df shall be an event cdst
        df = df.groupby(['event', 'npeak']).apply(drop).reset_index(drop=True)

        return df

    return drop_isolated

def drop_isolated_sensors(distance  : List[float]=[10., 10.],
                          variables : List[str  ]=[        ]) -> Callable:
    """
    Drops rogue/isolated hits (SiPMs) from a groupedby dataframe.

    Parameters
    ----------
    df      : GroupBy ('event' and 'npeak') dataframe

    Initialization parameters:
        distance  : Distance to check for other sensors. Usually equal to sensor pitch.
        variables : List with variables to be redistributed.

    Returns
    -------
    pass_df : hits after removing isolated hits
    """
    dist = np.sqrt(distance[0] ** 2 + distance[1] ** 2) # TODO Maybe use dist2 to prevent using np.sqrt millions of times

    def drop_isolated_sensors(df : pd.DataFrame) -> pd.DataFrame:
        x       = df.X.values
        y       = df.Y.values
        xy      = np.column_stack((x,y))
        dr2     = cdist(xy, xy) # compute all square distances

        if not np.any(dr2>0):
            return df.iloc[:0] # Empty dataframe

        closest = np.apply_along_axis(lambda d: d[d > 0].min(), 1, dr2) # find closest that it's not itself
        mask_xy = closest <= dist # take those with at least one neighbour
        pass_df = df.loc[mask_xy, :].copy()

        with np.errstate(divide='ignore'): # TODO Hay que buscar mejores formas de handle errors
            columns  = pass_df.loc[:, variables]
            columns *= np.divide(df.loc[:, variables].sum().values, columns.sum())
            pass_df.loc[:, variables] = columns

        return pass_df

    return drop_isolated_sensors


def drop_isolated_clusters(distance   :  List[float],
                           nhits      :  int,
                           variables  :  List[str  ]) -> Callable:
    '''
    Drop isolated hits/clusters, where a cluster is defined by the proximity 
    between hits defined by  distance. A cluster is considered isolated if 
    the number of hits within the cluster is less than or equal to nhits.
    The method uses networkx's graph system to identify clusters.

    Parameters
    ----------
    df : Groupby ('event' and 'npeak') dataframe

    Initialisation parameters:
        distance  : Distance to check for other sensors, equal to sensor pitch and z rebinning.
        nhits     : Number of hits to classify a cluster.
        variables : List of variables to be redistributed (generally the energies)
    '''

   
    def drop_event(df):
        # normalise (x,y,z) array
        xyz = df[list("XYZ")].values / distance

        # build KDTree of datapoints, collect pairs within normalised distance (sqrt of 3)
        pairs = cKDTree(xyz).query_pairs(r = np.sqrt(3))
        
        # create graph that connects all close pairs between hit positions based on df index
        cluster_graph = nx.Graph()
        cluster_graph.add_nodes_from(range(len(df)))
        cluster_graph.add_edges_from((df.index[i], df.index[j]) for i,j in pairs)

        # Find all clusters within the graph
        clusters = nx.connected_components(cluster_graph)

        # collect indices of passing hits (cluster > nhit) within set
        passing_hits = reduce(set.union, filter(lambda x: len(x)>nhits, clusters))

        # apply mask to df to only include passing clusters      
        pass_df = df.loc[passing_hits, :].copy()

        # reweighting
        with np.errstate(divide='ignore'):
            columns = pass_df.loc[:, variables]
            columns *= np.divide(df.loc[:, variables].sum().values, columns.sum())
            pass_df.loc[:, variables] = columns

        return pass_df

    return drop_event

def e_from_q(qs: np.ndarray, e: float) -> np.ndarray:
    """
    Distribute some energy among the hits according to their charge.

    Parameters
    ----------
    qs: np.ndarray, shape (n,)
        The charge of each hit.

    e_slice: float
        The energy to be shared, typically of a given slice.

    Returns
    -------
    es: np.ndarray, shape (n,)
        The associated hit energy.
    """
    return qs * e / (qs.sum() + EPSILON)


def sipms_above_threshold(xys: np.ndarray, qs: np.ndarray, thr:float, energy: float):
    """
    Finds SiPMs with charge above threshold and returns their position, charge
    and associated energy.

    Parameters
    ----------
    xys: np.ndarray, shape (n,2)
        SiPM positions
    qs: np.ndarray, shape (n,)
        Charge of each SiPM.
    thr: float
        Threshold on SiPM charge.
    energy: float
        Energy to be shared among the hits.

    Returns
    -------
    xs: np.ndarray, shape (m,)
        x positions of the SiPMs above threshold
    ys: np.ndarray, shape (m,)
        y positions of the SiPMs above threshold
    qs: np.ndarray, shape (m,)
        Charge of the SiPMs above threshold
    es: np.ndarray, shape (m,)
        Associated energy of each hit
    """
    over_thr = qs >= thr
    nonempty = np.any(over_thr)

    xy =        xys[over_thr]
    qs =         qs[over_thr] if nonempty else [NN]
    xs =             xy[:, 0] if nonempty else [NN]
    ys =             xy[:, 1] if nonempty else [NN]
    es = e_from_q(qs, energy) if nonempty else [energy]
    return xs, ys, qs, es




def merge_NN_hits(hits: pd.DataFrame, same_peak: bool = True) -> pd.DataFrame:
    """
    Finds NN hits (defined as hits with Q=NN) and removes them without energy
    losses. The energy of each NN hit (both E and Ec) is distributed to the
    closest non-NN hit or hits (if many, they must be at exactly the same
    distance). This is done proportionally to the energy of the receiving non-NN
    hits.

    The definition of closest hit can be tweaked by the `same_peak` argument,
    which determiness if the receiving hits must be in the same S2 peak or can
    be from any S2 within the same event.

    If the input contains only NN hits, the output is empty.

    Parameters
    ----------
    hits: pd.DataFrame
        Input hits. Must include at least the following columns:
        `Q`, `npeak`, `Z`, `E`, and `Ec`.

    same_peak: bool, optional
        If `True`, only hits within the same S2 peak as the `NN` hit are
        considered for merging. If `False`, all hits are considered regardless
        of `npeak`. Default is True.

    Returns
    -------
    merged_hits: pd.DataFrame
        A copy of the input with NN hits removed and energy reassigned.

    Notes
    -----
    - The merging process conserves the total `E` and `Ec` across all hits.
    - If a `NN` hit does not have any neighbours, the `NN` hit is effectively
      dropped.
    """
    sel = hits.Q == NN
    if not np.any(sel): return hits # save some time

    nn_hits = hits.loc[ sel]
    hits    = hits.loc[~sel].copy()

    corrections = pd.DataFrame(dict(E=0, Ec=0), index=hits.index.values)
    for _, nn_hit in nn_hits.iterrows():
        candidates = hits.loc[hits.npeak == nn_hit.npeak] if same_peak else hits
        if len(candidates) == 0: continue # drop hit !!! dangerous

        # find closest hit or hits
        dz      = np.abs(candidates.Z - nn_hit.Z)
        closest = candidates.loc[np.isclose(dz, dz.min())]
        index   = closest.index

        # redistribute energy proportionally to the receiving hits' energy
        # corrections are accumulated to make this process order insentitive
        corr_e  = nn_hit.E  * closest.E  / closest.E .sum()
        corr_ec = nn_hit.Ec * closest.Ec / closest.Ec.sum()
        corrections.loc[index, "E Ec".split()] += np.stack([corr_e, corr_ec], axis=1)

    # apply correction factors based on original charge values
    hits.loc[:, "E Ec".split()] += corrections.values
    return hits


def empty_hit( event : int  , timestamp: float, peak_no: int
             , x_peak: float, y_peak   : float, z      : float
             , e     : float, ec       : float):
    """
    Produces an empty hit with NN x and y coordinates and NN charge.
    Non-tracking data is taken from input.
    """
    return pd.DataFrame(dict( event    = event
                            , time     = timestamp
                            , npeak    = peak_no
                            , Xpeak    = x_peak
                            , Ypeak    = y_peak
                            , nsipm    = 1
                            , X        = NN
                            , Y        = NN
                            , Xrms     = 0
                            , Yrms     = 0
                            , Z        = z
                            , Q        = NN
                            , E        = e
                            , Qc       = -1
                            , Ec       = ec
                            , track_id = -1
                            , Ep       = -1), index=[0])


def apply_threshold(hits: pd.DataFrame, th: float, on_corrected: bool = False) -> pd.DataFrame:
    """
    Apply a charge threshold to filter hits and renormalize their energies.

    Input hits with charge (either `Q` or `Qc`) below `th` are removed. The
    energy of the hit collection is preserved and redistributed to the surviving
    hits. If no hits survive the threshold, an empty hit (with NN charge, x and
    y) is returned using the event metadata of the first hit.

    Parameters
    ----------
    hits : pd.DataFrame
        Input hits. All hit columns must be present.

    th : float
        Charge threshold in pe.

    on_corrected : bool, optional
        Whether to use the regular charge `Q` or the corrected charge `Qc`.
        Default is False.

    Returns
    -------
    thresholded_hits: pd.DataFrame
        Hits surviving the threshold, with renormalized `E` and `Ec` values. If
        no hits pass the threshold, an empty hit (with NN charge and x,y
        position) is returned.

    Notes
    -----
    - Energy renormalization ensures that the sum of `E` and `Ec` for the remaining
      hits equals the sum of `E` and `Ec` in the original `hits` DataFrame.
    - If no hits survive the threshold, the returned DataFrame has a single "empty"
      hit corresponding to the first hit's event metadata.
    """
    raw_e_slice = np.   sum(hits.E ) # no   nan expected
    cor_e_slice = np.nansum(hits.Ec) # some nan expected

    col         = "Qc" if on_corrected else "Q"
    mask_thresh = hits[col] >= th

    if not mask_thresh.any():
        first = hits.iloc[0]
        return empty_hit( first.event, first.time
                        , first.npeak, first.Xpeak, first.Ypeak
                        , first.Z, raw_e_slice, cor_e_slice)

    hits = hits.loc[mask_thresh].copy()
    qsum = np.nansum(hits.Q) + EPSILON

    # EPSILON added to avoid division by zero
    # This should be only necessary for Ec, but we apply it to E to be safe
    hits.loc[:, "E" ] = (hits.Q / qsum) * (raw_e_slice + EPSILON)
    hits.loc[:, "Ec"] = (hits.Q / qsum) * (cor_e_slice + EPSILON)
    return hits


def threshold_hits(hits: pd.DataFrame, th: float, on_corrected: bool=False) -> pd.DataFrame:
    """
    Apply a charge threshold (`th`)vto the hits for each Z slice separately. If
    the threshold is negative or zero, the function returns the input DataFrame
    unchanged.

    Parameters
    ----------
    hits : pd.DataFrame
        Input hits. All hit columns must be present.

    th : float
        Charge threshold in pe.

    on_corrected : bool, optional
        Whether to use the regular charge `Q` or the corrected charge `Qc`.
        Default is False.

    Returns
    -------
    thresholded_hits: pd.DataFrame
        Hits surviving the threshold, with renormalized `E` and `Ec` values.
        Slices with no surviving hits produce an empty hit (a.k.a NN hit).

    Notes
    -----
    - Energy renormalization ensures that the sum of `E` and `Ec` for the remaining
      hits equals the sum of `E` and `Ec` in the original `hits` DataFrame.
    - If no hits survive the threshold, the returned DataFrame has a single "empty"
      hit corresponding to the first hit's event metadata.
    - See `apply_threshold` for further details.
    """
    if th <= 0: return hits
    return (hits.groupby("Z", as_index=False)
                .apply(apply_threshold, th=th, on_corrected=on_corrected))

#######################
# Satelites de Luismi #
#######################

@njit(cache=True)
def _drop_hits_numba(Es, indptr, indices, e_thr_eff, n_neigh_thr,
                     redistribute_all, redistribute_weighted :  bool = False):
    n = Es.size
    alive = np.ones(n, dtype=np.bool_)

    modified = True
    while modified:
        modified = False

        # Recorremos "en orden creciente de energía" sin argsort:
        # repetimos n veces buscando el mínimo vivo no procesado
        processed = np.zeros(n, dtype=np.bool_)

        for _ in range(n):
            # buscar idx mínimo vivo y no procesado con Es>0 (o >=0 si quieres)
            idx = -1
            Emin = 1e308
            for i in range(n):
                if alive[i] and (not processed[i]):
                    Ei = Es[i]
                    if Ei < Emin:
                        Emin = Ei
                        idx = i

            if idx == -1:
                break  # no queda nada por procesar

            processed[idx] = True

            # vecinos CSR
            start = indptr[idx]
            end   = indptr[idx + 1]

            # contar vecinos vivos
            n_neigh = 0
            for kk in range(start, end):
                j = indices[kk]
                if alive[j]:
                    n_neigh += 1

            # caso 1: aislado -> elimina y reparte
            if n_neigh == 0:
                Ei = Es[idx]
                alive[idx] = False
                Es[idx] = 0.0

                if redistribute_all:
                    if redistribute_weighted:
                        # suma energía viva actual (sin idx)
                        suma = 0.0
                        for j in range(n):
                            if alive[j]:
                                suma += Es[j]

                        if suma > 0.0 and Ei != 0.0:
                            for j in range(n):
                                if alive[j]:
                                    Ej = Es[j]
                                    Es[j] = Ej + (Ej / suma) * Ei
                    else:
                        # reparto equitativo entre vivos
                        n_alive = 0
                        for j in range(n):
                            if alive[j]:
                                n_alive += 1
                        if n_alive > 0 and Ei != 0.0:
                            share = Ei / n_alive
                            for j in range(n):
                                if alive[j]:
                                    Es[j] += share

                modified = True
                continue

            # caso 2: energía sobre umbral -> no tocar
            if Es[idx] >= e_thr_eff:
                continue

            # caso 3: pocos vecinos -> elimina y reparte a TODOS los vivos
            if n_neigh < n_neigh_thr:
                Ei = Es[idx]
                alive[idx] = False
                Es[idx] = 0.0

                if redistribute_all:
                    if redistribute_weighted:
                        # suma energía viva actual (sin idx)
                        suma = 0.0
                        for j in range(n):
                            if alive[j]:
                                suma += Es[j]

                        if suma > 0.0 and Ei != 0.0:
                            for j in range(n):
                                if alive[j]:
                                    Ej = Es[j]
                                    Es[j] = Ej + (Ej / suma) * Ei
                    else:
                        # reparto equitativo entre vivos
                        n_alive = 0
                        for j in range(n):
                            if alive[j]:
                                n_alive += 1
                        if n_alive > 0 and Ei != 0.0:
                            share = Ei / n_alive
                            for j in range(n):
                                if alive[j]:
                                    Es[j] += share

                modified = True

    return alive, Es


def drop_hits_satellites_xy_z_variable(hitc, e_thr, r_iso, thr_percentile=40, n_neigh_thr=4,
                                       redistribute_all=True,
                                       redistribute_weighted=False):
    hits = list(hitc.hits)
    zscale = 15.55 / 3.97
    if not hits:
        return hitc

    n = len(hits)
    xs = np.fromiter((h.X  for h in hits), dtype=np.float64, count=n)
    ys = np.fromiter((h.Y  for h in hits), dtype=np.float64, count=n)
    zs = np.fromiter((h.Z  for h in hits), dtype=np.float64, count=n)
    Es = np.fromiter((h.Ep for h in hits), dtype=np.float64, count=n)

    Es_pos = Es[np.isfinite(Es) & (Es > 0)]
    e_thr_eff = float(np.percentile(Es_pos, thr_percentile)) if Es_pos.size else float(e_thr)

    # KDTree + vecinos
    P = np.column_stack((xs, ys, zscale * zs))
    tree = cKDTree(P)
    neigh_lists = tree.query_ball_point(P, r_iso)

    # Convertir vecinos a CSR (Compressed Sparse Row):
    # indices[indptr[i]:indptr[i+1]] devuelve los vecinos del hit i
    indptr = np.zeros(n + 1, dtype=np.int64)
    total = 0
    for i in range(n):
        li = neigh_lists[i]
        # quitar self si está (normalmente sí)
        cnt = len(li) - (1 if i in li else 0)
        if cnt < 0:
            cnt = 0
        total += cnt
        indptr[i + 1] = total

    indices = np.empty(total, dtype=np.int32)
    k = 0
    for i in range(n):
        for j in neigh_lists[i]:
            if j == i:
                continue
            indices[k] = j
            k += 1

    # Numba core
    alive, Es_out = _drop_hits_numba(Es.copy(), indptr, indices, e_thr_eff, n_neigh_thr,
                                     redistribute_all, redistribute_weighted)

    # escribir energías de vuelta a objetos
    for h, E_new in zip(hits, Es_out):
        h.Ep = float(E_new)

    filtered_hits = [h for h, keep in zip(hits, alive) if keep]
    return HitCollection(event_number=hitc.event, event_time=-1, hits=filtered_hits)


def drop_satellite_clusters(hitc, r_iso, *,
                            method="top_n",
                            keep_top_n=1,
                            frac_min=0.05,
                            n_hits_min=3,
                            e_min=0.0,
                            zscale=15.55 / 3.97,
                            redistribute_all=True,
                            redistribute_weighted=True):
    """
    Drops satellite clusters based on connected components in XYZ.

    method:
      - "top_n": keep the N most energetic clusters
      - "frac":  keep clusters with E_cluster / E_total >= frac_min
      - "hybrid": keep clusters that satisfy either rule
    """
    hits = list(hitc.hits)
    if not hits:
        return hitc

    n = len(hits)
    xs = np.fromiter((h.X for h in hits), dtype=np.float64, count=n)
    ys = np.fromiter((h.Y for h in hits), dtype=np.float64, count=n)
    zs = np.fromiter((h.Z for h in hits), dtype=np.float64, count=n)
    Es = np.fromiter((h.Ep for h in hits), dtype=np.float64, count=n)

    P = np.column_stack((xs, ys, zscale * zs))
    tree = cKDTree(P)
    pairs = tree.query_pairs(r_iso)

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(pairs)
    components = list(nx.connected_components(g))

    # Compute cluster energies and sizes
    cluster_info = []
    for comp in components:
        idx = np.fromiter(comp, dtype=np.int64)
        e_sum = float(np.nansum(Es[idx]))
        cluster_info.append((comp, e_sum, len(comp)))

    total_e = float(np.nansum(Es))
    if total_e <= 0:
        return HitCollection(event_number=hitc.event, event_time=-1, hits=[])

    # Determine principal clusters
    keep = set()
    if method in ("top_n", "hybrid"):
        top = sorted(cluster_info, key=lambda x: x[1], reverse=True)
        for comp, e_sum, _ in top[:max(0, keep_top_n)]:
            keep |= set(comp)

    if method in ("frac", "hybrid"):
        for comp, e_sum, n_hits in cluster_info:
            if e_sum / total_e >= frac_min and n_hits >= n_hits_min and e_sum >= e_min:
                keep |= set(comp)

    # Apply n_hits_min / e_min filter to all kept clusters
    if keep:
        keep_filtered = set()
        for comp, e_sum, n_hits in cluster_info:
            if comp & keep and n_hits >= n_hits_min and e_sum >= e_min:
                keep_filtered |= set(comp)
        keep = keep_filtered

    if not keep:
        return HitCollection(event_number=hitc.event, event_time=-1, hits=[])

    # Redistribute removed energy to kept hits
    if redistribute_all:
        removed = [i for i in range(n) if i not in keep]
        e_removed = float(np.nansum(Es[removed])) if removed else 0.0
        if e_removed != 0.0:
            kept = np.array(sorted(keep), dtype=np.int64)
            if redistribute_weighted:
                weights = Es[kept].copy()
                wsum = float(np.nansum(weights))
                if wsum > 0:
                    Es[kept] = Es[kept] + (weights / wsum) * e_removed
            else:
                share = e_removed / len(kept)
                Es[kept] = Es[kept] + share

    # Write back Ep and filter
    for i, h in enumerate(hits):
        h.Ep = float(Es[i])
    filtered_hits = [h for i, h in enumerate(hits) if i in keep]
    return HitCollection(event_number=hitc.event, event_time=-1, hits=filtered_hits)
