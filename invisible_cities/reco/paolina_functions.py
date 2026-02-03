from functools   import reduce
from itertools   import combinations

import copy

import numpy    as np
import networkx as nx

from networkx           import Graph
from .. evm.event_model import Voxel
from .. core.exceptions import NoHits
from .. core.exceptions import NoVoxels
from .. evm.event_model import BHit
from .. evm.event_model import Track
from .. evm.event_model import Blob
from .. evm.event_model import TrackCollection
from .. core            import system_of_units as units
from .. types.symbols   import Contiguity
from .. types.symbols   import HitEnergy

from typing import Sequence
from typing import List
from typing import Tuple
from typing import Dict

MAX3D = np.array([float(' inf')] * 3)
MIN3D = np.array([float('-inf')] * 3)
 #
 # #######################################################################################################
 #  
def geodesic_ball(center : Voxel, r : float, dist : Dict) -> Tuple[List, List]:
    row = dist.get(center, {})
    voxels = [u for u, d in row.items() if d <= r]
    dists = [d for u, d in row.items() if d <= r]
    return (voxels, dists)

def hits_around_blob(track_graph,
                     radius : float,
                     extreme: Voxel,
                     dist: Dict,
                     zscale : float = 15.55 / 3.97) -> List:
    blob_pos = np.asarray(extreme.pos, float)
    r2 = float(radius) * float(radius)
    dist_from_extreme = dist.get(extreme, {})
    blob_hits = []
    for v in track_graph.nodes():
        d_graph = dist_from_extreme.get(v, np.inf)
        if d_graph > radius:
            continue
        for h in v.hits:
            p = np.asarray(h.pos, float)
            dx, dy, dz = p - blob_pos
            d2 = dx*dx + dy*dy + (zscale*dz)*(zscale*dz)
            if d2 <= r2:
                blob_hits.append(h)
    
    return blob_hits

def closest_voxel(nodes: List, point : np.ndarray):
    if not nodes:
        return None
    p = np.asarray(point, float)
    
    return min(nodes, key=lambda v: np.linalg.norm(np.asarray(v.pos, float) - p))

def barycenter(ball : List,
               fallback_pos,
               distances_to_center : List,
               e_attr : str = "Ec",
               d0 : float = 45 *units.mm,
               p : int = 3) -> np.ndarray:
    
    if len(ball) != len(distances_to_center):
        raise ValueError("ball and distances_to_center should have the same length")
    
    dist_map = {v: float(d) for v, d in zip(ball, distances_to_center)}
    
    P, W = [], []
    for v in ball:
        dv = dist_map.get(v, 0.0) # TODO Best to define the distance of the hits instead of the voxel they are from
        denom = 1.0 + (dv / d0) ** p

        for h in v.hits:
            e = getattr(h, e_attr, np.nan)
            if np.isfinite(e) and e > 0:
                P.append(np.asarray(h.pos, float))
                W.append(e / denom)
    
    if not W or np.sum(W) <= 0:
        return np.asarray(fallback_pos, float)
    
    return np.average(np.vstack(P), weights=np.asarray(W), axis=0)

def hits_energy(hits : List, e_attr : str = "Ec") -> float:
    vals = [getattr(h, e_attr, np.nan) for h in hits]
    vals = [x for x in vals if np.isfinite(x) and x >= 0]
    return float(np.sum(vals)) if vals else 0.0

def find_highest_encapsulating_node(extreme : Voxel,
                                    track_graph,
                                    dist : Dict,
                                    e_attr : str = "Ec",
                                    small_radius : float = 15.55 * units.mm,
                                    big_radius : float = 90 * units.mm,
                                    zscale : float = 15.55 / 3.97) -> Voxel:
    
    nodes_within_radius = [node for node in track_graph.nodes if dist[extreme].get(node, np.inf) <= big_radius]
    if not nodes_within_radius:
        return extreme

    def energy_within_radius(node :  Voxel) -> float:
        hits = hits_around_blob(track_graph,
                                small_radius,
                                node,
                                dist,
                                zscale)
        
        return hits_energy(hits, e_attr)
    
    return max(nodes_within_radius, key=energy_within_radius)

def blob_center_zaira(track_graph: Graph,
                      endpoint_voxel: Voxel,
                      R: float,
                      dist: Dict = None,
                      e_attr: str = "Ec",
                      max_iter: int = 20,
                      tol: float = 0,
                      zscale: float = 15.55 / 3.97) -> Tuple[np.ndarray, float, List, List, object]:

    if track_graph.number_of_nodes() == 0 or endpoint_voxel not in track_graph:
        return np.zeros(3), 0.0, [], [], None

    if dist is None:
        dist = shortest_paths(track_graph)

    center = find_highest_encapsulating_node(extreme=endpoint_voxel,
                                             track_graph=track_graph,
                                             dist=dist,
                                             big_radius=2 * R,
                                             small_radius=15.55 * units.mm,
                                             e_attr=e_attr,
                                             zscale=1) # OJO AQUI NOTE

    prev = np.asarray(center.pos, float)

    for _ in range(max_iter):
        ball, distances_to_center = geodesic_ball(center, R, dist)
        if not ball:
            break

        bary = barycenter(ball,
                          fallback_pos=center.pos,
                          distances_to_center=distances_to_center,
                          e_attr=e_attr)

        new_center = closest_voxel(ball, bary)
        if new_center is None:
            break

        cur = np.asarray(new_center.pos, float)
        if np.linalg.norm(cur - prev) <= tol:
            center = new_center
            break

        center = new_center
        prev = cur

    ball, distances_to_center = geodesic_ball(center, R, dist)
    hits_blob = hits_around_blob(track_graph,
                                 radius=45*units.mm, # NOTE Quiza sea bueno tocar aqui para agarrar mas energia? NOTE NOTE TODO TODO
                                 extreme=center,
                                 dist=dist,
                                 zscale=zscale)

    E_blob = hits_energy(hits_blob, e_attr=e_attr)
    bary_final = barycenter(ball,
                            fallback_pos=center.pos,
                            distances_to_center=distances_to_center,
                            e_attr=e_attr)

    return bary_final, E_blob, ball, hits_blob, center
 #
 # #######################################################################################################
 #  

def bounding_box(seq : BHit) -> Sequence[np.ndarray]:
    """Returns two arrays defining the coordinates of a box that bounds the voxels"""
    posns = [x.pos for x in seq]
    return (reduce(np.minimum, posns, MAX3D),
            reduce(np.maximum, posns, MIN3D))

def round_hits_positions_in_place(hits, decimals=5):
    """
    Rounds the hits positions to `decimals` decimals to avoid floating point
    comparison issues. The operation is performed inplace to avoid an
    unnecessary copy.
    """
    for hit in hits:
        hit.xyz = np.round(hit.xyz, decimals)

def voxelize_hits(hits             : Sequence[BHit],
                  voxel_dimensions : np.ndarray,
                  strict_voxel_size: bool = False,
                  energy_type      : HitEnergy = HitEnergy.E) -> List[Voxel]:
    # 1. Find bounding box of all hits.
    # 2. Allocate hits to regular sub-boxes within bounding box, using histogramdd.
    # 3. Calculate voxel energies by summing energies of hits within each sub-box.
    if not hits:
        raise NoHits
    hlo, hhi = bounding_box(hits)
    bounding_box_centre = (hhi + hlo) / 2
    bounding_box_size   =  hhi - hlo
    number_of_voxels = np.ceil(bounding_box_size / voxel_dimensions).astype(int)
    number_of_voxels = np.clip(number_of_voxels, a_min=1, a_max=None)
    if strict_voxel_size: half_range = number_of_voxels * voxel_dimensions / 2
    else                : half_range =          bounding_box_size          / 2
    voxel_edges_lo = bounding_box_centre - half_range
    voxel_edges_hi = bounding_box_centre + half_range

    # Expand the voxels a tiny bit, in order to include hits which
    # fall within the margin of error of the voxel bounding box.
    eps = 3e-12 # geometric mean of range that seems to work
    voxel_edges_lo -= eps
    voxel_edges_hi += eps

    hit_positions = np.array([h.pos                   for h in hits]).astype('float64')
    hit_energies  =          [getattr(h, energy_type.value) for h in hits]
    E, edges = np.histogramdd(hit_positions,
                              bins    = number_of_voxels,
                              range   = tuple(zip(voxel_edges_lo, voxel_edges_hi)),
                              weights = hit_energies)

    def centres(a : np.ndarray) -> np.ndarray:
        return (a[1:] + a[:-1]) / 2
    def   sizes(a : np.ndarray) -> np.ndarray:
        return  a[1:] - a[:-1]

    (   cx,     cy,     cz) = map(centres, edges)
    size_x, size_y, size_z  = map(sizes  , edges)

    nz = np.nonzero(E)
    true_dimensions = np.array([size_x[0], size_y[0], size_z[0]])

    hit_x = np.array([h.X for h in hits])
    hit_y = np.array([h.Y for h in hits])
    hit_z = np.array([h.Z for h in hits])
    hit_coordinates = [hit_x, hit_y, hit_z]

    indx_coordinates = []
    for i in range(3):
        # find the bins where hits fall into
        # numpy.histogramdd() uses [,) intervals...
        index = np.digitize(hit_coordinates[i], edges[i], right=False) - 1
        # ...except for the last one, which is [,]: hits on the last edge,
        # if any, must fall into the last bin
        index[index == number_of_voxels[i]] = number_of_voxels[i] - 1
        indx_coordinates.append(index)

    h_indices = np.array([(i, j, k) for i, j, k in zip(indx_coordinates[0], indx_coordinates[1], indx_coordinates[2])])

    voxels = []
    for (x,y,z) in np.stack(nz).T:

        indx_comp = (h_indices == (x, y, z))
        hits_in_bin = list(h for i, h in zip(indx_comp, hits) if all(i))

        voxels.append(Voxel(cx[x], cy[y], cz[z], E[x,y,z], true_dimensions, hits_in_bin, energy_type))

    return voxels


def neighbours(va : Voxel, vb : Voxel, contiguity : Contiguity = Contiguity.CORNER) -> bool:
    return np.linalg.norm((va.pos - vb.pos) / va.size) < contiguity.value


def make_track_graphs(voxels           : Sequence[Voxel],
                      contiguity       : Contiguity = Contiguity.CORNER) -> Sequence[Graph]:
    """Create a graph where the voxels are the nodes and the edges are any
    pair of neighbour voxel. Two voxels are considered to be
    neighbours if their distance normalized to their size is smaller
    than a contiguity factor.
    """

    voxel_graph = nx.Graph()
    voxel_graph.add_nodes_from(voxels)
    for va, vb in combinations(voxels, 2):
        if neighbours(va, vb, contiguity):
            voxel_graph.add_edge(va, vb, distance = np.linalg.norm(va.pos - vb.pos))

    return tuple(connected_component_subgraphs(voxel_graph))


def connected_component_subgraphs(G):
    return (G.subgraph(c).copy() for c in nx.connected_components(G))


def voxels_from_track_graph(track: Graph) -> List[Voxel]:
    """Create and return a list of voxels from a track graph."""
    return track.nodes()


def shortest_paths(track_graph : Graph) -> Dict[Voxel, Dict[Voxel, float]]:
    """Compute shortest path lengths between all nodes in a weighted graph."""
    def voxel_pos(x):
        return x[0].pos.tolist()

    distances = dict(nx.all_pairs_dijkstra_path_length(track_graph, weight='distance'))

    # sort the output so the result is reproducible
    distances = { v1 : {v2:d for v2, d in sorted(dmap.items(), key=voxel_pos)}
                  for v1, dmap in sorted(distances.items(), key=voxel_pos)}
    return distances



def find_extrema_and_length(distance : Dict[Voxel, Dict[Voxel, float]]) -> Tuple[Voxel, Voxel, float]:
    """Find the extrema and the length of a track, given its dictionary of distances."""
    if not distance:
        raise NoVoxels
    if len(distance) == 1:
        only_voxel = next(iter(distance))
        return (only_voxel, only_voxel, 0.)
    first, last, max_distance = None, None, 0
    for (voxel1, dist_from_voxel_1_to), (voxel2, _) in combinations(distance.items(), 2):
        d = dist_from_voxel_1_to[voxel2]
        if d > max_distance:
            first, last, max_distance = voxel1, voxel2, d
    return first, last, max_distance


def find_extrema(track: Graph) -> Tuple[Voxel, Voxel]:
    """Find the pair of voxels separated by the greatest geometric
      distance along the track.
    """
    distances = shortest_paths(track)
    extremum_a, extremum_b, _ = find_extrema_and_length(distances)
    return extremum_a, extremum_b


def length(track: Graph) -> float:
    """Calculate the length of a track."""
    distances = shortest_paths(track)
    _, _, length = find_extrema_and_length(distances)
    return length


def energy_of_voxels_within_radius(distances : Dict[Voxel, float], radius : float) -> float:
    return sum(v.E for (v, d) in distances.items() if d < radius)


def voxels_within_radius(distances : Dict[Voxel, float],
                         radius : float) -> List[Voxel]:
    return [v for (v, d) in distances.items() if d < radius]


def blob_centre(voxel: Voxel) -> Tuple[float, float, float]:
    """Calculate the blob position, starting from the end-point voxel."""
    positions = [h.pos              for h in voxel.hits]
    energies  = [getattr(h, voxel.Etype) for h in voxel.hits]
    if sum(energies):
        bary_pos = np.average(positions, weights=energies, axis=0)
    # Consider the case where voxels are built without associated hits
    else:
        bary_pos = voxel.pos

    return bary_pos


def hits_in_blob(track_graph : Graph,
                 radius      : float,
                 extreme     : Voxel,
                 distances : Dict[Voxel, float] =  None) -> Sequence[BHit]:
    """Returns the hits that belong to a blob."""
    if distances is None:
        distances         = shortest_paths(track_graph)
    dist_from_extreme = distances[extreme]
    blob_pos          = blob_centre(extreme)
    diag              = np.linalg.norm(extreme.size)

    blob_hits = []
    # First, consider only voxels at a certain distance from the end-point, along the track.
    # We allow for 1 extra contiguity, because this distance is calculated between
    # the centres of the voxels, and not the hits. In the second step we will refine the
    # selection, using the euclidean distance between the blob position and the hits.
    for v in track_graph.nodes():
        voxel_distance = dist_from_extreme[v]
        if voxel_distance < radius + diag:
            for h in v.hits:
                hit_distance = np.linalg.norm(blob_pos - h.pos)
                if hit_distance < radius:
                    blob_hits.append(h)

    return blob_hits


def blob_energies_hits_and_centres(track_graph : Graph, radius : float, distances : Dict = None, zscale : float = 15.55 / 3.97) -> Tuple[float, float, Sequence[BHit], Sequence[BHit], Tuple[float, float, float], Tuple[float, float, float]]:
    """Return the energies, the hits and the positions of the blobs.
       For each pair of observables, the one of the blob of largest energy is returned first."""
    
    # Only calculate shortest_paths() if needed
    if distances is None:
        distances = shortest_paths(track_graph)

    a, b, _   = find_extrema_and_length(distances)

    voxels = list(track_graph.nodes())
    e_type = voxels[0].Etype

    ca, Ea, _, ha, center1 = blob_center_zaira(track_graph, a, radius, max_iter=1,e_attr=e_type, dist=distances, zscale=zscale)
    cb, Eb, _, hb, center2 = blob_center_zaira(track_graph, b, radius, max_iter=1,e_attr=e_type, dist=distances, zscale=zscale)

    if center1 == center2:
        center1 = find_highest_encapsulating_node(extreme=a, track_graph=track_graph, dist=distances, big_radius=radius, small_radius=15.55*units.mm, zscale=1)
        center2 = find_highest_encapsulating_node(extreme=b, track_graph=track_graph, dist=distances, big_radius=radius, small_radius=15.55*units.mm, zscale=1)
        ca = blob_centre(center1)
        cb = blob_centre(center2)
        ha = hits_around_blob(track_graph, radius, center1, distances, zscale)
        hb = hits_around_blob(track_graph, radius, center2, distances, zscale)
        Ea = sum(getattr(h, e_type) for h in ha)
        Eb = sum(getattr(h, e_type) for h in hb)

        if center1 == center2:
            ca = blob_centre(a)
            cb = blob_centre(b)
            ha = hits_around_blob(track_graph, radius, a, distances, zscale)
            hb = hits_around_blob(track_graph, radius, b, distances, zscale)
            Ea = sum(getattr(h, e_type) for h in ha)
            Eb = sum(getattr(h, e_type) for h in hb)
    # Consider the case where voxels are built without associated hits
    if len(ha) == 0 and len(hb) == 0 :
        Ea = energy_of_voxels_within_radius(distances[a], radius)
        Eb = energy_of_voxels_within_radius(distances[b], radius)

    if Eb > Ea:
        return (Eb, Ea, hb, ha, cb, ca)
    else:
        return (Ea, Eb, ha, hb, ca, cb)


def blob_energies(track_graph : Graph, radius : float) -> Tuple[float, float]:
    """Return the energies around the extrema of the track.
       The largest energy is returned first."""
    E1, E2, _, _, _, _ = blob_energies_hits_and_centres(track_graph, radius)

    return E1, E2


def blob_energies_and_hits(track_graph : Graph, radius : float) -> Tuple[float, float, Sequence[BHit], Sequence[BHit]]:
    """Return the energies and the hits around the extrema of the track.
       The largest energy is returned first, as well as its hits."""
    E1, E2, h1, h2, _, _ = blob_energies_hits_and_centres(track_graph, radius)

    return (E1, E2, h1, h2)


def blob_centres(track_graph : Graph, radius : float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return the positions of the blobs.
       The blob of largest energy is returned first."""
    _, _, _, _, c1, c2 = blob_energies_hits_and_centres(track_graph, radius)

    return (c1, c2)


def make_tracks(evt_number       : float,
                evt_time         : float,
                voxels           : List[Voxel],
                voxel_dimensions : np.ndarray,
                contiguity       : Contiguity = Contiguity.CORNER,
                blob_radius      : float = 30 * units.mm,
                energy_type      : HitEnergy = HitEnergy.E) -> TrackCollection:
    """Make a track collection."""
    tc = TrackCollection(evt_number, evt_time) # type: TrackCollection
    track_graphs = make_track_graphs(voxels, contiguity) # type: Sequence[Graph]
    for trk in track_graphs:
        energy_a, energy_b, hits_a, hits_b = blob_energies_and_hits(trk, blob_radius)
        a, b                               = blob_centres(trk, blob_radius)
        blob_a = Blob(a, hits_a, blob_radius, energy_type) # type: Blob
        blob_b = Blob(b, hits_b, blob_radius, energy_type)
        blobs = (blob_a, blob_b)
        track = Track(voxels_from_track_graph(trk), blobs)
        tc.tracks.append(track)
    return tc


def drop_end_point_voxels(voxels           : Sequence[Voxel],
                          energy_threshold : float,
                          min_vxls         : int = 3,
                          contiguity       : Contiguity = Contiguity.CORNER) -> Sequence[Voxel]:
    """Eliminate voxels at the end-points of a track, recursively,
       if their energy is lower than a threshold. Returns 1 if the voxel
       has been deleted succesfully and 0 otherwise."""

    e_type = voxels[0].Etype

    def drop_voxel(voxels: Sequence[Voxel], the_vox: Voxel, contiguity: Contiguity = Contiguity.CORNER) -> int:
        """Eliminate an individual voxel from a set of voxels and give its energy to the hit
           that is closest to the barycenter of the eliminated voxel hits, provided that it
           belongs to a neighbour voxel."""
        the_neighbour_voxels = [v for v in voxels if neighbours(the_vox, v, contiguity)]

        pos = [h.pos              for h in the_vox.hits]
        qs  = [getattr(h, e_type) for h in the_vox.hits]

        #if there are no hits associated to voxels the pos will be an empty list
        if len(pos) == 0:
            min_dist  = min(np.linalg.norm(the_vox.pos-v.pos) for v in the_neighbour_voxels)
            min_v     = [v for v in the_neighbour_voxels if  np.isclose(np.linalg.norm(the_vox.pos-v.pos), min_dist)]

            ### add dropped voxel energy to closest voxels, proportional to the  voxels energy
            sum_en_v = sum(v.E for v in min_v)
            for v in min_v:
                v.E += the_vox.E/sum_en_v * v.E
            return

        bary_pos = np.average(pos, weights=qs, axis=0)

        ### find hit with minimum distance, only among neighbours
        min_dist = min(np.linalg.norm(bary_pos-hh.pos) for v in the_neighbour_voxels for hh in v.hits)
        min_h_v  = [(h, v) for v in the_neighbour_voxels for h in v.hits if np.isclose(np.linalg.norm(bary_pos-h.pos), min_dist)]
        min_hs   = set(h for (h,v) in min_h_v)
        min_vs   = set(v for (h,v) in min_h_v)

        ### add dropped voxel energy to closest hits/voxels, proportional to the hits/voxels energy
        sum_en_h = sum(getattr(h, e_type) for h in min_hs)
        sum_en_v = sum(v.E                for v in min_vs)
        for h in min_hs:
            setattr(h, e_type, getattr(h, e_type) + getattr(h, e_type) * the_vox.E/sum_en_h)
        for v in min_vs:
            v.E = sum(getattr(h, e_type) for h in v.hits)

    def nan_energy(voxel):
        voxel.E = np.nan
        for hit in voxel.hits:
            setattr(hit, e_type, np.nan)

    mod_voxels     = copy.deepcopy(voxels)
    dropped_voxels = []

    modified = True
    while modified:
        modified = False
        trks = make_track_graphs(mod_voxels, contiguity)

        for t in trks:
            if len(t.nodes()) < min_vxls:
                continue

            for extreme in find_extrema(t):
                if extreme.E < energy_threshold:
                    ### be sure that the voxel to be eliminated has at least one neighbour
                    ### beyond itself
                    n_neighbours = sum(neighbours(extreme, v, contiguity) for v in mod_voxels)
                    if n_neighbours > 1:
                        mod_voxels    .remove(extreme)
                        dropped_voxels.append(extreme)
                        drop_voxel(mod_voxels, extreme)
                        nan_energy(extreme)
                        modified = True

    return mod_voxels, dropped_voxels


def get_track_energy(track):
    return sum([vox.Ehits for vox in track.nodes()])
