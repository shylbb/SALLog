"""Bounded-memory offline selection on unique candidate rows.

The SVD neighbor graph estimates marginal gain; it is deliberately not a
coverage certificate. Every added reference is compared with *all* same-label
fine-feature rows by a sparse matrix/vector product. Coverage and redundancy
therefore refer to actual fine-feature cosine comparisons, including across
coarse-cluster boundaries. No global pairwise distance matrix is constructed.
"""
from __future__ import annotations

import heapq
import time
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


DEFAULTS = {
    "min_size": 8000, "max_size": 20000, "seed": 42,
    "cover_similarity": 0.90, "neighbor_count": 64, "local_neighbors": 32,
    "neighbor_query_batch_size": 256, "prune_redundant": True,
    "fine_pair_nnz_budget": 2000000,
    "diagnostic_pair_limit": 1000,
    "progress": True, "progress_every": 1000,
}


def allocate_minimum_quotas(capacities, deficit):
    """Square-root weights, largest remainders, capacity clipping/reallocation.

    Input order is the stable unit tie order. Multiplicities of raw duplicate
    sources never enter this calculation.
    """
    capacities = np.asarray(capacities, dtype=np.int64)
    if capacities.ndim != 1 or np.any(capacities < 0) or deficit < 0:
        raise ValueError("nonnegative one-dimensional capacities/deficit required")
    target = min(int(deficit), int(capacities.sum()))
    allocation = np.zeros(len(capacities), dtype=np.int64)
    while int(allocation.sum()) < target:
        remaining = capacities - allocation
        active = np.flatnonzero(remaining > 0)
        slots = target - int(allocation.sum())
        weights = np.sqrt(remaining[active].astype(float))
        shares = slots * weights / weights.sum()
        base = np.minimum(np.floor(shares).astype(np.int64), remaining[active])
        allocation[active] += base
        slots = target - int(allocation.sum())
        if not slots:
            break
        # One largest-remainder round; clipped units are omitted. Any residual
        # capacity shortage is redistributed by the next water-filling round.
        order = sorted(range(len(active)), key=lambda j: (-(shares[j] % 1), int(active[j])))
        for j in order:
            unit = int(active[j])
            if allocation[unit] < capacities[unit]:
                allocation[unit] += 1
                slots -= 1
                if not slots:
                    break
    return allocation.tolist()


def _config(config):
    supplied = dict(config or {})
    if "neighbors" in supplied and "neighbor_count" not in supplied:
        supplied["neighbor_count"] = supplied["neighbors"]
    cfg = {**DEFAULTS, **supplied}
    for key in ("min_size", "max_size", "seed", "neighbor_count", "local_neighbors",
                "neighbor_query_batch_size", "fine_pair_nnz_budget", "diagnostic_pair_limit", "progress_every"):
        value = cfg[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{key} must be an integer")
        cfg[key] = int(value)
        if value < (1 if key in ("neighbor_count", "local_neighbors", "neighbor_query_batch_size", "fine_pair_nnz_budget", "progress_every") else 0):
            raise ValueError(f"invalid {key}")
    if cfg["max_size"] < cfg["min_size"]:
        raise ValueError("max_size must be >= min_size")
    if not 0 < float(cfg["cover_similarity"]) <= 1:
        raise ValueError("cover_similarity must lie in (0, 1]")
    cfg["cover_similarity"] = float(cfg["cover_similarity"])
    return cfg


def _fine_pair_cosines(X, left, right, row_nnz, nnz_budget):
    """Bound expanded sparse slices even for very long HDFS sessions.

    A fixed row batch alone is insufficient: repeating one long session for
    each recalled neighbor could allocate hundreds of millions of entries.
    One unavoidable individual pair is allowed to exceed the budget.
    """
    costs = row_nnz[left].astype(np.int64) + row_nnz[right]
    cumulative = np.concatenate(([0], np.cumsum(costs)))
    result = np.empty(len(left), dtype=np.float32)
    start = 0
    while start < len(left):
        stop = min(len(left), max(start + 1, int(np.searchsorted(
            cumulative, cumulative[start] + nnz_budget, side="right")) - 1))
        result[start:stop] = np.asarray(
            X[left[start:stop]].multiply(X[right[start:stop]]).sum(axis=1)
        ).ravel()
        start = stop
    return result


def _neighbor_graph(X, reduced, units, cfg):
    """Recall within coarse×label units in SVD, then rerank in fine space.

    KNN in SVD is exact, but it is only an approximate candidate recall for
    fine-space neighbors. The graph is symmetric, includes identity edges, and
    counts unique candidate rows once. Query batches bound pairwise workspaces.
    """
    n = X.shape[0]
    graph_rows, graph_cols = [], []
    representativeness = np.ones(n, dtype=np.float32)
    threshold = cfg["cover_similarity"]
    batch_size = cfg["neighbor_query_batch_size"]
    row_nnz = np.diff(X.indptr)
    for rows in units:
        m = len(rows)
        if m == 1:
            graph_rows.append(rows.copy())
            graph_cols.append(rows.copy())
            continue
        k = min(m, max(cfg["neighbor_count"], cfg["local_neighbors"]) + 1)
        vectors = np.asarray(reduced[rows], dtype=np.float32)
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute", n_jobs=1)
        nn.fit(vectors)
        for start in range(0, m, batch_size):
            batch_rows = rows[start:start + batch_size]
            recalled = rows[nn.kneighbors(vectors[start:start + batch_size], return_distance=False)]
            repeated = np.repeat(batch_rows, k)
            similarities = _fine_pair_cosines(
                X, repeated, recalled.ravel(), row_nnz, cfg["fine_pair_nnz_budget"]
            ).reshape(len(batch_rows), k)
            similarities = np.clip(similarities, 0, 1)
            same = recalled == batch_rows[:, None]
            similarities[same] = 1
            for offset, row in enumerate(batch_rows):
                values = similarities[offset, ~same[offset]]
                take = min(cfg["local_neighbors"], len(values))
                if take:
                    representativeness[row] = float(np.partition(values, -take)[-take:].mean())
                keep = similarities[offset] + 1e-7 >= threshold
                graph_rows.append(np.full(int(keep.sum()), row, dtype=np.int32))
                graph_cols.append(recalled[offset, keep].astype(np.int32))
            # Identity must exist even if tied zero SVD vectors exclude self.
            graph_rows.append(batch_rows.astype(np.int32))
            graph_cols.append(batch_rows.astype(np.int32))
    if not n:
        return sparse.csr_matrix((0, 0), dtype=np.int8), representativeness
    graph = sparse.coo_matrix((np.ones(sum(map(len, graph_rows)), dtype=np.int8),
                               (np.concatenate(graph_rows), np.concatenate(graph_cols))), shape=(n, n)).tocsr()
    graph.data[:] = 1
    graph = graph.maximum(graph.T).tocsr()
    return graph, representativeness


def select_bank(X, reduced, clusters, labels, source_indices, config=None):
    """Select real deduplicated rows; return JSON-compatible records/report.

    ``X`` contains fine features only, never labels. It is row-normalized here
    to make comparisons cosine similarities. ``reduced`` is solely a neighbor
    recall representation. All candidate inputs must already be fit_train-only
    and correctly deduplicated by content/parameters/full sequence/label.

    Source indices are unique stable integer IDs. Labels constrain coverage,
    not distance. If every feature block is empty, an all-zero row receives
    self-only coverage by explicit identity policy, never covers another row.
    """
    started = time.perf_counter()
    cfg = _config(config)
    X = sparse.csr_matrix(X, dtype=np.float32)
    n = X.shape[0]
    labels = np.asarray(labels)
    clusters = np.asarray(clusters)
    source_indices = np.asarray(source_indices)
    reduced = np.asarray(reduced)
    if any(a.ndim != 1 or len(a) != n for a in (labels, clusters, source_indices)):
        raise ValueError("labels/clusters/source_indices must align with X")
    if reduced.ndim != 2 or len(reduced) != n or not np.all(np.isfinite(reduced)):
        raise ValueError("reduced must be an aligned finite dense matrix")
    if len(set(source_indices.tolist())) != n:
        raise ValueError("source_indices must be unique; deduplicate candidates before selection")
    if n and (source_indices.dtype.kind not in "iu" or np.any(source_indices < 0)):
        raise ValueError("source_indices must be nonnegative integers")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("labels must be binary 0/1")
    if n and clusters.dtype.kind not in "iu":
        raise ValueError("clusters must be integer IDs")
    if not np.all(np.isfinite(X.data)) or np.any(X.data < 0):
        raise ValueError("fine features must be finite nonnegative TF-IDF blocks")
    if n:
        X = normalize(X, copy=False).tocsr()
    labels = labels.astype(np.int8)
    units_keys = sorted(set(zip(clusters.tolist(), labels.tolist())))
    units = [np.asarray(sorted(np.flatnonzero((clusters == cluster) & (labels == label)),
                               key=lambda row: int(source_indices[row])), dtype=np.int64)
             for cluster, label in units_keys]
    graph_started = time.perf_counter()
    def progress(message):
        if cfg["progress"]:
            print(f"[selection threshold={cfg['cover_similarity']:.2f}] {message} "
                  f"elapsed_seconds={time.perf_counter() - started:.1f}", flush=True)

    progress(f"phase=neighbor_graph_start candidates={n} fine_nnz={X.nnz}")
    graph, local_representativeness = _neighbor_graph(X, reduced, units, cfg)
    graph_seconds = time.perf_counter() - graph_started
    progress(f"phase=neighbor_graph_complete graph_edges={graph.nnz}")
    # Inverted feature postings avoid scanning every candidate's sparse row
    # on each update. The transpose is built once, never once per reference.
    transpose = X.T.tocsr()
    selected, records, pruned = [], [], []
    selected_mask = np.zeros(n, dtype=bool)
    covered = np.zeros(n, dtype=bool)
    counts = np.zeros(n, dtype=np.int32)
    max_similarity = np.zeros(n, dtype=np.float32)
    # One candidate-length reusable dense row, not a pairwise matrix. CSR's
    # compiled dense conversion is faster here than Python/NumPy scatter-max.
    dense_similarity = np.empty((1, n), dtype=np.float32)
    threshold = cfg["cover_similarity"]
    pair_count, pairs, exact_products = 0, [], 0
    addition_counts = Counter()
    active_stage = None

    def compare(row, *, update_max=False):
        nonlocal exact_products
        exact_products += 1
        # CSR SpGEMM already accumulates each output column once. Keep its
        # sparse row directly: converting to COO needlessly allocates an N-hit
        # row-coordinate array on every reference. Clip the newly owned product
        # data in place, preserving the input feature matrix.
        product = X.getrow(row) @ transpose
        rows = product.indices
        values = product.data
        np.clip(values, 0, 1, out=values)
        keep = labels[rows] == labels[row]
        values[rows == row] = 1
        if update_max:
            # Keep opposite labels out of the dense max update. toarray(out=)
            # clears/reuses this N-float buffer and sums any duplicate sparse
            # coordinates in compiled code; no N×selected array is retained.
            values[~keep] = 0
            product.toarray(out=dense_similarity)
            np.maximum(max_similarity, dense_similarity[0], out=max_similarity)
        rows, values = rows[keep], values[keep]
        if not X.indptr[row + 1] - X.indptr[row]:
            rows = np.append(rows, row)
            values = np.append(values, np.float32(1))
            if update_max:
                max_similarity[row] = 1
        # Explicit self avoids rounding a normalized row infinitesimally below 1.
        self_position = rows == row
        values[self_position] = 1
        return rows, values

    def add(row, stage, score, estimated_gain):
        nonlocal pair_count, active_stage
        if stage != active_stage:
            if active_stage is not None:
                progress(f"phase={active_stage}_complete additions={addition_counts[active_stage]} "
                         f"selected={len(selected)} covered={int(covered.sum())}/{n}")
            active_stage = stage
            progress(f"phase={stage}_start selected={len(selected)}")
        old_max = float(max_similarity[row])
        rows, values = compare(row, update_max=True)
        hits = rows[values + 1e-7 >= threshold]
        gain = int(np.count_nonzero(~covered[hits]))
        cross = hits[selected_mask[hits] & (clusters[hits] != clusters[row])]
        pair_count += len(cross)
        for other in cross[:max(0, cfg["diagnostic_pair_limit"] - len(pairs))]:
            pairs.append([int(source_indices[row]), int(source_indices[other])])
        counts[hits] += 1
        covered[hits] = True
        selected.append(int(row))
        selected_mask[row] = True
        records.append({
            "row_index": int(row), "source_index": int(source_indices[row]),
            "label": int(labels[row]), "cluster": int(clusters[row]), "stage": stage,
            "score": float(score), "coverage_gain": gain,
            "estimated_coverage_gain": int(estimated_gain),
            "max_same_label_similarity_before": old_max,
            "local_representativeness": float(local_representativeness[row]),
        })
        addition_counts[stage] += 1
        if addition_counts[stage] % cfg["progress_every"] == 0:
            progress(f"phase={stage} additions={addition_counts[stage]} selected={len(selected)} "
                     f"covered={int(covered.sum())}/{n} last_exact_gain={gain}")

    # One real representative per observed coarse×label unit, as capacity allows.
    for rows in units:
        if len(selected) >= cfg["max_size"]:
            break
        degrees = np.diff(graph.indptr)[rows]
        row = int(rows[np.argmax(degrees)])  # rows already in stable source order
        add(row, "initial_representative", int(degrees.max()), int(degrees.max()))

    # Lazy submodular max-gain heap. The graph counts unique rows and supplies
    # only an estimate; exact coverage updates suppress already covered modes.
    heap = [(-int(graph.indptr[row + 1] - graph.indptr[row]), int(source_indices[row]), row)
            for row in range(n) if not selected_mask[row]]
    heapq.heapify(heap)
    while heap and not bool(covered.all()) and len(selected) < cfg["max_size"]:
        old_negative, source, row = heapq.heappop(heap)
        if selected_mask[row]:
            continue
        neighbors = graph.indices[graph.indptr[row]:graph.indptr[row + 1]]
        gain = int(np.count_nonzero(~covered[neighbors]))
        if gain != -old_negative:
            heapq.heappush(heap, (-gain, source, row))
            continue
        if not gain:
            break
        add(row, "coverage_expansion", gain, gain)

    # Self edges mean every nonempty uncovered row has positive graph gain.
    # This repair is defensive and ensures approximate recall never proves
    # convergence. Exact uncovered rows remain eligible until max_size.
    for row in sorted(np.flatnonzero(~covered), key=lambda r: int(source_indices[r])):
        if len(selected) >= cfg["max_size"]:
            break
        if not covered[row] and not selected_mask[row]:
            add(int(row), "coverage_expansion", 1, 1)

    coverage_before_pruning = int(covered.sum())
    if cfg["prune_redundant"] and bool(covered.all()) and len(selected) > 1:
        # Later representatives may make earlier ones redundant. Removal is
        # permitted only if every currently covered candidate keeps a witness.
        # Reverse order preserves stable early representatives when possible.
        progress(f"phase=coverage_safe_pruning_start references={len(selected)}")
        for checked, row in enumerate(selected[::-1], 1):
            rows, values = compare(row)
            hits = rows[values + 1e-7 >= threshold]
            if np.all(counts[hits] > 1):
                counts[hits] -= 1
                selected_mask[row] = False
                pruned.append(int(source_indices[row]))
            if checked % cfg["progress_every"] == 0:
                progress(f"phase=coverage_safe_pruning checked={checked}/{len(selected)} removed={len(pruned)}")
        if pruned:
            selected = [row for row in selected if selected_mask[row]]
            records = [record for record in records if selected_mask[record["row_index"]]]
            max_similarity[:] = 0
            for row in selected:
                compare(row, update_max=True)
        if int(np.count_nonzero(counts)) != coverage_before_pruning:
            raise AssertionError("coverage-safe pruning lost coverage")
        progress(f"phase=coverage_safe_pruning_complete removed={len(pruned)} selected={len(selected)}")

    remaining = [int(np.count_nonzero(~selected_mask[rows])) for rows in units]
    deficit = max(0, min(cfg["min_size"], n) - len(selected))
    quotas = allocate_minimum_quotas(remaining, deficit)
    minimum_trace = []
    for key, rows, quota in zip(units_keys, units, quotas):
        unit_heap = [(-float(local_representativeness[row] * (1 - max_similarity[row])),
                      int(source_indices[row]), int(row)) for row in rows if not selected_mask[row]]
        heapq.heapify(unit_heap)
        for _ in range(quota):
            while unit_heap:
                old_negative, source, row = heapq.heappop(unit_heap)
                score = float(local_representativeness[row] * (1 - max_similarity[row]))
                if abs(score + old_negative) > 1e-12:
                    heapq.heappush(unit_heap, (-score, source, row))
                    continue
                before = float(max_similarity[row])
                add(row, "minimum_fill", score, 0)
                minimum_trace.append({"source_index": source, "max_similarity_before": before,
                                      "score": score})
                break

    if active_stage is not None:
        progress(f"phase={active_stage}_complete additions={addition_counts[active_stage]} "
                 f"selected={len(selected)} covered={int(covered.sum())}/{n}")

    uncovered = np.flatnonzero(~covered)
    actual = len(selected)
    if len(uncovered) and actual >= cfg["max_size"]:
        stop = "capacity_limit_coverage_incomplete"
    elif len(uncovered):
        stop = "available_candidates_exhausted_coverage_incomplete"
    elif actual < cfg["min_size"]:
        stop = "available_unique_candidates_below_minimum"
    else:
        stop = "coverage_complete_and_minimum_reached"
    per_unit = {}
    for key, rows, quota in zip(units_keys, units, quotas):
        per_unit[f"{int(key[0])}:{int(key[1])}"] = {
            "cluster": int(key[0]), "label": int(key[1]), "candidates": len(rows),
            "selected": int(selected_mask[rows].sum()), "covered": int(covered[rows].sum()),
            "uncovered": int((~covered[rows]).sum()), "minimum_fill_quota": quota,
            "coverage_fraction": float(covered[rows].mean()),
        }
    per_label = {}
    for label in (0, 1):
        rows = np.flatnonzero(labels == label)
        per_label[str(label)] = {"candidates": len(rows), "selected": int(selected_mask[rows].sum()),
                                 "covered": int(covered[rows].sum()),
                                 "uncovered": int((~covered[rows]).sum()),
                                 "coverage_fraction": float(covered[rows].mean()) if len(rows) else None}
    fills = [r for r in records if r["stage"] == "minimum_fill"]
    report = {
        "config": cfg, "actual_size": actual, "min_size": cfg["min_size"],
        "max_size": cfg["max_size"], "shortfall": max(0, cfg["min_size"] - actual),
        "shortfall_reason": "available_unique_candidates_below_minimum" if n < cfg["min_size"] else None,
        "stop_reason": stop, "coverage_complete": not len(uncovered),
        "cover_similarity_status": "provisional", "covered": int(covered.sum()),
        "uncovered": len(uncovered), "coverage_fraction": float(covered.mean()) if n else None,
        "coverage_method": "exact_incremental_sparse_same_label_cosine_all_candidates",
        "coverage_float_tolerance": 1e-7,
        "gain_estimation": {"algorithm": "symmetric_coarse_label_SVD_kNN_fine_reranked_graph",
                            "approximate": True, "neighbor_count": cfg["neighbor_count"],
                            "query_batch_size": cfg["neighbor_query_batch_size"],
                            "fine_pair_nnz_budget": cfg["fine_pair_nnz_budget"],
                            "global_N_by_N_matrix": False, "graph_edges": int(graph.nnz)},
        "local_representativeness_method": "mean_top_fine_cosines_among_SVD_recalled_neighbors",
        "local_neighbors": cfg["local_neighbors"], "singleton_representativeness": 1.0,
        "zero_feature_rows": int(np.count_nonzero(np.diff(X.indptr) == 0)),
        "zero_feature_policy": "self_only_identity_coverage",
        "stage_counts": dict(Counter(record["stage"] for record in records)),
        "addition_attempts_by_stage": dict(addition_counts),
        "per_label": per_label, "per_unit": per_unit,
        "cross_cluster_near_duplicate_pairs_encountered": int(pair_count),
        "cross_cluster_near_duplicate_source_pairs_sample": pairs,
        "pruned_source_indices": pruned, "coverage_safe_pruned_count": len(pruned),
        "minimum_fill_high_similarity_count": sum(r["max_same_label_similarity_before"] + 1e-7 >= threshold for r in fills),
        "minimum_fill_high_similarity_fraction": (sum(r["max_same_label_similarity_before"] + 1e-7 >= threshold for r in fills) / len(fills)) if fills else None,
        "minimum_fill_redundancy_update": "exact_global_same_label_after_each_addition",
        "exact_sparse_matrix_vector_products": exact_products,
        "max_similarity_update": "exact_CSR_to_reused_candidate_length_dense_buffer_then_contiguous_maximum",
        "graph_build_seconds": graph_seconds, "selection_seconds": time.perf_counter() - started,
        "resource_bytes": {"fine_csr": int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes),
                           "fine_inverted_postings_csr": int(transpose.data.nbytes + transpose.indices.nbytes + transpose.indptr.nbytes),
                           "dense_max_update_buffer": int(dense_similarity.nbytes),
                           "neighbor_graph_csr": int(graph.data.nbytes + graph.indices.nbytes + graph.indptr.nbytes)},
    }
    return {"selected_indices": selected,
            "selected_source_indices": [int(source_indices[row]) for row in selected],
            "selected_labels": [int(labels[row]) for row in selected],
            "records": records, "report": report,
            "max_similarity": max_similarity.tolist(),
            "uncovered_indices": uncovered.astype(int).tolist(),
            "minimum_fill_trace": minimum_trace}
