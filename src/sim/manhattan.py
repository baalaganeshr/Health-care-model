"""Manhattan distance and clustering utilities.

Provides Manhattan (L1) distance calculation and cluster assignment
functionality for wireless sensor network simulations.
"""

import numpy as np
from typing import List, Tuple


def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Manhattan (L1) distance between two points.
    
    Args:
        a: First point as numpy array [x, y]
        b: Second point as numpy array [x, y]
        
    Returns:
        Manhattan distance between points a and b
    """
    return np.sum(np.abs(a - b))


def assign_to_ch(positions: np.ndarray, ch_indices: List[int]) -> np.ndarray:
    """Assign each node to the nearest cluster head using Manhattan distance.
    
    Args:
        positions: Node positions as (N, 2) array
        ch_indices: List of cluster head indices
        
    Returns:
        Array of cluster assignments for each node (cluster head index)
    """
    if not ch_indices:
        raise ValueError("No cluster heads provided")
    
    n_nodes = positions.shape[0]
    assignments = np.zeros(n_nodes, dtype=int)
    
    for i in range(n_nodes):
        if i in ch_indices:
            # Cluster head assigns to itself
            assignments[i] = i
        else:
            # Find nearest cluster head
            min_dist = float('inf')
            nearest_ch = ch_indices[0]
            
            for ch_idx in ch_indices:
                dist = manhattan(positions[i], positions[ch_idx])
                if dist < min_dist:
                    min_dist = dist
                    nearest_ch = ch_idx
            
            assignments[i] = nearest_ch
    
    return assignments


def calculate_intra_cluster_distance(positions: np.ndarray, 
                                   assignments: np.ndarray) -> float:
    """Calculate total intra-cluster Manhattan distance.
    
    Args:
        positions: Node positions as (N, 2) array
        assignments: Cluster assignments for each node
        
    Returns:
        Total intra-cluster distance
    """
    total_distance = 0.0
    unique_clusters = np.unique(assignments)
    
    for cluster_id in unique_clusters:
        cluster_members = np.where(assignments == cluster_id)[0]
        ch_pos = positions[cluster_id]  # Cluster head position
        
        for member_idx in cluster_members:
            if member_idx != cluster_id:  # Don't count CH to itself
                total_distance += manhattan(positions[member_idx], ch_pos)
    
    return total_distance
