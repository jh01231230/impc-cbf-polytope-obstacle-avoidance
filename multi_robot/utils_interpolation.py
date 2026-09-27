"""Interpolates a path of 2D points to a finer resolution."""
import numpy as np

def interpolate_path(path: np.ndarray, resolution: float) -> np.ndarray:
    """
    Interpolates a path of 2D points to a finer resolution.
    
    Args:
        path: (N, 2) array of points.
        resolution: Desired distance between points.
        
    Returns:
        (M, 2) array of interpolated points.
    """
    if len(path) < 2:
        return path
        
    new_path = [path[0]]
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-6:
            continue
            
        num_points = int(np.ceil(dist / resolution))
        # Generate points between p1 and p2 (excluding p1, including p2 if last segment?)
        # Linspace includes start and end. We skip start to avoid duplicate.
        # But we want uniform spacing.
        # simple way: vector p1->p2. 
        vec = (p2 - p1) / dist
        for j in range(1, num_points + 1):
            # Distance from p1
            d = j * resolution
            if d > dist: 
                # If we overshoot, just add p2? 
                # Or keep it strictly resolution spaced?
                # Usually we want to preserve the vertices (corners).
                # So let's just add intermediate points and p2.
                break
            pt = p1 + vec * d
            new_path.append(pt)
        
        # Ensure p2 is added if not close to last point
        if np.linalg.norm(new_path[-1] - p2) > 1e-6:
             new_path.append(p2)
             
    return np.array(new_path)

