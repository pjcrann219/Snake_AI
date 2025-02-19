import numpy as np

def closest_one_distances(grid, head_idx):
    """
    Calculates the distance to the nearest 1 in each direction from a given position.
    
    Parameters:
        grid (ndarray): Binary grid representation
        head_idx (list): [row, col] position to measure distances from
    
    Returns:
        list: Distances to nearest 1 in [up, down, left, right] directions
    """
    rows, cols = grid.shape
    r, c = head_idx
    
    # Initialize distances
    up_dist = down_dist = left_dist = right_dist = 0
    
    # Check up direction
    for i in range(r - 1, -1, -1):  # Traverse upwards
        if grid[i, c] == 1:
            up_dist = r - i
            break
        up_dist += 1
    
    # Check down direction
    for i in range(r + 1, rows):  # Traverse downwards
        if grid[i, c] == 1:
            down_dist = i - r
            break
        down_dist += 1
    
    # Check left direction
    for j in range(c - 1, -1, -1):  # Traverse left
        if grid[r, j] == 1:
            left_dist = c - j
            break
        left_dist += 1
    
    # Check right direction
    for j in range(c + 1, cols):  # Traverse right
        if grid[r, j] == 1:
            right_dist = j - c
            break
        right_dist += 1
    
    return [up_dist, down_dist, left_dist, right_dist]

def get_features_10(state, dim, norm = False):
    """
    Extracts feature vector from game state for neural network input.
    
    Parameters:
        state (ndarray): Game state representation
        dim (int): Dimension of the game board
        norm (bool): Whether to normalize features
    
    Returns:
        list: Feature vector containing relative positions and distances
    """
    fruit_idx = np.where(state[2,:,:] == 1)
    fruit_idx = [fruit_idx[0][0], fruit_idx[1][0]]
    head_idx = np.where(state[0,:,:] == 1)
    head_idx = [head_idx[0][0], head_idx[1][0]]

    delta_head_y = head_idx[0] - fruit_idx[0]
    delta_head_x = fruit_idx[1] - head_idx[1]

    wall_up = head_idx[0]
    wall_down = dim - head_idx[0] - 1
    wall_left = head_idx[1]
    wall_right = dim - head_idx[1] - 1

    body_up_down_left_right = closest_one_distances(state[1,:,:], head_idx)

    return [x/dim for x in [delta_head_x, delta_head_y, wall_up, wall_down, wall_left, wall_right] + body_up_down_left_right]
