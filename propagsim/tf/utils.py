import tensorflow as tf
import warnings
import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)


def append(a: tf.Tensor, b: tf.Tensor):
    """
    b is a scalar tensor
    """
    a = tf.cast(a, tf.float32)
    b = tf.expand_dims(b, 0)
    b = tf.cast(b, tf.float32)
    appended_tensor = tf.concat([a, b], axis=-1)
    return appended_tensor


def squarify(xcoords: tf.Tensor, ycoords: tf.Tensor):
    """
    https://stackoverflow.com/questions/42237575/find-unique-pairs-of-values-in-tensorflow
    """

    assert xcoords.dtype == tf.float32
    assert ycoords.dtype == tf.float32

    xy = tf.concat([xcoords, ycoords], axis=1)
    xy = tf.round(xy)
    xy64 = tf.bitcast(xy, type=tf.float64)

    unique64, idx = tf.unique(xy64)
    unique_points = tf.bitcast(unique64, type=tf.float32)

    return unique_points, idx


def get_square_sampling_probas(attractivity_cells, square_ids_cells, coords_squares, dscale=1):
    # compute sum attractivities in squares
    sum_attractivity_squares, unique_squares = sum_by_group(values=attractivity_cells, groups=square_ids_cells)
    # Compute distances between all squares and squares having sum_attractivity > 0
    mask_attractivity = (sum_attractivity_squares > 0)
    eligible_squares = unique_squares[mask_attractivity]
    sum_attractivity_squares = sum_attractivity_squares[mask_attractivity]
    print(f'get_square_sampling_probas - shape of mask_attractivity: {mask_attractivity.shape}')
    print(f'get_square_sampling_probas - shape of eligible_squares: {eligible_squares.shape}')
    # Compute distance between cells, add `intra_square_dist` for average intra cell distance
    inter_square_dists = cdist(coords_squares)
    inter_square_dists = tf.cast(inter_square_dists, tf.float32)
    inter_square_dists = tf.gather(inter_square_dists, eligible_squares, axis=1)

    print(f'get_square_sampling_probas - shape of inter_square_dists: {inter_square_dists.shape}')
    # NOTE: mulitplier le dscale par -0.1 plutôt ?
    square_sampling_probas = tf.multiply(inter_square_dists, -0.1 * dscale)
    square_sampling_probas = tf.math.exp(square_sampling_probas)
    square_sampling_probas = tf.multiply(square_sampling_probas, sum_attractivity_squares)
    square_sampling_probas /= tf.linalg.norm(square_sampling_probas, ord=1, axis=1, keepdims=True)
    square_sampling_probas = tf.cast(square_sampling_probas, tf.float32)
    print(f'get_square_sampling_probas - shape of square_sampling_probas: {square_sampling_probas.shape}')
    return square_sampling_probas


def get_cell_sampling_probas(attractivity_cells, square_ids_cells):
    unique_square_ids, inverse, counts = tf.unique_with_counts(square_ids_cells)
    print(f'get_cell_sampling_probas - shape of unique_square_ids: {unique_square_ids.shape}')
    print(f'get_cell_sampling_probas - shape of inverse: {inverse.shape}')
    print(f'get_cell_sampling_probas - shape of counts: {counts.shape}')
    # `inverse` is an re-numering of `square_ids_cells` following its order: 3, 4, 6 => 0, 1, 2
    width_sample = int(tf.reduce_max(counts).numpy())
    print(f'get_cell_sampling_probas - value of width_sample: {width_sample}')
    # create a sequential index dor the cells in the squares: 
    # 1, 2, 3... for the cells in the first square, then 1, 2, .. for the cells in the second square
    # Trick: 1. shift `counts` one to the right, remove last element and append 0 at the beginning:

    # replace insert
    cell_index_shift = tf.concat([tf.zeros(shape=1, dtype=tf.int32), counts[:-1]], axis=0)
    cell_index_shift = tf.cumsum(cell_index_shift)  # [0, ncells in square0, ncells in square 1, etc...]
    print(f'get_cell_sampling_probas - shape of cell_index_shift: {cell_index_shift.shape}')
    to_subtract = repeat(cell_index_shift, counts)  # repeat each element as many times as the corresponding square has cells
    print(f'get_cell_sampling_probas - shape of to_subtract: {to_subtract.shape}')

    inds_cells_in_square = tf.range(0, attractivity_cells.shape[0])
    inds_cells_in_square = tf.math.subtract(inds_cells_in_square, to_subtract)  # we have the right sequential order

    print(inds_cells_in_square)
    print(f'get_cell_sampling_probas - shape of inds_cells_in_square: {inds_cells_in_square.shape}')
    print(f'get_cell_sampling_probas - dtype of inds_cells_in_square: {inds_cells_in_square.dtype}')

    order = tf.argsort(inverse)
    print(f'get_cell_sampling_probas - shape of order: {order.shape}')
    inverse = tf.gather(inverse, order, axis=-1)
    attractivity_cells = tf.gather(attractivity_cells, order, axis=-1)

    # Create `sample_arr`: one row for each square. The values first value in each row are the attractivity of its cell. Padded with 0.
    print(f'DEBUG: type(unique_square_ids.shape[0]): {type(unique_square_ids.shape[0])}, type(width_sample): {type(width_sample)}')
    cell_sampling_probas = tf.zeros((unique_square_ids.shape[0], width_sample))

    # assignment wih TF ?
    cell_sampling_probas = tf.numpy_function(
        func=assign_tensor_with_numpy,
        inp=[
            cell_sampling_probas,
            inverse,
            inds_cells_in_square,
            attractivity_cells
        ],
        Tout=tf.float32
    )
    print(f'get_cell_sampling_probas - shape of cell_sampling_probas: {cell_sampling_probas.shape}')
    # Normalize the rows of `sample_arr` s.t. the rows are probability distribution
    cell_sampling_probas /= tf.cast(tf.linalg.norm(cell_sampling_probas, ord=1, axis=1, keepdims=True), tf.float32)

    return cell_sampling_probas, cell_index_shift


def vectorized_choice(prob_matrix, axis=1):
    """
    selects index according to weights in `prob_matrix` rows (if `axis`==0), cols otherwise 
    see https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix
    """
    # s = prob_matrix.cumsum(axis=axis)
    # r = tf.random.uniform(shape=[prob_matrix.shape[1 - axis]])
    r = tf.random.uniform(shape=[tf.shape(prob_matrix)[1 - axis]])
    r = tf.reshape(r, (2 * (1 - axis) - 1, 2 * axis - 1))
    k = tf.math.reduce_sum(tf.cast(prob_matrix < r, tf.float32), axis=axis)
    # max_choice = prob_matrix.shape[axis]
    # k[k>max_choice] = max_choice
    return k


def group_max(data, groups):
    """
    To Do Later
    """
    groups = tf.cast(groups, tf.float32)
    groups_np = groups.numpy()
    order = tf.sort(tf.stack([data, groups], axis=1))
    order = tf.cast(order, tf.int32)
    groups = tf.gather(groups, order) # this is only needed if groups is unsorted
    data = tf.gather(data, order)
    # index = cp.empty(groups.shape[0], 'bool')
    index = np.ones(groups.shape[0], dtype=bool)
    index[-1] = True
    index[:-1] = (groups_np[1:] != groups_np[:-1])
    result = (tf.boolean_mask(data, index), tf.convert_to_tensor(index))
    return result


def group_max_np(data, groups):
    # data = data.numpy()
    # groups = groups.numpy()
    order = np.lexsort((data, groups))
    groups = groups[order] # this is only needed if groups is unsorted
    data = data[order]
    index = np.empty(groups.shape[0], 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    return data[index], index


def sum_by_group(values: tf.Tensor, groups: tf.Tensor):

    assert values.shape == groups.shape
    print(f'sum_by_group - shape of input values: {values.shape}')
    print(f'sum_by_group - shape of input groups: {groups.shape}')

    unique_groups, _ = tf.unique(groups)
    values = tf.math.unsorted_segment_sum(values, groups, unique_groups.shape[0])

    print(f'sum_by_group - shape of ouput values: {values.shape}')
    print(f'sum_by_group - shape of output groups: {unique_groups.shape}')

    return values, unique_groups


def assign_tensor_with_numpy(tensor, slice_x, slice_y, assignement):
    tensor[slice_x, slice_y] = assignement
    return tensor

# For distances:
# see: https://stackoverflow.com/questions/52030458/vectorized-spatial-distance-in-python-using-numpy


def cdist(a: tf.Tensor):
    print(f'cdist - shape of input a: {a.shape}')

    dist = pairwise_dist(a, a)

    print(f'cdist - shape of output dist: {dist.shape}')
    return dist


def repeat(data, count):
    repeated_tensor = tf.repeat(data, count)
    return repeated_tensor
    # data, count = data.tolist(), count.tolist()
    # return cp.array(list(itertools.chain(*(itertools.repeat(elem, n) for elem, n in zip(data, count)))))


def pairwise_dist(A, B):
    """
    See: https://github.com/tensorflow/tensorflow/issues/30659
    Computes pairwise euclidean distances between each elements of A and each elements of B.
    Args:
        A,    [m,d] matrix
        B,    [n,d] matrix
    Returns:
        D,    [m,n] matrix of pairwise distances
    """
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidean difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D
