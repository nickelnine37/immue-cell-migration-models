from skimage.feature import blob_log
import numpy as np
from scipy.spatial.distance import cdist
from utils.distributions import Normal, TruncatedNormal
from scipy.special import erfc
from scipy.optimize import linear_sum_assignment
import time
from utils.exceptions import DimensionsError
import skimage
from typing import Union
from utils.plotting import make_gif


def LoG(image: np.ndarray, min_sigma: int=3, max_sigma: int=10, num_sigma: int=10, threshold: float=0.05) -> np.ndarray:
    """
    Not much more than a call to the skimage function skimage.feature.blob_log for calculating the
    Laplacian of Guassians.

    https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log

    ----------
    image           A numpy array image, to detect blobs in

    Returns
    -------
    A numpy array of shape (N, 3) containing the N x-y-r coordinates
    of the detected cells

    """

    # call the skimage function for Laplacian of Guassians
    detected_cells = blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)

    out = detected_cells[:, np.array([1, 0, 2])]  # reorder columns so it's x-y-r
    out[:, 2] = out[:, 2] * 2 ** 0.5
    return out


def detect_all_cells(frames: np.ndarray, min_sigma: int=3, max_sigma: int=10, num_sigma: int=10, threshold: float=0.05):
    """

    Parameters
    ----------
    frames          A (T, X, Y) numpy array with the single-channel frames containing
                    pictures of the cells.
    min_sigma       The minimum sigma for calculating the Laplacian of Gaussians
    max_sigma       The maximum sigma
    num_sigma       The number of sigma increments
    threshold       The threshold for detection

    Returns
    -------

    A list of length T containing numpy arrays with the x-y-r coordinates of each cell
    in each frame.

    """
    all_cells = []
    t0 = time.time()
    T = frames.shape[0]

    print('Calculating Laplacian of Guassians')

    # loop through each frame
    for t in range(T):

        # calculate the LoG for this frame
        cells = LoG(frames[t, :, :], min_sigma, max_sigma, num_sigma, threshold)

        # print progress
        t_elapsed = time.time() - t0
        string = '\rFrame {}/{} completed: {} cells found. Approx time remaining: {:.2f}s'
        print(string.format(t+1, T, len(cells), t_elapsed * (T / (t + 1) - 1)), end='')

        all_cells.append(cells)

    print()
    print('Caclculated LoGs in {:.2f}s'.format(time.time() - t0))
    return all_cells


def p_in_out(positions: np.ndarray, box_x: int, box_y: int, step_size: float=20) -> float:
    """
    The probability that a cell steps out of camera view in the next frame,
    or has stepped in to camera view from the last frame. Unlikely to be
    useful for anything except internal functions (underscore prefix)

    Parameters
    ----------
    positions      A (N, 2) numpy array of x-y coordinates
    step_size      The standard deviation of the cells' step size, measured in pixels.
    box_x          The width of the images in pixels
    box_y          The height of the images in pixels

    Returns
    -------
    A (N, ) numpy array with the probability that each coordinate steps out of frame

    """

    point_x, point_y = positions[:, 0], positions[:, 1]
    dx = np.array([point_x, box_x - point_x]).min(0)
    dy = np.array([point_y, box_y - point_y]).min(0)

    erfx = erfc(dx / step_size * 2 ** 0.5)
    erfy = erfc(dy / step_size * 2 ** 0.5)

    return 0.5 * (erfx + erfy - 0.5 * erfx * erfy)


def find_transitions(cells1: np.ndarray, cells2: np.ndarray, box_x: int, box_y: int, step_size: float=20) -> list:
    """

    Parameters
    ----------
    cells1         A numpy array of size (n1, 2) that specifies the x-y coordinated of
                   all the cells present in frame t
    cells2         The same, for cells in frame t+1, (n2, 2)
    step_size      The standard deviation of the cells' step size, measured in pixels.
    box_x          The width of the images in pixels
    box_y          The height of the images in pixels

    Returns
    -------

    transitions    A list of all the transitions that occur between frame t and frame t+1.
                   Formatted as tuples, where the first element refers to the first frame
                   and the second element to the second frame. E.g. (2, 3) indicates that
                   the cell at position cells1[2, :] transitioned to the cell at position
                   cells2[3, :]. A cell appearing from nothing in frame 2 has a first
                   tuple element 'START' and a cell from frame 1 disappearing is indicated
                   by a second element 'END'.

                   e.g [(0, 1), (1, 0), ('START', 2)]

    """

    step_distribution = TruncatedNormal(mu=0, sig=step_size)

    # static probability of failire of detection
    p_LoG_failure = 0.05

    # find the number of detected cells in the t-th and t+1th frame
    n_cells1, n_cells2 = cells1.shape[0], cells2.shape[0]

    # this will be the matrix to perform the munkres algorithm on
    matrix = np.ones((n_cells1 + n_cells2, n_cells1 + n_cells2))

    # get the value proportional to the pdf for a gaussian 2d step, at each pairwise distance
    distance_matrix = cdist(cells1, cells2)
    pdf_matrix = step_distribution.pdf(distance_matrix)

    # if distance is greater than 6 sigma, set the probability to zero
    # pdf_matrix[distance_matrix > 6 * step_size] = 0

    # the probability that cells at t move out of frame. shape: (n1, 1)
    p_out = p_in_out(cells1, box_x, box_y, step_size)[:, None]

    # the probabillity that cells at t+1 came from out of frame: shape: (1, n2)
    p_in = p_in_out(cells2, box_x, box_y, step_size)[None, :]

    # add probability of failing to be detected
    p_out += (1 - p_out) * p_LoG_failure
    p_in += (1 - p_in) * p_LoG_failure

    p_out_pdf = step_distribution.pdf_at_prob(p_out)
    p_in_pdf = step_distribution.pdf_at_prob(p_in)

    # renormalise pdf values
    # pdf_matrix /= (pdf_matrix.sum(1)[:, None] + 1e-20)

    # set upper left quadrant of matrix to be pdf matrix
    matrix[:n_cells1, :n_cells2] = pdf_matrix

    # set other upper right and lower left quadrants
    matrix[:n_cells1, n_cells2:] = np.eye(n_cells1) * p_out_pdf
    matrix[n_cells1:, :n_cells2] = np.eye(n_cells2) * p_in_pdf

    # take logs and fix
    cost_matrix = - np.log(matrix)
    cost_matrix[cost_matrix == np.inf] = 1e9

    # solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    transitions = []

    # interpret the output into more understandable format
    for frame1_cell_no, frame2_cell_no in zip(row_ind, col_ind):

        if frame2_cell_no >= n_cells2 and frame1_cell_no >= n_cells1:
            # lower right quadrant: nothing interesting
            pass

        elif frame2_cell_no >= n_cells2:
            # upper right quadrant: id1 disappeared after frame t: end of a path
            transitions.append((frame1_cell_no, 'END'))

        elif frame1_cell_no >= n_cells1:
            # lower left quadrant: id2 appeared in frame t+1: start of a new path
            transitions.append(('START', frame2_cell_no))

        else:
            # upper left quadrant: regular transition
            transitions.append((frame1_cell_no, frame2_cell_no))

    return transitions


def paths_to_array(paths: list, min_t: int=5) -> np.ndarray:
    """
    Convert a list of paths, with their start times, into a single numpy
    array that can be passed to the MCMC inference class. It has shape
    (longest_path, number_of_paths), with paths shorter than the longest
    path having their finishing values filled with nans.

    [(t1, path1), (t2, path2), (t3, path3), ... ]

    -->
       ---  ---  ---
    [ ---  ---  ---
       |    |    |
       |    |  path3
     path1  |    |--
       |  path2 ---
       |--  |   nan   ...
      ---   |   nan
      nan   |-- nan
      nan  ---  nan
      nan  nan  nan


    Parameters
    ----------
    paths           A list of tuples, with the frame number that th path
                    appears in as the first element, and the (t, 2) numpy
                    array of x-y coordinates as the second element.
    min_t           The minimum number of steps a path should have in
                    order to be included in the matrix. Setting this
                    to a higher value speeds up inference a lot as the
                    matrix has far fewer columns.

    Returns
    -------
    paths_array     A numpy array formatting of the paths ready to be
                    passed to the inference class.
    """


    max_t = max(paths, key=lambda x: x[1].shape[0])[1].shape[0]
    paths = [(t0, path) for t0, path in paths if path.shape[0] >= min_t]

    paths_array = np.zeros((max_t, 2, len(paths)))
    paths_array[:] = np.nan

    for i, (t0, path) in enumerate(paths):
        paths_array[:path.shape[0], :, i] = path

    return paths_array


def track_paths(cells: list, box_x: int, box_y: int, step_size: float=20):
    """
    Given a list of numpy arrays, which represent the x-y-r coordinates
    of the cells detected in each of T frames, find all the paths.

    Parameters
    ----------
    cells          A list of (n_t, 3) or (n_t, 2) numpy arrays, with the x-y(-r) coordinates of
                   the cells detected in frame t.
    box_x          The width of the images in pixels
    box_y          The height of the images in pixels
    step_size      The standard deviation of the cells' step size, measured in pixels.


    Returns
    -------
    paths          A list of all paths detected. Paths are represented as a tuple: (int, np.array)
                   where the first element represents the starting frame of the path, and the
                   second element of shape (n, 2) represents the x-y coordinates of the path taken

    """

    # the number of frames present
    T = len(cells)

    # add a start tag to each path
    transitions_list = [[('START', i) for i in range(cells[0].shape[0])]]

    t0 = time.time()
    print('Beginning Hungarian Optimisation')

    # find the transitions between each frame
    for t in range(T - 1):
        transitions = find_transitions(cells[t][:, :2], cells[t+1][:, :2], box_x, box_y, step_size)
        transitions_list.append(transitions)

    print('Completed Hungarian Optimisation in {:.2f}s'.format(time.time() - t0))

    # add ending tags to a final paths
    transitions_list.append([(id2, 'END') for id1, id2 in transitions_list[-1] if id2 != 'END'])

    # identify when and where paths are starting
    path_starts = [(t, id2) for t in range(T) for id1, id2 in transitions_list[t] if id1 == 'START']
    paths = []

    # for each place that we believe a path starts, follow it through the frames
    for start_time, start_index in path_starts:

        # get the initial x-y coordinates of the path
        path = [cells[start_time][start_index, :2]]

        # initialise
        previous_index = start_index
        finished = False

        for t in range(start_time, T):

            # if we've found the end of the path, stop checking
            if finished:
                break

            # search through transitions
            transitions = transitions_list[t+1]

            for id1, id2 in transitions:

                if id1 == previous_index:
                    # we've found the next link in the chain

                    if id2 != 'END':
                        # keep going
                        path.append(cells[t+1][id2, :2])
                    else:
                        # add a new path
                        path = np.array(path)
                        paths.append((start_time, path))
                        finished = True

                    previous_index = id2
                    break

    return paths


class CellTracker:
    """
    This class uses a Laplacian of Guassians to detect cells in each frame,
    then uses a probability matrix and the Hungarian algorithm to determine
    trajectories.
    """

    def __init__(self, frames: Union[np.ndarray, str], min_sigma: int=3, max_sigma: int=10,
                 num_sigma: int=10, LOG_threshold: float=0.05, step_size: int=10):
        """
        Parameters
        ----------
        LOG_threshold   The threshold value for calculating the Laplacian of
                        Guassians. Passed to skimage.blobs_log
        step_size       The standard deviation of the step size for cells,
                        measured in pixels.
        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = LOG_threshold
        self.step_size = step_size

        self.frames = None
        self.cells = None
        self.paths = None
        self.paths_array = None

        if isinstance(frames, np.ndarray):
            self.load_from_numpy(frames)
        elif isinstance(frames, str):
            self.load_from_tif(frames)
        else:
            raise TypeError('frames must be a string to a tif file or a numpy array')

    def load_from_tif(self, tif_file: str):
        """
        Load cell image data from a tif file.

        Parameters
        -------
        tif_file    String, path to tif file

        """

        self.frames = skimage.io.imread(tif_file)[:, 0, :, :]
        if self.frames.shape[1] != 1:
            print('WARNING: {} color channels found. Taking first only'.format(self.frames.shape[1]))

        self.T, self.Y, self.X = self.frames.shape

    def load_from_numpy(self, array: np.ndarray, time_first: bool=True):
        """
        Load cell image data from a numpy array.

        Parameters
        ----------
        array       A (T, X, Y) or (X, Y, T) numpy array, with T 2D frames
        time_first  Whether time appears in the first axis.

        Returns
        -------

        """

        # handle dimensions
        dims = len(array.shape)
        if dims != 3:
            s = ''
            if dims == 2:
                s = ' (Try adding a time dimension?)'
            elif dims == 4:
                s = ' (Try removing color channel?)'
            raise DimensionsError('array must be 3D but it is {}D.'.format(dims) + s)

        # check if the t-dimension is in index 0
        if not time_first:
            self.frames = array.transpose((2, 0, 1))
        else:
            self.frames = array

        self.T, self.Y, self.X = self.frames.shape

    def make_paths_matrix(self, min_t: int=5):
        """
        Output a matrix of paths ready for use in MCMC inference

        Parameters
        ----------
        min_t

        Returns
        -------

        """

        # get a full list of all cells in all frames
        if self.cells is None:
            self.cells = detect_all_cells(self.frames, self.min_sigma, self.max_sigma, self.num_sigma, self.threshold)

        # detect the transitions between frames and find the fulls paths
        if self.paths is None:
            self.paths = track_paths(self.cells, self.X, self.Y, self.step_size)

        return paths_to_array(self.paths, min_t)

    def make_paths_list(self):

        # get a full list of all cells in all frames
        if self.cells is None:
            self.cells = detect_all_cells(self.frames, self.min_sigma, self.max_sigma, self.num_sigma, self.threshold)

        # detect the transitions between frames and find the fulls paths
        if self.paths is None:
            self.paths = track_paths(self.cells, self.X, self.Y, self.step_size)

        return self.paths

    def make_gif(self, save_as: str='my_giff', delay: int=20, add_LoG: bool=False):

        # get a full list of all cells in all frames
        if self.cells is None:
            self.cells = detect_all_cells(self.frames, self.min_sigma, self.max_sigma, self.num_sigma, self.threshold)

        # detect the transitions between frames and find the fulls paths
        if self.paths is None:
            self.paths = track_paths(self.cells, self.X, self.Y, self.step_size)

        if add_LoG:
            make_gif(self.frames, save_as=save_as, delay=delay, paths=self.paths, cells=self.cells)
        else:
            make_gif(self.frames, save_as=save_as, delay=delay, paths=self.paths)


if __name__ == '__main__':

    from collections import defaultdict
    from in_silico.rw_inference import BPbayesian
    from in_silico.sources import PointSource
    from utils.plotting import plot_wpb_dist
    import pandas as pd
    from utils.plotting import make_gif

    tif_file = '/media/ed/DATA/Datasets/Leukocytes/Control wounded 1hr/Pupae 1 concatenated ubi-ecad-GFP, srpGFP; srp-3xH2Amch x white-1HR.tif'

    def run_experiments():

        EXPERIMENT1 = False  # run inference on the paths output in the Weavers csv
        EXPERIMENT2 = True  # run inference on the paths I extract and track from the videos
        EXPERIMENT3 = False  # run inference on the detected points from weavers, linked together via my tracking algorithm

        centre = np.array([350, 420])
        source = PointSource(position=centre)
        frames = skimage.io.imread(tif_file)[:, 0, :, :]
        T, Y, X = frames.shape

        min_sigma = 3
        max_sigma = 10
        num_sigma = 10
        threshold = 0.05
        step_size= 10
        min_t = 5

        if EXPERIMENT1:

            title = 'Path data taken from csv trajectories'

            # extract the data from the csv files
            data_ = pd.read_csv(
                '/media/ed/DATA/Datasets/Leukocytes/Control wounded 1hr/control 1hour wound Links in tracks statistics.csv')
            stats = pd.read_csv(
                '/media/ed/DATA/Datasets/Leukocytes/Control wounded 1hr/control 1hr wound Track statistics.csv')

            # 30 seconds per frame
            stats['TRACK_START'] = stats['TRACK_START'].astype(int) // 30
            data_ = data_[['TRACK_ID', 'EDGE_X_LOCATION', 'EDGE_Y_LOCATION']]

            # convert from micro meters to pixels
            factor = 3.72

            paths = []  # (t0, [path])
            for id, path in data_.groupby('TRACK_ID'):
                t0 = stats[['TRACK_START', 'TRACK_ID']][stats['TRACK_ID'] == 0]['TRACK_START'].values[0]
                paths.append((t0, factor * path[['EDGE_X_LOCATION', 'EDGE_Y_LOCATION']].values))

            paths_array = paths_to_array(paths, 5)
            BI = BPbayesian(paths_array, source)

            t1 = time.time()
            params = BI.multi_infer(n_walkers=5,
                                    n_steps=10000,
                                    burn_in=3000,
                                    sneaky_init=True)
            plot_wpb_dist(params, title=title, save_as='experiment1')
            t2 = time.time()
            print('Completed inference in {:.2f}s'.format(t2 - t1))

            make_gif(frames, paths=paths, save_as='experiment1')

        if EXPERIMENT2:

            title = 'x-y coordinates extracted and tracked manually'

            tracker = CellTracker(frames, min_sigma, max_sigma, num_sigma, threshold, step_size)
            paths_matrix = tracker.make_paths_matrix(min_t)
            BI = BPbayesian(paths_matrix, source)

            t1 = time.time()
            params = BI.multi_infer(n_walkers=5,
                                    n_steps=10000,
                                    burn_in=3000,
                                    sneaky_init=True)
            plot_wpb_dist(params, title=title, save_as='experiment2')
            t2 = time.time()
            print('Completed inference in {:.2f}s'.format(t2 - t1))

            make_gif(frames, paths=tracker.paths, save_as='experiment2')


        if EXPERIMENT3:

            title = 'x-y coordinates taken from csv, trajectories manually tracked'

            # extract the data from the csv files
            data_ = pd.read_csv(
                '/media/ed/DATA/Datasets/Leukocytes/Control wounded 1hr/control 1hour wound Links in tracks statistics.csv')
            stats = pd.read_csv(
                '/media/ed/DATA/Datasets/Leukocytes/Control wounded 1hr/control 1hr wound Track statistics.csv')

            # 30 seconds per frame
            stats['TRACK_START'] = stats['TRACK_START'].astype(int) // 30
            data_ = data_[['TRACK_ID', 'EDGE_X_LOCATION', 'EDGE_Y_LOCATION']]

            # convert from micro meters to pixels (found by trial and error)
            factor = 3.72

            paths = []  # (t0, [path])
            for id, path in data_.groupby('TRACK_ID'):
                t0 = stats[['TRACK_START', 'TRACK_ID']][stats['TRACK_ID'] == 0]['TRACK_START'].values[0]
                paths.append((t0, factor * path[['EDGE_X_LOCATION', 'EDGE_Y_LOCATION']].values))

            cells = defaultdict(list)
            for t0, path in paths:
                for i in range(path.shape[0]):
                    cells[t0 + i].append(path[i, :])

            cells = sorted(list(cells.items()), key=lambda pair: pair[0])
            cells_ = []
            for t0, path in cells:
                cells_.append(np.array(path))

            paths = track_paths(cells_, X, Y, step_size)
            paths_matrix = paths_to_array(paths, min_t)
            BI = BPbayesian(paths_matrix, source)

            t1 = time.time()
            params = BI.multi_infer(n_walkers=5,
                                    n_steps=10000,
                                    burn_in=3000,
                                    sneaky_init=True)
            plot_wpb_dist(params, title=title, save_as='experiment3')
            t2 = time.time()
            print('Completed inference in {:.2f}s'.format(t2 - t1))

            make_gif(frames, paths=paths, save_as='experiment3')


    run_experiments()



