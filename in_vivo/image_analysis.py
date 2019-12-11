import sys
import os
sys.path.append(os.path.abspath('..'))
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from numbers import Number
from skimage.io import imread
from skimage.feature import blob_log, blob_dog
from scipy.special import erfc
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from utils.exceptions import ArgumentError
from utils.distributions import TruncatedNormal
from utils.plotting import make_gif
from in_silico.sources import Source


def detect_cells(image: np.ndarray, detector: str='LoG', min_sigma: int=3, max_sigma: int=10,
                 num_sigma: int=10, threshold: float=0.05) -> np.ndarray:
    """
    Detect cells either using Laplcian of Gaussians or Difference og Gaussians. Pass an input image of pixel
    intensities of shape (Y, X) and return the x-y-r coordinates of all the detected cells.

    For details, see: https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log

    Parameters
    ----------
    image           A numpy array of shape (Y, X). This contains the raw pixel intensity values
    detector        Should be one of 'LoG' or 'DoG'. Whether to use Laplacian of Gaussians or Differnce of Gaussians.
    min_sigma       The minimum Gaussian width to detect cells at
    max_sigma       The maximum Gaussian width to detect cells at
    num_sigma       The number of increments between min_sigma and max_sigma
    threshold       The minimum value of the convolved image to identify a cell

    Returns
    -------
    A numpy array of shape (N, 3) containing the N x-y-r coordinates of the detected cells

    """

    # call the appropriate skimage function
    if detector == 'LoG':
        detected_cells = blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    elif detector == 'DoG':
        detected_cells = blob_dog(image, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    else:
        raise ArgumentError('the argument "detector" should be either "LoG" or "DoG"')

    out = detected_cells[:, np.array([1, 0, 2])]  # reorder columns so it's x-y-r
    out[:, 2] = out[:, 2] * 2 ** 0.5
    return out


def detect_all_cells(frames: np.ndarray, detector: str='LoG', min_sigma: int=3, max_sigma: int=10,
                     num_sigma: int=10, threshold: float=0.05) -> dict:
    """

    Just repeated calls to the LoG function. Pass a block of frames and get back a dictionary
    containing all cell detections in each frame.

    Parameters
    ----------
    frames          A (T, Y, X) numpy array with the single-channel frames containing
                    the pixel intensity readins.
    min_sigma       The minimum Gaussian width to detect cells at
    max_sigma       The maximum Gaussian width to detect cells at
    num_sigma       The number of increments between min_sigma and max_sigma
    threshold       The minimum value of the convolved image to identify a cell

    Returns
    -------

    A dictionary, indexed by frame number, containing numpy arrays with the x-y-r coordinates
    of each cell in that frame.

    """

    all_cells = {}
    T = frames.shape[0]

    time.sleep(0.1)

    # loop through each frame
    pbar = tqdm(range(T))
    pbar.set_description('Detecting cells using {}.'.format(detector))
    for t in pbar:

        # calculate the LoG for this frame
        cells = detect_cells(frames[t, :, :], detector=detector, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
        all_cells[t] = cells
        pbar.set_description('Detecting cells using {}. Found {} cells.'.format(detector, cells.shape[0]))

    return all_cells


def p_in_out(positions: np.ndarray, box_x: int, box_y: int, step_size: float=10) -> float:
    """
    The probability that a cell steps out of camera view in the next frame,
    or has stepped in to camera view from the last frame.

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


def find_transitions(cells1: np.ndarray, cells2: np.ndarray, box_x: int, box_y: int,
                     step_size: float=10, d: float=None, b: float=None, t_sep: int=1) -> list:
    """

    Use the Hungarian (Munkres) algorithm to output the transitions between cell position detections
    in two consecutive frames. Based on the implementation 'Robust single-particle tracking in live-cell
    time-lapse sequences' by Jaqaman et al, with some minor modifications.

    See https://www.nature.com/articles/nmeth.1237 for details.

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


    sig = step_size * t_sep ** 0.5
    step_distribution = TruncatedNormal(sig=sig)

    if d is None:
        d = step_distribution.pdf(2 * sig)
    if b is None:
        b = step_distribution.pdf(2 * sig)

    # find the number of detected cells in the t-th and t+1th frame
    n_cells1, n_cells2 = cells1.shape[0], cells2.shape[0]

    # this will become the matrix to perform the munkres algorithm on
    matrix = np.ones((n_cells1 + n_cells2, n_cells1 + n_cells2))

    # get the value proportional to the pdf for a gaussian 2d step, at each pairwise distance
    pdf_matrix = step_distribution.pdf(cdist(cells1, cells2))

    d_to_closest_edge = np.array([cells1[:, 0], box_x - cells1[:, 0], cells1[:, 1], box_y - cells1[:, 1]]).min(0)[:, None]
    d_from_closest_edge = np.array([cells2[:, 0], box_x - cells2[:, 0], cells2[:, 1], box_y - cells2[:, 1]]).min(0)[None, :]

    # the probability that cells at t move out of frame. shape: (n1, 1)
    # p_out = p_in_out(cells1, box_x, box_y, step_size)[:, None]

    # the probabillity that cells at t+1 came from out of frame: shape: (1, n2)
    # p_in = p_in_out(cells2, box_x, box_y, step_size)[None, :]

    # add probability of failing to be detected
    # p_out += (1 - p_out) * p_LoG_failure
    # p_in += (1 - p_in) * p_LoG_failure

    # p_out_pdf = step_distribution.pdf(step_distribution.ppf(1 - p_out))
    # p_in_pdf = step_distribution.pdf(step_distribution.ppf(1 - p_in))

    # set upper left and lower right quadrant of matrix to be pdf matrix
    matrix[:n_cells1, :n_cells2] = pdf_matrix
    matrix[n_cells1:, n_cells2:] = pdf_matrix.T

    # # set other upper right and lower left quadrants
    # matrix[:n_cells1, n_cells2:] = np.eye(n_cells1) * p_out_pdf
    # matrix[n_cells1:, :n_cells2] = np.eye(n_cells2) * p_in_pdf

    matrix[:n_cells1, n_cells2:] = np.eye(n_cells1) * (step_distribution.pdf(d_to_closest_edge) + d)
    matrix[n_cells1:, :n_cells2] = np.eye(n_cells2) * (step_distribution.pdf(d_from_closest_edge) + b)

    # take logs and fix
    with np.errstate(divide='ignore'):
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


def find_paths(cells: dict, box_x: int, box_y: int, step_size: float=10,
               d: float=None, b: float=None) -> list:
    """
    Given a disctionary which contains numpy arrays, indexed by frame number t, which represent
    the x-y-r coordinates of the cells detected in each of T frames, find all the paths that
    link them together.

    Parameters
    ----------
    cells          A list of (n_t, 3) or (n_t, 2) numpy arrays, with the x-y(-r) coordinates of
                   the cells detected in frame t.
    box_x          The width of the images in pixels
    box_y          The height of the images in pixels
    step_size      The standard deviation of the cells' step size, measured in pixels.


    Returns
    -------
    paths          A list of all paths detected. Paths are represented as a numpy array of shape
                   (n, 3). The first column indexes the frame number, and the second two columns
                   hold the x and y coordinates of that cell at that time respectively.

    """

    # the number of frames present
    T = len(cells)

    # add a start tag to each path
    transitions_list = [[('START', i) for i in range(cells[0].shape[0])]]

    # find the transitions between each frame
    for t in tqdm(range(T - 1), desc='Running LAP across frames'):
        transitions = find_transitions(cells[t][:, :2], cells[t+1][:, :2], box_x, box_y, step_size, d, b)
        transitions_list.append(transitions)

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
                        t_index = np.arange(start_time, start_time + path.shape[0] - 0.1, 1)[:, None]
                        paths.append(np.concatenate([t_index, path], axis=1))
                        finished = True

                    previous_index = id2
                    break

    return paths


def link_paths(paths: list, box_x: int, box_y: int, step_size: float=10,
               d: float=None, b: float=None) -> list:

    time.sleep(0.1)
    print('Running LAP to link broken trajectories')
    t0 = time.time()

    # find all the places where paths start and stop mid-movie
    final_frame = np.concatenate(paths)[:, 0].max()
    starts = [(path_id, path[0, 0], path[0, 1], path[0, 2]) for path_id, path in enumerate(paths) if path[0, 0] != 0]
    stops = [(path_id, path[-1, 0], path[-1, 1], path[-1, 2]) for path_id, path in enumerate(paths) if path[-1, 0] != final_frame]

    starts_ = (pd.DataFrame(starts, columns=['path_id', 'start_time', 'x', 'y']).sort_values(by=['start_time']))
    stops_ = (pd.DataFrame(stops, columns=['path_id', 'stop_time', 'x', 'y']).sort_values(by=['stop_time']))

    stops_ = {stop_t: data.drop('stop_time', inplace=False, axis=1) for stop_t, data in stops_.groupby('stop_time')}
    starts_ = {start_time: data.drop('start_time', inplace=False, axis=1) for start_time, data in
               starts_.groupby('start_time')}

    stops = stops_.copy()
    starts = starts_.copy()

    links = []

    max_t = int(np.concatenate(paths)[:, 0].max())

    for t_sep in range(1, 10):

        for i in range(0, int(max_t)):

            try:

                stop_coords = stops[i][['x', 'y']].values
                start_coords = starts[i + t_sep][['x', 'y']].values
                out = find_transitions(stop_coords, start_coords, box_x, box_y, t_sep=t_sep, step_size=step_size, d=d, b=b)

            except KeyError:
                continue

            to_delete = np.array([[a, b] for a, b in out if a != 'START' and b != 'END'])

            if len(to_delete) > 0:

                stop_inds = to_delete[:, 0]
                start_inds = to_delete[:, 1]

                stop_ids = stops[i]['path_id'].values[stop_inds]
                start_ids = starts[i + t_sep]['path_id'].values[start_inds]

                for stop_id, start_id in zip(stop_ids, start_ids):
                    links.append((stop_id, start_id))

                stops[i].drop(stops[i].index[stop_inds], inplace=True)
                starts[i + t_sep].drop(starts[i + t_sep].index[start_inds], inplace=True)

    linked_paths = set(np.concatenate([np.array(links)[:, 0], np.array(links)[:, 1]]))

    new_paths = []
    for i, path in enumerate(paths):
        if i not in linked_paths:
            new_paths.append([path])

    for start in np.array(links)[:, 0]:
        path = []
        while True:
            for link in links:
                if link[0] == start:
                    path.append(paths[start])
                    start = link[1]
                    continue

            path.append(paths[start])
            break

        new_paths.append(path)

    print('Connected {} broken trajectories in {:.2f}s'.format(len(paths) - len(new_paths), time.time() - t0))

    return [np.concatenate(sub_paths, axis=0) for sub_paths in new_paths]


def segment_paths(paths: list,
                  source: Source,
                  t_bins: tuple = ((0, 15), (15, 30), (30, 45), (45, 60), (60, 90), (90, 120)),
                  r_bins: tuple=((0, 50), (50, 100), (100, 150)),
                  min_path_length: int=5,
                  path_units: str='pixels',
                  t_units: str = 'minutes',
                  r_units: str='microns',
                  seconds_per_frame: float=30,
                  microns_per_pixel: float=0.269) -> dict:
    """

    Parameters
    ----------
    paths               A list of numpy arrays, which represent the paths taken. Each array should have shape
                        (t, 3). The first column holds the frame number, and the second and third column hold
                        the x and y coordinates respectively. The frame number is needed to segment the paths
                        in time.

    source              A Source object, which hold the coordinates of the source. This is necessary to segment
                        the paths by distance from the wound.

    r_bins              A tuple of tuples, holding the bin edges for the spatial bins. This can be in units of
                        pixels or microns, which should be specified in the r_units argument.

    t_bins              Another tuple of tuples, holding the edges for the temporal bins. Should be in units of
                        seconds or minutes which, again, must be specified.

    min_path_length     This is the minimum path length to be included in the output.

    source_units        The units for the spatial coordinates of the source. Should be pixels or microns. It's
                        sometimes easier to provide this in pixels, because you can extract it visually from
                        the imshow plots.

    r_units             The units that the spatial bins have been provided in. Should be microns or pixels.

    t_units             The units that the temporal bins have been provided in. Should be seconds or minutes.

    Returns
    -------

    A dictionary is returned. The indicies for this dictionary will be a tuple of spatial and temporal bins. For
    example, one index could be ((0, 900), (0, 50)). This would represent the temporal/spatial bin of all cells
    falling within 0-900 seconds and 0-50 microns. This will always have the form ((t1, t2), (r1, r2)) where
    time is in units of seconds and distance is in units of microns, regardless of the units of the input
    variables. The items in this dictionary will be a list of numpy arrays. Like the paths variable that was
    passed, each array in this list has shape (t, 3), where column 0 holds the frame number, and columns 1 and 2
    hold the x and y coordinates respectively. Again, regardless of the units that the input variables had, these
    arrays will be in units of microns.

    """

    # make sure the units are valid
    fail_string = '"{}" should be one of {} but it is {}'
    distance_units = ['pixels', 'microns']
    time_units = ['seconds', 'minutes']
    assert path_units in distance_units, fail_string.format('path_units', distance_units, path_units)
    assert r_units in distance_units, fail_string.format('r_units', distance_units, r_units)
    assert t_units in time_units, fail_string.format('t_units', time_units, t_units)


    # add a column to each which holds an integer, representing the path number
    paths = [np.concatenate([path_id * np.ones((path.shape[0], 1)), path], axis=1) for path_id, path in enumerate(paths)]

    # concatenate all paths, and make into a dataframe
    df = pd.DataFrame(np.concatenate(paths), columns=['path', 'frame', 'x (μm)', 'y (μm)'])

    # ensure x, y are in μm
    if path_units is 'pixels':
        df['x (μm)'] *= microns_per_pixel
        df['y (μm)'] *= microns_per_pixel

    # ensure the source position is in μm
    if source.units is 'pixels':
        source.to_microns(microns_per_pixel)

    # add a time in seconds column
    df['time (s)'] = df['frame'] * seconds_per_frame

    # calculate the distance from the wound for each point
    df['r (μm)'] = ((df['x (μm)'] - source.x) ** 2 + (df['y (μm)'] - source.y) ** 2) ** 0.5

    # make sure t_bins and r_bins are in  seconds and μm
    if t_units is 'minutes':
        t_bins = 60 * np.array(t_bins)
    if r_units is 'pixels':
        r_bins = microns_per_pixel * np.array(r_bins)

    out = {}
    for t1, t2 in t_bins:
        for r1, r2 in r_bins:

            # mask the dataframe by the time and space conditions
            t_mask = (df['time (s)'] >= t1) & (df['time (s)'] <= t2)
            r_mask = (df['r (μm)'] >= r1) & (df['r (μm)'] <= r2)
            masked_df = df[t_mask & r_mask]

            # get a list of seperate paths that meet these conditions
            paths = [sub_df[['frame', 'x (μm)', 'y (μm)']].values for path, sub_df in masked_df.groupby('path') if len(sub_df) >= min_path_length]

            # add the full matrix to the dictionary
            out[((t1, t2), (r1, r2))] = paths

    return out


class CellTracker:
    """
    This class is used for extracting information from a tif file. It is initialised with a
    single string that locates the tif file. Then, the class will extract the x-y coordinates
    of all the cell tracjectories within that tif file.
    """

    def __init__(self, video_file: str,
                 seconds_per_frame: float=30,
                 microns_per_pixel: float=0.269,
                 color_channel: int=0):
        """
        Initialise a cell tracker object. Pass in the path to your tif file.

        Parameters
        ----------
        video_file          A string with the path to the tif or mp4 file.

        seconds_per_frame   The number of seconds that each frame of the input files lasts.
                            If this is the same for all input files, pass a single float/int.
                            If it is different, specify this with a list of the same length
                            as the input files list.

        microns_per_pixel   Same as above, but specifying the number of microns per pixel in
                            the input files.

        color_channel       The right color channel to use. (0 -> first channel, 1 -> 2nd etc).
                            Again, can be a list if different for different files.
        """



        # read in the tif file
        self.file_path = video_file

        # if we have a tif file ...
        if video_file[-3:] == 'tif':
            self.frames = imread(video_file)[:, color_channel, :, :]
            self.T, self.Y, self.X = self.frames.shape

        # if we have an mp4 file
        if video_file[-3:] == 'mp4':

            import ffmpeg

            probe = ffmpeg.probe(video_file)
            info = [x for x in probe['streams'] if x['codec_type'] == 'video'][0]
            self.T, self.Y, self.X = [int(info[info_type]) for info_type in ['nb_frames', 'height', 'width']]
            out, _ = ffmpeg.input(video_file).output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True)
            self.frames = np.frombuffer(out, np.uint8).reshape((-1, self.Y, self.X, 3))[:, :, :, color_channel]

        # check the unit conversion arguments
        if not isinstance(seconds_per_frame, Number):
            raise TypeError('seconds_per_frame must be a number')
        if not isinstance(microns_per_pixel, Number):
            raise TypeError('microns_per_pixel must be a number')

        self.seconds_per_frame = seconds_per_frame
        self.microns_per_pixel = microns_per_pixel

        # initialise some empty objects
        self.cells = None
        self.paths = None
        self.dataframe = None

    def compute_paths(self,
                      detector: str='LoG',
                      min_sigma: int=3,
                      max_sigma: int=8,
                      num_sigma: int=10,
                      threshold: float=0.05,
                      step_size: float=10,
                      d: float=None,
                      b: float=None):
        """
        Compute all the tracjectories on the given tif files. This method will save a
        few different objects inside the class instance. Each one is a list, where each
        element refers to the respective input file. (If only one input file was provided,
        it is a list containing one element only). The objects are:

        CellTracker.cells:            Each element is a dictionary, indexed by frame number,
                                      holding the x-y-r coordinates of each cell detected in
                                      that frame.

        CellTracker.paths:            Each element is a list, holding the N separate trajectories
                                      found in that file. Each trajectory is an (t, 3) numpy
                                      array, where the first column holds the frame number,
                                      and the other columns are the x-y coordinates of that cell.

        CellTracker.dataframe:       Each element is a pandas DataFrame, with columns ['path',
                                      'frame number', 'time (seconds)', 'x (pixels)', 'y (pixels)',
                                      'x (microns)', 'y (microns)']

        Parameters
        ----------
        detector        'LoG' or 'DoG'. Which detector type to use
        min_sigma       Detector arguments (see detect_cells)
        max_sigma       "
        num_sigma       "
        threshold       "
        step_size       The typical step size for cells measured in pixels. Important to
                        get this roughly right!
        d               The death parameter
        b               The birth parameter

        """

        # keyword arguments to pass to detector function
        detector_kwargs = {'detector': detector,
                           'min_sigma': min_sigma,
                           'max_sigma': max_sigma,
                           'num_sigma': num_sigma,
                           'threshold': threshold}

        # add cells object to class
        self.cells = detect_all_cells(self.frames, **detector_kwargs)

        # keyword arguments for the find_paths and link_paths functions
        path_kwargs = {'box_x': self.X,
                       'box_y': self.Y,
                       'step_size': step_size,
                       'd': d,
                       'b': b}

        # add a paths object to the class
        self.paths = link_paths(find_paths(self.cells, **path_kwargs), **path_kwargs)

        # create a dataframe object and add columns for microns and time in seconds
        cols = ['TRACK_ID', 'FRAME', 'TIME', 'PIXELS_X', 'PIXELS_Y', 'POSITION_X', 'POSITION_Y']
        paths = [np.concatenate([path_no * np.ones((path.shape[0], 1)), path], axis=1) for path_no, path in enumerate(self.paths)]
        self.dataframe = pd.DataFrame(np.concatenate(paths, axis=0), columns=['TRACK_ID', 'FRAME', 'PIXELS_X', 'PIXELS_Y'])
        self.dataframe[['POSITION_X', 'POSITION_Y']] = self.dataframe[['PIXELS_X', 'PIXELS_Y']] * self.microns_per_pixel
        self.dataframe['TIME'] = self.dataframe['FRAME'] * self.seconds_per_frame
        self.dataframe = self.dataframe[cols]

    def segment_paths(self, source: Source,
                      r_bins: tuple = ((0, 25), (25, 50), (50, 100), (100, 150)),
                      t_bins: tuple = ((0, 15), (15, 30), (30, 45), (45, 60), (60, 90), (90, 120)),
                      min_path_length: int = 5,
                      r_units: str = 'microns',
                      t_units: str = 'minutes') -> dict:

        # perform checks on source
        assert isinstance(source, Source)

        # convert all sources to microns
        if source.units is 'pixels':
            source.to_microns(self.microns_per_pixel)

        # the keyword arguments for the segemnt function
        kwargs = {'r_bins': r_bins,
                  't_bins': t_bins,
                  'min_path_length': min_path_length,
                  'path_units': 'pixels',
                  'r_units': r_units,
                  't_units': t_units,
                  'seconds_per_frame': self.seconds_per_frame,
                  'microns_per_pixel': self.microns_per_pixel}

        # a dictionary, indexed by t-s bin
        segments = segment_paths(self.paths, source, **kwargs)

        return segments

    def to_csv(self, save_as: str=None):
        """

        Save the trajectory data to a csv file with a path specified by save_as.
        If save_as is left as None, csv will have the same name as the tif file
        that created the frames and will be saved in the working directory.

        Parameters
        ----------
        save_as       The file path to save the csv file as

        """

        if self.cells is None:
            raise Exception('Run compute_paths before trying to output csv')

        if save_as is None:
            save_as = os.path.split(self.file_path)[-1][:-4] + '.csv'

        if save_as[-4:] != '.csv':
            save_as += '.csv'

        self.dataframe.to_csv(save_as, index=False)

    def make_gif(self, save_as: str=None, show_paths: bool=True, dpi: int=200, delay: int=10):

        if save_as is None:
            save_as = os.path.split(self.file_path)[-1][:-4]

        if show_paths:
            if self.cells is None:
                raise Exception('Run compute_paths before trying to make gif with paths overlayed')

        make_gif(self.frames, save_as=save_as, delay=delay, dpi=dpi, paths=self.paths)


if __name__ == '__main__':


    tif_file = '/Users/linus/Dropbox/projects/cellMigration/Wood/woundMigration/Control wounded 1hr/Pupae 1 concatenated ubi-ecad-GFP, srpGFP; srp-3xH2Amch x white-1HR.tif'

    tracker = CellTracker(tif_file, color_channel=0)
    tracker.compute_paths(detector='DoG')
    tracker.to_csv()
