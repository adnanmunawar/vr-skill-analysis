import h5py
import numpy as np
import math

from scipy.spatial.transform import Rotation as R
from scipy import signal
from scipy import fft
from scipy import integrate

# fname = '../data/d4/20220302_131424.hdf5'
# fname = '../data/20220408_191041.hdf5'
# fname = '../data/20220408_210049.hdf5' # 3 strokes
# fname = '../data/20220409_184944.hdf5' # perpendicular drill angle
# fname = '../data/20220409_185037.hdf5' # varying drill angle

# fname = '../Angles/45deg.hdf5' # 45 degree drill angle
fname = '../Angles/90deg.hdf5'  # 90 degree drill angle
# fname = '../Angles/random.hdf5' # random drill angle

f = h5py.File(fname, 'r')
print('File keys', f.keys())
print(type(f))


def stats_per_stroke(stroke_arr: np.ndarray):

    mean = np.mean(stroke_arr)
    med_ = np.median(stroke_arr)
    max_ = np.max(stroke_arr)

    return mean, med_, max_


def get_strokes(stream: np.ndarray, timepts: np.ndarray, k=6):
    stream = stream[:, :3]
    X_P = []

    # Compute k-cosines for each pivot point
    for j, P in enumerate(stream):

        # Cannot consider edge points as central k's
        if (j - k < 0) or (j + k >= stream.shape[0]):
            continue

        P_a = stream[j - k]
        P_c = stream[j + k]

        k_cos = np.dot(P_a, P_c) / (np.linalg.norm(P_a) * np.linalg.norm(P_c))
        k_cos = max(min(k_cos, 1), -1)
        X_P.append(180 - (math.acos(k_cos) * (180/np.pi)))

    # Detect pivot points
    mu = np.mean(X_P)
    sig = np.std(X_P)

    for i in range(k):

        X_P.insert(0, mu)
        X_P.append(mu)

    X_P = np.array(X_P)

    F_c = [1 if x_P > mu + sig else 0 for x_P in X_P]

    j = 0
    for i in range(1, len(F_c)):

        if F_c[i] == 1 and F_c[i-1] == 0:
            j += 1
        elif sum(F_c[i:i+k]) == 0 and j != 0:
            ind = math.floor(j/2)
            F_c[i-j:i] = [0] * j
            F_c[i-ind] = 1
            j = 0
        elif j != 0:
            j += 1

    st = np.insert(timepts[[s == 1 for s in F_c]], 0, min(timepts))

    return F_c, st


def stroke_force(strokes: np.ndarray, stroke_times: np.ndarray,
                 force_stream: np.ndarray, force_times: np.ndarray):

    avg_stroke_force = []
    for i in range(sum(strokes)):

        stroke_mask = [ft >= stroke_times[i] and ft <
                       stroke_times[i+1] for ft in force_times]
        stroke_forces = np.linalg.norm(force_stream[stroke_mask], axis=1)
        avg_stroke_force.append(np.mean(stroke_forces))

    return np.array(avg_stroke_force)


def stroke_length(strokes: np.ndarray, stream: np.ndarray):

    stream = stream[:, :3]

    lens = []
    inds = np.insert(np.where(strokes == 1), 0, 0)
    for i in range(sum(strokes)):
        stroke_len = 0
        curr_stroke = stream[inds[i]:inds[i+1]]
        for j in range(1, len(curr_stroke)):

            stroke_len += np.linalg.norm(curr_stroke[j-1] - curr_stroke[j])

        lens.append(stroke_len)

    return np.array(lens)


def bone_removal_rate(strokes: np.ndarray, stroke_times: np.ndarray,
                      stream: np.ndarray, voxel_times: np.ndarray):

    vox_rm = []
    for i in range(sum(strokes)):

        stroke_voxels = [vt >= stroke_times[i] and vt <
                         stroke_times[i+1] for vt in voxel_times]
        vox_rm.append(sum(stroke_voxels))

    rate = np.divide(np.array(vox_rm), stroke_length(
        np.array(strokes), stream))

    return rate


def procedure_duration(timepts: np.ndarray):
    return (max(timepts) - min(timepts))


def drill_orientation(stream: np.ndarray, timepts: np.ndarray,
                      force_stream: np.ndarray, force_times: np.ndarray):

    angles = []
    normals = []
    drill_vecs = []

    forces = force_stream[np.linalg.norm(force_stream, axis=1) > 0]
    med = np.median(np.linalg.norm(forces, axis=1))

    for i, t in enumerate(timepts):
        ind = np.argmin(np.abs(force_times - t))
        if np.isclose(np.abs(force_times[ind] - t), 0) and np.linalg.norm(force_stream[ind]) > med:

            normal = force_stream[ind]
            normal = np.divide(normal, np.linalg.norm(normal))
            normals.append(normal)

            v = R.from_quat(stream[i, 3:]).apply([-1, 0, 0])
            v = np.divide(v, np.linalg.norm(v))
            drill_vecs.append(v)

    for i, v in enumerate(drill_vecs):

        n = normals[i]

        angle = (np.arccos(np.clip(np.dot(n, v), -1.0, 1.0)) * (180/np.pi))
        if angle > 90:
            angle = 180 - angle

        angles.append(90 - angle)

    return angles


def get_stroke_indices(stroke_cutoffs):
    '''
    Returns a list of timestamp indices indicating the beginning of a new stroke

    Parameters:
        stroke_cutoffs (list): List of 1's and 0's indicating whether a stroke has ended at the timestamp at its index

    Returns:
        indices (list): List of integer indices naming the indices at which a new stroke is initiated
    '''

    # Find  all indices of stroke_cutoff where the value is a 1
    indices = (np.where(stroke_cutoffs == 1))[0]
    # 0 index is always the start of a stroke
    indices.insert(0, 0)

    return indices


def extract_kinematics(drill_pose, timestamps, stroke_indices):
    '''
    Returns mean, median, and max velocity values across all strokes from drill pose data

    Parameters:
        drill_pose (list): Drill pose data directly extracted from hdf5 file
        timestamps (list): Timestamp data directly extracted from hdf5 file
        stroke_indices (list): List of integer indices naming indices of timestamps at which a new stroke is initiated

    Returns: 
        velocities (list): List containing mean, median, and max velocity
        accelerations (list): List containing mean, median, and max acceleration
    '''

    # Extract x, y, and z data from drill pose data
    x = [i[0] for i in drill_pose]
    y = [i[1] for i in drill_pose]
    z = [i[2] for i in drill_pose]

    stroke_vx = []
    stroke_vy = []
    stroke_vz = []
    stroke_t = []
    stroke_velocities = []

    # Store velocity information for acceleration and calculate average velocity
    for i in range(len(stroke_indices) - 1):
        stroke_start = stroke_indices[i]
        next_stroke = stroke_indices[i+1]

        # Split up positional data for each stroke
        stroke_x = x[stroke_start:next_stroke]
        stroke_y = y[stroke_start:next_stroke]
        stroke_z = z[stroke_start:next_stroke]
        t = timestamps[stroke_start:next_stroke]

        # Calculate numerical derivatives between successive timestamps
        stroke_vx.append(np.gradient(stroke_x, t))
        stroke_vy.append(np.gradient(stroke_y, t))
        stroke_vz.append(np.gradient(stroke_z, t))

        # Calculate distance traveled during stroke and use to calculate velocity
        curr_stroke = [[stroke_x[k], stroke_y[k], stroke_z[k]]
                       for k in range(len(stroke_x))]
        dist = 0
        for l in range(1, len(curr_stroke)):
            dist += np.linalg.norm(curr_stroke[l] - curr_stroke[l - 1])
        stroke_velocities.append(dist / np.ptp(t))

    # Calculate average acceleration using velocity information
    stroke_accelerations = []
    for i in range(len(stroke_vx)):
        curr_stroke = [[stroke_vx[i][j], stroke_vy[i][j], stroke_vz[i][j]]
                       for j in range(len(stroke_vx[i]))]
        vel = 0
        for k in range(1, len(curr_stroke)):
            vel += np.linalg.norm(curr_stroke[k] - curr_stroke[k - 1])
            stroke_accelerations.append(vel / np.ptp(stroke_t[i]))

    return stroke_velocities, stroke_accelerations


def preprocess(drill_pose):
    '''
    Applies second order Butterworth filter and FFT for curve smoothing and pattern repetition isolation

    Parameters:
        drill_pose(list): Drill pose data directly extracted from hdf5 file

    Returns:
        x (list): Preprocessed x position data
        y (list): Preprocessed y position data
        z (list): Preprocessed z position data
    '''
    # Extract x, y, and z data from drill pose data
    x_raw = [i[0] for i in drill_pose]
    y_raw = [i[1] for i in drill_pose]
    z_raw = [i[2] for i in drill_pose]

    # Smooth data using a second order Butterworth filter
    b, a = signal.butter(2, [0.005, 1], 'bandpass', analog=False)
    x_filt = signal.filtfilt(b, a, x_raw)
    y_filt = signal.filtfilt(b, a, y_raw)
    z_filt = signal.filtfilt(b, a, z_raw)

    # Apply a FFT to data
    x = fft(x_filt)
    y = fft(y_filt)
    z = fft(z_filt)

    return x, y, z


def extract_jerk(drill_pose, timestamps, stroke_indices):
    '''
    Returns mean, median, and max jerk across all strokes from drill pose data

    Parameters:
        drill_pose (list): Drill pose data directly extracted from hdf5 file
        timestamps (list): Timestamp data directly extracted from hdf5 file
        stroke_indices (list): List of integer indices naming indices of timestamps at which a new stroke is initiated

    Returns: 
        jerks (list): List containing mean, median, and max jerk
    '''

    # Get preprocessed, x, y, and z position data
    x, y, z = preprocess(drill_pose)

    stroke_vx = []
    stroke_vy = []
    stroke_vz = []
    stroke_t = []

    # Store velocity information for acceleration
    for i in range(len(stroke_indices) - 1):
        stroke_start = stroke_indices[i]
        next_stroke = stroke_indices[i+1]

        # Split up positional data for each stroke
        stroke_x = x[stroke_start:next_stroke]
        stroke_y = y[stroke_start:next_stroke]
        stroke_z = z[stroke_start:next_stroke]
        t = timestamps[stroke_start:next_stroke]

        stroke_vx.append(np.gradient(stroke_x, t))
        stroke_vy.append(np.gradient(stroke_y, t))
        stroke_vz.append(np.gradient(stroke_z, t))

    stroke_ax = []
    stroke_ay = []
    stroke_az = []

    stroke_accelerations = []
    # Store acceleration information for jerk
    for i in range(len(stroke_vx)):

        # Calculate numerical derivatives between successive timestamps
        stroke_ax.append(np.gradient(stroke_vx[i], t))
        stroke_ay.append(np.gradient(stroke_vy[i], t))
        stroke_az.append(np.gradeint(stroke_vz[i], t))

    # Calculate average jerk using acceleration information
    stroke_jerks = []
    for i in range(len(stroke_ax)):
        curr_stroke = [[stroke_ax[i][j], stroke_ay[i][j], stroke_az[i][j]]
                       for j in range(len(stroke_ax[i]))]
        acc = 0
        for k in range(1, len(curr_stroke)):
            acc += np.linalg.norm(curr_stroke[k] - curr_stroke[k - 1])
            stroke_jerks.append(acc / np.ptp(stroke_t[i]))

    return stroke_jerks


def extract_curvature(drill_pose, timestamps, stroke_indices):
    '''
    Returns mean, median, and max spaciotemporal curvatures across all strokes from drill pose data

    Parameters:
        drill_pose (list): Drill pose data directly extracted from hdf5 file
        timestamps (list): Timestamp data directly extracted from hdf5 file
        stroke_indices (list): List of integer indices naming indices of timestamps at which a new stroke is initiated

    Returns: 
        curvatures (list): List containing mean, median, and max spaciotemporal curvature
    '''

    # Get preprocessed, x, y, and z position data
    x, y, z = preprocess(drill_pose)

    curvatures = []
    stroke_curvatures = []
    for i in range(len(stroke_indices) - 1):
        stroke_start = stroke_indices[i]
        next_stroke = stroke_indices[i + 1]

        # Split up positional data for each stroke
        stroke_x = x[stroke_start:next_stroke]
        stroke_y = y[stroke_start:next_stroke]
        stroke_z = z[stroke_start:next_stroke]
        stroke_t = timestamps[stroke_start:next_stroke]

        # Calculate velocity and acceleration for each stroke
        stroke_vx = np.gradient(stroke_x, stroke_t)
        stroke_vy = np.gradient(stroke_y, stroke_t)
        stroke_vz = np.gradient(stroke_z, stroke_t)
        stroke_ax = np.gradient(stroke_vx, stroke_t)
        stroke_ay = np.gradient(stroke_vy, stroke_t)
        stroke_az = np.gradient(stroke_vz, stroke_t)

        curvature = []
        for j in range(len(stroke_vx)):

            # Calculate r' and r'' at specific time point
            r_prime = [stroke_vx[j], stroke_vy[j], stroke_vz[j]]
            r_dprime = [stroke_ax[j], stroke_ay[j], stroke_az[j]]
            k = np.linalg.norm(np.cross(r_prime, r_dprime)) / \
                ((np.linalg.norm(r_prime)) ** 3)
            curvature.append(k)

        # Average value of function over an interval is integral of function divided by length of interval
        stroke_curvatures.append(integrate.simpson(
            curvature, stroke_t) / np.ptp(stroke_t))

    return stroke_curvatures
