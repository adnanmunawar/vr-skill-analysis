import numpy as np
import h5py
from scipy import signal
from scipy import fft
from scipy import integrate
import math
import argparse


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
        X_P.append(180 - (math.acos(k_cos) * (180 / np.pi)))

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

        if F_c[i] == 1 and F_c[i - 1] == 0:
            j += 1
        elif sum(F_c[i:i + k]) == 0 and j != 0:
            ind = math.floor(j / 2)
            F_c[i - j:i] = [0] * j
            F_c[i - ind] = 1
            j = 0
        elif j != 0:
            j += 1

    st = np.insert(timepts[[s == 1 for s in F_c]], 0, min(timepts))

    return F_c, st

def stroke_force(strokes: np.ndarray, stroke_times: np.ndarray,
                force_stream: np.ndarray, force_times: np.ndarray):

    avg_stroke_force = []
    for i in range(sum(strokes)):

        stroke_mask = [ft >= stroke_times[i] and ft < stroke_times[i+1] for ft in force_times]
        stroke_forces = np.linalg.norm(force_stream[stroke_mask], axis=1)
        avg_stroke_force.append(np.mean(stroke_forces))

    return np.array(avg_stroke_force)

def get_stroke_indices(stroke_cutoffs):
    '''
    Returns a list of timestamp indices indicating the beginning of a new stroke

      Parameters:
          stroke_cutoffs (list): List of 1's and 0's indicating whether a stroke has ended at the timestamp at its index

      Returns:
          indices (list): List of integer indices naming the indices at which a new stroke is initiated
    '''
    # Find  all indices of stroke_cutoff where the value is a 1
    #indices = (np.where(stroke_cutoffs == 1))[0]
    indices = []
    indices.append(0)
    for i in range(len(stroke_cutoffs)):
        if stroke_cutoffs[i] == 1:
            indices.append(i)
    # 0 index is always the start of a stroke
    #stroke_indices = np.insert(indices,0, 0)
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
    for i in range(len(stroke_indices)):
        stroke_start = stroke_indices[i]
        next_stroke = len(timestamps)
        if i != len(stroke_indices) - 1:
            next_stroke = stroke_indices[i + 1]
        # Split up positional data for each stroke
        stroke_x = x[stroke_start:next_stroke]
        stroke_y = y[stroke_start:next_stroke]
        stroke_z = z[stroke_start:next_stroke]
        t = timestamps[stroke_start:next_stroke]
        # Calculate numerical derivatives between successive timestamps
        # v_x = np.diff(stroke_x) / np.diff(stroke_t)
        # v_y = np.diff(stroke_y) / np.diff(stroke_t)
        # v_z = np.diff(stroke_z) / np.diff(stroke_t)
        v_x = np.gradient(stroke_x, t)
        v_y = np.gradient(stroke_y, t)
        v_z = np.gradient(stroke_z, t)
        stroke_vx.append(v_x)
        stroke_vy.append(v_y)
        stroke_vz.append(v_z)
        stroke_t.append(t)
        # Calculate times at which derivatives are calculated
        # v_t = [(stroke_t[j + 1] + stroke_t[j]) / 2 for j in range(len(v_x))]
        # stroke_vt.append(v_t)
        # Calculate distance traveled during stroke and use to calculate velocity
        curr_stroke = [[stroke_x[k], stroke_y[k], stroke_z[k]] for k in range(len(stroke_x))]
        dist = 0
        for l in range(1, len(curr_stroke)):
            dist += np.linalg.norm(np.subtract(curr_stroke[l],curr_stroke[l - 1]))
        stroke_velocities.append(dist / np.ptp(t))
    stroke_accelerations = []
    # Calculate average acceleration using velocity information
    for i in range(len(stroke_vx)):
        curr_stroke = [[stroke_vx[i][j], stroke_vy[i][j], stroke_vz[i][j]] for j in range(len(stroke_vx[i]))]
        vel = 0
        for k in range(1, len(curr_stroke)):
            vel += np.linalg.norm(np.subtract(curr_stroke[k], curr_stroke[k - 1]))
        stroke_accelerations.append(vel / np.ptp(stroke_t[i]))
    # Return mean, median, and max kinematic values
    velocities = []
    accelerations = []
    velocities.append(np.mean(stroke_velocities))
    velocities.append(np.median(stroke_velocities))
    velocities.append(np.max(stroke_velocities))
    accelerations.append(np.mean(stroke_accelerations))
    accelerations.append(np.median(stroke_accelerations))
    accelerations.append(np.max(stroke_accelerations))
    return velocities, accelerations


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
    x = [i[0] for i in drill_pose]
    y = [i[1] for i in drill_pose]
    z = [i[2] for i in drill_pose]

    stroke_vx = []
    stroke_vy = []
    stroke_vz = []
    stroke_t = []
    # Store velocity information for acceleration
    for i in range(len(stroke_indices)):
        stroke_start = stroke_indices[i]
        next_stroke = len(timestamps)
        if i != len(stroke_indices) - 1:
            next_stroke = stroke_indices[i + 1]
        # Split up positional data for each stroke
        stroke_x = x[stroke_start:next_stroke]
        stroke_y = y[stroke_start:next_stroke]
        stroke_z = z[stroke_start:next_stroke]
        t = timestamps[stroke_start:next_stroke]
        # Calculate numerical derivatives between successive timestamps
        # v_x = np.diff(stroke_x) / np.diff(stroke_t)
        # v_y = np.diff(stroke_y) / np.diff(stroke_t)
        # v_z = np.diff(stroke_z) / np.diff(stroke_t)
        v_x = np.gradient(stroke_x, t)
        v_y = np.gradient(stroke_y, t)
        v_z = np.gradient(stroke_z, t)
        stroke_vx.append(v_x)
        stroke_vy.append(v_y)
        stroke_vz.append(v_z)
        stroke_t.append(t)
        # Calculate times at which derivatives are calculated
        # v_t = [(stroke_t[j + 1] + stroke_t[j]) / 2 for j in range(len(v_x))]
        # stroke_vt.append(v_t)

    stroke_ax = []
    stroke_ay = []
    stroke_az = []
    # stroke_at = []
    stroke_accelerations = []
    # Store acceleration information for jerk
    for i in range(len(stroke_vx)):
        # Calculate numerical derivatives between successive timestamps
        # s_ax = np.diff(stroke_vx) / np.diff(stroke_vt)
        # s_ay = np.diff(stroke_vy) / np.diff(stroke_vt)
        # s_az = np.diff(stroke_vz) / np.diff(stroke_vt)
        s_ax = np.gradient(stroke_vx[i], stroke_t[i])
        s_ay = np.gradient(stroke_vy[i], stroke_t[i])
        s_az = np.gradient(stroke_vz[i], stroke_t[i])
        stroke_ax.append(s_ax)
        stroke_ay.append(s_ay)
        stroke_az.append(s_az)
        # Calculate times at which derivatives are calculated
        # a_t = [(stroke_vt[j + 1] + stroke_vt[j] / 2 for j in range(len(s_ax)))]
        # stroke_at.append(a_t)

    stroke_jerks = []
    # Calculate average jerk using acceleration information
    for i in range(len(stroke_ax)):
        curr_stroke = [[stroke_ax[i][j], stroke_ay[i][j], stroke_az[i][j]] for j in range(len(stroke_ax[i]))]
        acc = 0
        for k in range(1, len(curr_stroke)):
            acc += np.linalg.norm(np.subtract(curr_stroke[k], curr_stroke[k - 1]))
        stroke_jerks.append(acc / np.ptp(stroke_t[i]))

    # Return mean, median, and max jerk values
    jerks = []
    jerks.append(np.mean(stroke_jerks))
    jerks.append(np.median(stroke_jerks))
    jerks.append(np.max(stroke_jerks))

    return jerks


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
    x = [i[0] for i in drill_pose]
    y = [i[1] for i in drill_pose]
    z = [i[2] for i in drill_pose]

    curvatures = []
    stroke_curvatures = []
    for i in range(len(stroke_indices)):
        stroke_start = stroke_indices[i]
        next_stroke = len(timestamps)
        if i != len(stroke_indices) - 1:
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
        stroke_t_copy = [t for t in stroke_t]
        for j in range(len(stroke_vx)):
            # Calculate r' and r'' at specific time point
            r_prime = [stroke_vx[j], stroke_vy[j], stroke_vz[j]]
            r_dprime = [stroke_ax[j], stroke_ay[j], stroke_az[j]]
            # Potentially remove
            if np.linalg.norm(r_prime) == 0:
                stroke_t_copy.pop(j)
                continue
            k = np.linalg.norm(np.cross(r_prime, r_dprime)) / ((np.linalg.norm(r_prime)) ** 3)
            curvature.append(k)
        # Average value of function over an interval is integral of function divided by length of interval
        stroke_curvatures.append(integrate.simpson(curvature, stroke_t_copy) / np.ptp(stroke_t_copy))

    # Return mean, median, and max curvature values
    curvatures.append(np.mean(stroke_curvatures))
    curvatures.append(np.median(stroke_curvatures))
    curvatures.append(np.max(stroke_curvatures))

    return curvatures

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', dest='infile')
    args = parser.parse_args()
    # Read in hdf5 file and get keys
    f = h5py.File(args.infile, 'r')
    data = f['data']
    force = f['force']
    drill_pose = data['pose_mastoidectomy_drill'][()]
    timestamps = data['time'][()]
    strokes, stroke_cutoffs = get_strokes(drill_pose, timestamps)
    print(strokes)
    #print(stroke_cutoffs)
    stroke_indices = get_stroke_indices(strokes)
    print(stroke_indices)
    velocities, accelerations = extract_kinematics(drill_pose, timestamps, stroke_indices)
    jerks = extract_jerk(drill_pose, timestamps,stroke_indices)
    curvatures = extract_curvature(drill_pose, timestamps, stroke_indices)
    #print(velocities)
    #print(accelerations)
    #print(jerks)
    print(curvatures)

if __name__ == '__main__':
    main()

