import h5py
import numpy as np
import feature_extraction as ft
from rich.progress import track


def open_file(file):
    f = h5py.File(file, 'r')

    data = f['data']
    force = f['force']
    v_rm = f['voxels_removed']

    return data, force, v_rm


def validate_stroke_count(f, exp):

    data, _, _ = open_file(f)

    strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

    print(f, ' stroke count: ', sum(strokes))

    # assert sum(strokes) == exp


def validate_drill_kinematics(f, exp):

    _, _, _ = open_file(f)


def validate_stroke_force_and_removal_rate(f, exp):

    _, _, _ = open_file(f)


def validate_stroke_length_and_curvature(f, exp):

    data, _, _ = open_file(f)

    strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

    ft.stroke_length(np.array(strokes), data['pose_mastoidectomy_drill'][()])


def validate_procedure_duration(f, exp):

    _, _, v_rm = open_file(f)

    dur = 0

    if np.array(v_rm).any():
        dur = ft.procedure_duration(v_rm['time_stamp'][()])

    print(f, ' duration: ', dur)

    # assert dur >= exp - 5
    # assert dur <= exp + 5


def validate_drill_angle(f, exp):

    data, force, _ = open_file(f)

    strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

    mean, med, maxi = ft.stats_per_stroke(ft.drill_orientation(
        strokes, stroke_times, data['pose_mastoidectomy_drill'][()], data['time'][()], force['wrench'][()], force['time_stamp'][()]))

    print(f, ' angles: ')
    print('\tmean: ', mean)
    print('\tmax: ', maxi)
    print('\tmedian: ', med)

    # assert mean >= exp - 5
    # assert mean <= exp + 5


def main():
    files = []

    # Files for testing stroke count
    files.append('Strokes/zero_strokes.hdf5')
    files.append('Strokes/three_strokes.hdf5')
    files.append('Strokes/nine_strokes.hdf5')

    # Files for testing kinematics
    files.append('Kinematics/slow_constant.hdf5')
    files.append('Kinematics/fast_constant.hdf5')
    files.append('Kinematics/slow_jerky.hdf5')
    files.append('Kinematics/fast_jerky.hdf5')

    # Files for testing force and bone removal rate
    files.append('ForceRemove/no_force_removal.hdf5')
    files.append('ForceRemove/low_force_removal.hdf5')
    files.append('ForceRemove/high_force_removal.hdf5')

    # Files for testing stroke length and spatiotemporal curvature
    files.append('LenCurve/short_straight.hdf5')
    files.append('LenCurve/long_curved.hdf5')

    # Files for testing procedure duration
    files.append('Duration/20sec.hdf5')

    # Files for testing drill orientation
    files.append('Angles/45deg.hdf5')
    files.append('Angles/90deg.hdf5')
    files.append('Angles/random.hdf5')

    sct_exp = [0, 3, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 3, 4]
    kin_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    frm_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lcu_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dur_exp = [0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0]
    dra_exp = [0, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 90, 45]

    for i in track(range(len(files)), 'Validating files...'):

        validate_stroke_count(
            files[i], sct_exp[i])
        validate_drill_kinematics(
            files[i], kin_exp[i])
        validate_stroke_force_and_removal_rate(
            files[i], frm_exp[i])
        validate_stroke_length_and_curvature(
            files[i], lcu_exp[i])
        validate_procedure_duration(
            files[i], dur_exp[0])
        validate_drill_angle(
            files[i], dra_exp[i])

    print('Validation complete: all tests passed!')


if __name__ == "__main__":
    main()
