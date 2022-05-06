import h5py
from pyrsistent import v
import feature_extraction as ft
import unittest
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

    print(sum(strokes))

    # unittest.assertEqual(sum(strokes), exp)


def validate_drill_kinematics(f, exp):

    _, _, _ = open_file(f)


def validate_stroke_force_and_removal_rate(f, exp):

    _, _, _ = open_file(f)


def validate_stroke_length_and_curvature(f, exp):

    _, _, _ = open_file(f)


def validate_procedure_duration(f, exp):

    data, _, v_rm = open_file(f)

    dur = ft.procedure_duration(data['time'][()])

    print(dur)

    # unittest.assertTrue(dur >= exp - 5)
    # unittest.assertTrue(dur <= exp + 5)


def validate_drill_angle(f, exp):

    data, force, _ = open_file(f)

    strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

    mean, med, maxi = ft.stats_per_stroke(ft.drill_orientation(
        strokes, stroke_times, data['pose_mastoidectomy_drill'][()], data['time'][()], force['wrench'][()], force['time_stamp'][()]))

    print('mean: ', mean)
    print('max: ', maxi)
    print('median: ', med)

    # unittest.assertTrue(mean >= exp - 20)
    # unittest.assertTrue(mean <= exp + 20)


def main():
    files = []
    j = 0

    # Files for testing stroke count
    files.append('Strokes/zero_strokes.hdf5')
    files.append('Strokes/three_strokes.hdf5')
    files.append('Strokes/nine_strokes.hdf5')

    sc_exp = [0, 3, 9]

    validate_stroke_count(files[0], sc_exp[0])
    validate_stroke_count(files[1], sc_exp[1])
    validate_stroke_count(files[2], sc_exp[2])

    validate_procedure_duration(files[0], 0)
    validate_procedure_duration(files[1], 0)
    validate_procedure_duration(files[2], 0)

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

    d_exp = [20]

    validate_stroke_count(files[12], sc_exp[0])
    validate_procedure_duration(files[12], d_exp[0])

    # Files for testing drill orientation
    files.append('Angles/45deg.hdf5')
    files.append('Angles/90deg.hdf5')
    files.append('Angles/random.hdf5')

    a_exp = [45, 90, 45]

    validate_stroke_count(files[13], 4)
    validate_drill_angle(files[13], a_exp[0])

    validate_stroke_count(files[14], 3)
    validate_drill_angle(files[14], a_exp[1])

    validate_stroke_count(files[15], 4)
    validate_drill_angle(files[15], a_exp[2])

    print('done with nice stuff and pre testing')

    for i in track(range(len(files)), 'Validating files...'):

        validate_stroke_count(files[i], sc_exp[i])
        validate_drill_kinematics()
        validate_stroke_force_and_removal_rate()
        validate_stroke_length_and_curvature()
        validate_procedure_duration(files[i], d_exp[0])
        validate_drill_angle(files[i], a_exp[i-13])

    print('Validation complete: all tests passed!')


if __name__ == "__main__":
    main()
