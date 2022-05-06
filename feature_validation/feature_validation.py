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

    if not (sum(strokes) != exp):
        print(f, ' stroke count test failed!')
        print('\tgot: ', sum(strokes))
        print('\texpected: ', exp)


def validate_drill_kinematics(f, exp):

    data, _, _ = open_file(f)

    strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

    inds = ft.get_stroke_indices(strokes)

    velocities, accelerations = ft.extract_kinematics(
        data['pose_mastoidectomy_drill'][()], data['time'][()], inds)

    mean, med, maxi = ft.stats_per_stroke(velocities)

    print(f, ' velocity:')
    print('\tgot: ', med)
    print('\texpected: ', exp)

    mean, med, maxi = ft.stats_per_stroke(accelerations)

    print(f, ' acceleration:')
    print('\tgot: ', med)
    print('\texpected: ', exp)

    jerks = ft.extract_jerk(
        data['pose_mastoidectomy_drill'][()], data['time'][()], inds)

    mean, med, maxi = ft.stats_per_stroke(jerks)

    print(f, ' jerk:')
    print('\tgot: ', med)
    print('\texpected: ', exp)


def validate_stroke_force(f, exp):

    data, force, _ = open_file(f)

    strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

    mean, med, maxi = ft.stats_per_stroke(ft.stroke_force(
        strokes, stroke_times, force['wrench'][()], force['time_stamp'][()]))

    print(f, ' stroke force:')
    print('\tgot: ', med)
    print('\texpected: ', exp)


def validate_removal_rate(f, exp):

    data, _, v_rm = open_file(f)

    strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

    try:
        _, med, _ = ft.stats_per_stroke(ft.bone_removal_rate(
            strokes, stroke_times, data['pose_mastoidectomy_drill'][()], v_rm['time_stamp'][()]))
    except:
        med = 0

    print(f, ' bone removal rate:')
    print('\tgot: ', med)
    print('\texpected: ', exp)


def validate_stroke_length(f, exp):

    data, _, _ = open_file(f)

    strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

    mean, med, maxi = ft.stats_per_stroke(ft.stroke_length(
        np.array(strokes), data['pose_mastoidectomy_drill'][()]))

    print(f, ' stroke length:')
    print('\tgot: ', med)
    print('\texpected: ', exp)


def validate_curvature(f, exp):

    data, _, _ = open_file(f)

    strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

    inds = ft.get_stroke_indices(strokes)

    mean, med, maxi = ft.stats_per_stroke(ft.extract_curvature(
        data['pose_mastoidectomy_drill'][()], data['time'][()], inds))

    print(f, ' curvature:')
    print('\tgot: ', med)
    print('\texpected: ', exp)


def validate_procedure_duration(f, exp):

    _, _, v_rm = open_file(f)

    try:
        dur = ft.procedure_duration(v_rm['time_stamp'][()])
    except:
        dur = 0

    if not ((dur >= exp - 5) and (dur <= exp + 5)):
        print(f, ' duration test failed!')
        print('\tgot: ', dur)
        print('\texpected: ', exp)


def validate_drill_angle(f, exp):

    data, force, _ = open_file(f)

    strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

    mean, med, maxi = ft.stats_per_stroke(ft.drill_orientation(
        strokes, stroke_times, data['pose_mastoidectomy_drill'][()], data['time'][()], force['wrench'][()], force['time_stamp'][()]))

    if not ((med >= exp - 5) and (med <= exp + 5)):
        print(f, ' angle test failed!')
        print('\tgot: ', med)
        print('\texpected: ', exp)


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

    # TODO: populate expected values for each test
    sct_exp = [0, 3, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 3, 4]
    kin_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    frc_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rmv_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    len_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cur_exp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dur_exp = [0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0]
    dra_exp = [0, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 90, 45]

    for i in track(range(len(files)), 'Validating files...'):

        validate_stroke_count(
            files[i], sct_exp[i])
        validate_drill_kinematics(
            files[i], kin_exp[i])
        validate_stroke_force(
            files[i], frc_exp[i])
        validate_removal_rate(
            files[i], rmv_exp[i])
        validate_stroke_length(
            files[i], len_exp[i])
        validate_curvature(
            files[i], cur_exp[i])
        validate_procedure_duration(
            files[i], dur_exp[0])
        validate_drill_angle(
            files[i], dra_exp[i])

    print('Validation complete!')


if __name__ == "__main__":
    main()
