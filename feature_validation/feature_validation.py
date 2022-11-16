import os
from statistics import median
import h5py
import natsort
import numpy as np
import feature_extraction as ft
from rich.progress import track
from evaluation_metrics import EvaluationMetrics

eval_metrics = EvaluationMetrics()


def open_file(file):
    f = h5py.File(file, 'r')

    data = f['data']
    v_rm = f['voxels_removed']
    force = []
    if 'drill_force_feedback' in f:
        force = f['drill_force_feedback']
    elif 'force' in f:
        force = f['force']

    return data, force, v_rm


def validate_stroke_count(f):

    data, _, _ = open_file(f)

    try:
        strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])
        print('\tstroke count: ', sum(strokes))
        eval_metrics.strokes.count = eval_metrics.strokes.count + sum(strokes)
    except Exception as e:
        print(e)


def validate_drill_kinematics(f):

    data, _, _ = open_file(f)

    try:
        strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

        inds = ft.get_stroke_indices(strokes)

        velocities, accelerations = ft.extract_kinematics(
            data['pose_mastoidectomy_drill'][()], data['time'][()], inds)

        mean, med, maxi, sdev = ft.stats_per_stroke(velocities)
        print('\tvelocity: ', med)
        eval_metrics.kinematics.velocity.add_mean(mean)

        mean, med, maxi, sdev = ft.stats_per_stroke(accelerations)
        print('\tacceleration: ', med)
        eval_metrics.kinematics.acceleration.add_mean(mean)

        jerks = ft.extract_jerk(
            data['pose_mastoidectomy_drill'][()], data['time'][()], inds)

        mean, med, maxi, sdev = ft.stats_per_stroke(jerks)
        print('\tjerk: ', med)
        eval_metrics.kinematics.jerk.add_mean(mean)
    except Exception as e:
        print(e)


def validate_stroke_force(f):

    data, force, _ = open_file(f)

    try:
        strokes, stroke_times = ft.get_strokes(
            data['pose_mastoidectomy_drill'][()], data['time'][()])

        mean, med, maxi, sdev = ft.stats_per_stroke(ft.stroke_force(
            strokes, stroke_times, force['wrench'][()], force['time_stamp'][()]*1e9))
        eval_metrics.strokes.force.add_mean(mean)

        print('\tstroke force: ', maxi)
    except Exception as e:
        print(e)


def validate_removal_rate(f):

    data, _, v_rm = open_file(f)

    try:
        strokes, stroke_times = ft.get_strokes(
            data['pose_mastoidectomy_drill'][()], data['time'][()])

        mean, med, _, sdev = ft.stats_per_stroke(ft.bone_removal_rate(
            strokes, stroke_times, data['pose_mastoidectomy_drill'][()], v_rm['voxel_time_stamp'][()]))
        eval_metrics.removal_rate.add_mean(mean)
    except Exception as e:
        print(e)
        med = 0

    print('\tbone removal rate: ', med)


def validate_stroke_length(f):

    data, _, _ = open_file(f)

    try:
        strokes, stroke_times = ft.get_strokes(
            data['pose_mastoidectomy_drill'][()], data['time'][()])

        mean, med, maxi, sdev = ft.stats_per_stroke(ft.stroke_length(
            np.array(strokes), data['pose_mastoidectomy_drill'][()]))
        eval_metrics.strokes.length.add_mean(mean)
    except Exception as e:
        print(e)
        med = 0

    print('\tstroke length: ', med)


def validate_curvature(f):

    data, _, _ = open_file(f)

    try:
        strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

        inds = ft.get_stroke_indices(strokes)

        mean, med, maxi, sdev = ft.stats_per_stroke(ft.extract_curvature(
            data['pose_mastoidectomy_drill'][()], data['time'][()], inds))
        eval_metrics.strokes.curvature.add_mean(mean)
    except Exception as e:
        print(e)
        med = 0

    print('\tcurvature: ', med)


def validate_procedure_duration(f):
    _, _, v_rm = open_file(f)

    try:
        dur = ft.procedure_duration(v_rm['voxel_time_stamp'][()])
        eval_metrics.duration = eval_metrics.duration + dur
    except:
        dur = 0

    print('\tduration: ', dur)


def validate_drill_angle(f):

    data, force, _ = open_file(f)

    try:
        strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

        mean, med, maxi, sdev = ft.stats_per_stroke(ft.drill_orientation(
            strokes, stroke_times, data['pose_mastoidectomy_drill'][()], data['time'][()], force['wrench'][()], force['time_stamp'][()]*1e9))

        print('\tangles:')
        print('\t\tmean: ', mean)
        print('\t\tmedian: ', med)
        print('\t\tmax: ', maxi)

    except Exception as e:
        print(e)


def validate_sensitive_voxels_removed(f):
    _, _, vrm = open_file(f)

    try:
        cnt = eval_metrics.check_voxels_removed(vrm['voxel_color'][()])
        print('\t\tsensitive voxels removed: ', cnt)
    except Exception as e:
        print(e)

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

    for i in track(range(len(files)), 'Validating files...'):

        print('\nNow testing: ', files[i])

        validate_stroke_count(files[i])
        validate_drill_kinematics(files[i])
        validate_stroke_force(files[i])
        validate_removal_rate(files[i])
        validate_stroke_length(files[i])
        validate_curvature(files[i])
        validate_procedure_duration(files[i])
        validate_drill_angle(files[i])

    print('Validation complete!')
    eval_metrics.print()


if __name__ == "__main__":
    main()
