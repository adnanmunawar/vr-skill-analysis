import h5py
import feature_extraction
from rich.progress import track


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
        f = h5py.File(files[i], 'r')

        data = f['data']
        force = f['force']
        v_rm = f['voxels_removed']

        validate_stroke_count()
        validate_drill_kinematics()
        validate_stroke_force_and_removal_rate()
        validate_stroke_length_and_curvature()
        validate_procedure_duration()
        validate_drill_angle()

    print('Validation complete: all tests passed!')


if __name__ == "__main__":
    main()
