from rich.progress import track
import feature_extraction


def main():
    files = []

    files.append('Angles/45deg.hdf5')  # 45 degree drill angle
    files.append('Angles/90deg.hdf5')  # 90 degree drill angle
    files.append('Angles/random.hdf5')  # random drill angle


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
