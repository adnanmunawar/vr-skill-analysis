import os
import h5py
import numpy as np
from natsort import natsorted
from collections import OrderedDict


class DataMerger:
    def __init__(self):
        self._data = OrderedDict()
        self.file_names = []

    def _clear_data(self):
        for g in self._data.keys():
            self._data[g].clear()

        self.file_names = []

    def get_merged_data(self, dir, verbose=False):
        self._clear_data()

        os.chdir(dir)
        names = os.listdir(dir)

        for n in names:
            if n.endswith('.hdf5'):
                self.file_names.append(n)

        self.file_names = natsorted(self.file_names)
        print('Number of Files ', len(self.file_names))

        for idx, file_name in enumerate(self.file_names):
            file = h5py.File(file_name, 'r')
            if verbose: print(idx, 'Opening', file_name)
            for grp in file.keys():
                if grp == 'metadata':
                    continue

                if grp not in self._data:
                    self._data[grp] = OrderedDict()
                if verbose: print('\t Processing Group ', grp)
                for dset in file[grp].keys():
                    if grp == 'data' and dset != 'time' and 'pose_' not in dset:
                        continue

                    if len(file[grp][dset]) == 0:
                        continue

                    if verbose: print('\t\t Processing Dataset ', dset)
                    if dset not in self._data[grp]:
                        self._data[grp][dset] = file[grp][dset][()]
                    else:
                        self._data[grp][dset] = np.append(self._data[grp][dset], file[grp][dset][()], axis=0)
            file.close()
        return self._data


def main():
    data_merge = DataMerger()
    data = data_merge.get_merged_data('/home/adnan/Downloads/Participant_7/three/')

    output_file = h5py.File('output.hdf5', 'w')
    for grp in data.keys():
        output_grp = output_file.create_group(grp)
        for dset in data[grp].keys():
            print('Writing Dataset', dset)
            output_grp.create_dataset(dset, data=data[grp][dset], compression='gzip')

    output_file.close()


if __name__ == "__main__":
    main()

