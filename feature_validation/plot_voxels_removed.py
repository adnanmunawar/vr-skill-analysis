from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import h5py
import numpy as np

from data_merger import DataMerger


def rgb_to_hex(r, g, b):
    return '#%02x%02x%02x' % (int(r), int(g), int(b))


data_merger = DataMerger()
data = data_merger.get_merged_data('/home/adnan/Downloads/Participant_7/2022-11-11 10:58:18/')
vrm = data['voxels_removed']['voxel_removed'][()]
vcol = data['voxels_removed']['voxel_color'][()]

colors = [None for _ in range(vcol.shape[0])]

for i, c in enumerate(vcol):
    colors[i] = rgb_to_hex(c[1], c[2], c[3])

params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (9, 5),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}

plt.rcParams.update(params)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(vrm[:, 1], vrm[:, 2], vrm[:, 3], alpha=.3, c=colors)
# ax.scatter([1, 2, 3], [5, 6, 4], [9, 5, 4], label="X")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# plt.legend(["Dura", "Tegmen"])
plt.title('Removed Voxels')
plt.show()