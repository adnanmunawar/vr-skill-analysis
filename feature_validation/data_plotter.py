import numpy as np
import numpy.linalg as la

from data_merger import *
import matplotlib.pyplot as plt

data_merger = DataMerger()
files = []
files.append(['P4', '/home/amunawa2/RedCap/Guidance/Participant_4/2022-11-09 19:26:43'])
# files.append(['P5', '/home/amunawa2/RedCap/Guidance/Participant_5/2022-11-10 13:14:55'])
files.append(['P5', '/home/amunawa2/RedCap/Guidance/Participant_5/2022-11-10 12:54:16'])
files.append(['P6', '/home/amunawa2/RedCap/Guidance/Participant_6/2022-11-10 18:16:42'])
files.append(['P7', '/home/amunawa2/RedCap/Guidance/Participant_7/2022-11-11 10:42:18'])
mags = []

x_labels = []
for lab, fl in files:
    data = data_merger.get_merged_data(fl)

    try:
        force_data = data['drill_force_feedback']['wrench'][:, :3]

        m = np.zeros([force_data.shape[0]])

        for i, force_data in enumerate(force_data):
            m[i] = la.norm(force_data)
        mags.append(m)
        x_labels.append(lab)
    except Exception as e:
        print(e)

params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (9, 5),
    "axes.labelsize": "xx-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "xx-large",
    "ytick.labelsize": "xx-large",
}

plt.rcParams.update(params)

fig = plt.figure()
# sns.boxplot(x="Forces", y="Magnitude", data=mags, showfliers=False)
ax = fig.add_subplot(111)
ax.boxplot(mags, showfliers=False)
plt.xticks(np.arange(1, len(x_labels)+1), x_labels, rotation=30)
# ax.set_xlabel(x_labels)
ax.set_ylabel('Force Magnitude(N)')
plt.grid(True)
plt.show()