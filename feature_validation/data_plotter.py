import numpy as np
import numpy.linalg as la

from data_merger import *
import matplotlib.pyplot as plt
import seaborn as sns

data_merger = DataMerger()
files = []
files.append(['Pete1', '/home/adnan/Downloads/Participant_7/2022-11-11 10:23:45/'])
files.append(['Pete2', '/home/adnan/Downloads/Participant_7/2022-11-11 10:42:18/'])
files.append(['Pete3', '/home/adnan/Downloads/Participant_7/2022-11-11 10:47:51/'])
files.append(['Pete4', '/home/adnan/Downloads/Participant_7/2022-11-11 10:54:32/'])
files.append(['Pete5', '/home/adnan/Downloads/Participant_7/2022-11-11 10:58:18/'])
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