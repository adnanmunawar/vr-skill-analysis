import math


class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count = self.count + 1


class Stats:
    def __init__(self):
        self.mean = 0
        self.median = 0
        self.max = 0
        self.std_dev = 0
        self.counter = Counter()

    def add_mean(self, val):
        if math.isnan(val):
            return
        self.mean = self.mean + val
        self.counter.increment()

    def get_mean(self):
        if self.counter.count == 0:
            return 0
        else:
            return self.mean / self.counter.count


class KinematicMetrics:
    def __init__(self):
        self.velocity = Stats()
        self.acceleration = Stats()
        self.jerk = Stats()


class StrokeMetrics:
    def __init__(self):
        self.count = 0
        self.length = Stats()
        self.curvature = Stats()
        self.force = Stats()


class EvaluationMetrics:
    def __init__(self):
        self.kinematics = KinematicMetrics()
        self.removal_rate = Stats()
        self.strokes = StrokeMetrics()
        self.sensitive_voxels_removed = Counter()
        self.bone_voxels_removed = Counter()
        self.duration = 0

    def check_voxels_removed(self, voxels_colors):
        ctr = Counter()
        for c in voxels_colors:
            if c[1] == 255 or c[2] == 249 or c[3] == 219:
                self.bone_voxels_removed.increment()
            else:
                self.sensitive_voxels_removed.increment()
                ctr.increment()

        return ctr.count
    def print(self):
        print('Total Metrics: ')
        print('\t Stroke Count: ', self.strokes.count)
        print('\t Bone Voxels Removed: ', self.bone_voxels_removed.count)
        print('\t Sensitive Voxels Removed: ', self.sensitive_voxels_removed.count)
        print('\t Mean Stroke Length: ', self.strokes.length.get_mean())
        print('\t Mean Stroke Curvature: ', self.strokes.curvature.get_mean())
        print('\t Mean Stroke Force: ', self.strokes.force.get_mean())
        print('\t Mean Removal Rate ', self.removal_rate.get_mean())
        print('\t Mean Velocity : ', self.kinematics.velocity.get_mean())
        print('\t Mean Acceleration : ', self.kinematics.acceleration.get_mean())
        print('\t Mean Jerk : ', self.kinematics.jerk.get_mean())
        print('\t Procedure Duration ', self.duration)