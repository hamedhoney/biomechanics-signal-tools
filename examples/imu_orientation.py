"""Demonstrates scalar-first orientation, missing quaternion frames, and SLERP."""
import numpy as np
from biomechanics_signal_tools.rotation import Rotation, quaternions as q

known_times = np.array([0., 1., 2.])
known_quaternions = q.quaternion_from_axis_angle(np.deg2rad([0, 45, 90]),
                                                 [0, 0, 1])
requested = np.arange(0, 2.001, 0.25)
resampled = q.interpolate_quaternions(requested, known_times, known_quaternions)
rotated_x = q.rotate_vectors(resampled, [1., 0., 0.])
print("Rotation at t = 1.5 s, x vector:", rotated_x[6])

with_gap = known_quaternions.copy()
with_gap[1] = np.nan
recovered = q.fill_missing_quaternions(with_gap)
print("Missing orientation reconstructed (wxyz):", recovered[1])

pelvis = Rotation.from_euler([0, 0, 90], seq="xyz", degrees=True)
print("Pelvis-local x in global frame:", pelvis.apply([1, 0, 0]))
