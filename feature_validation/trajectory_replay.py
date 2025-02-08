#!/usr/bin/env python
# //==============================================================================
# /*
#     Software License Agreement (BSD License)
#     Copyright (c) 2020-2025 Johns Hopkins University (JHU), All Rights Reserved.


#     All rights reserved.

#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions
#     are met:

#     * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.

#     * Neither the name of authors nor the names of its contributors may
#     be used to endorse or promote products derived from this software
#     without specific prior written permission.

#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#     "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#     LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#     FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#     COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#     INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#     BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#     ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#     POSSIBILITY OF SUCH DAMAGE.


#     \author    <amunawa2@jh.edu>
#     \author    Adnan Munawar
#     \version   1.0
# */
# //==============================================================================

import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from data_merger import DataMerger
from PyKDL import Frame, Rotation, Vector
import rospy
from geometry_msgs.msg import Pose
from ambf_client import Client
import time
from interpolation import Interpolation

import sys, signal
def signal_handler(signal, frame):
    print("\n WARNING! CTRL+C PRESSED. Terminating program")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def pose_list_to_pose_matrix(data):
    p = pose_list_to_position(data)
    r = pose_list_to_rotation(data)
    return Frame(r, p)

def pose_list_to_position(data):
    p = Vector(data[0], data[1], data[2])
    return p

def pose_list_to_rotation(data):
    r = Rotation.Quaternion(data[3], data[4], data[5], data[6])
    return r

def pose_list_to_pose_msg(data):
    p = Pose()
    p.position.x = data[0]
    p.position.y = data[1]
    p.position.z = data[2]

    p.orientation.x = data[3]
    p.orientation.y = data[4]
    p.orientation.z = data[5]
    p.orientation.w = data[6]

    return p

def pos_rpy_list_to_pose_msg(data):
    p = Pose()
    p.position.x = data[0]
    p.position.y = data[1]
    p.position.z = data[2]

    q = Rotation.RPY(data[3], data[4], data[5]).GetQuaternion()
    p.orientation.x = q[0]
    p.orientation.y = q[1]
    p.orientation.z = q[2]
    p.orientation.w = q[3]

    return p

def pose_list_to_pos_rpy_list(data):
    r = pose_list_to_rotation(data)
    rpy = r.GetRPY()
    return [data[0], data[1], data[2], rpy[0], rpy[1], rpy[2]]


def main(args):
    print(args)
    resolved_path = Path(args.path)
    data_meger = DataMerger()
    data = data_meger.get_merged_data(resolved_path)
    
    interpolate_ = args.interpolate
    interpolator = Interpolation()

    # rospy.init_node('drill_trajectory_replay')
    # pose_pub = rospy.Publisher("/ambf/env/mastoidectomy_drill/Command")
    client  = Client()
    client.connect()

    drill_handle = client.get_obj_handle("mastoidectomy_drill")
    time.sleep(0.1)

    pose_list_array = data['data']['pose_mastoidectomy_drill']
    timestamp_list = data['data']['time']

    size_poses = len(pose_list_array)
    size_times = len(timestamp_list)
    print("Size of timestamps list ", size_poses)
    print("Size of poses list ", size_times)

    if size_times != size_poses:
        raise Exception("Error! Size of timestamps and poses do not match")
    
    if size_poses <= 1:
        raise Exception("Error! Not enough poses")
    
    dt = 0.001
    v0 = np.array([0, 0, 0, 0, 0, 0])
    vf = np.array([0, 0, 0, 0, 0, 0])
    a0 = np.array([0, 0, 0, 0, 0, 0])
    af = np.array([0, 0, 0, 0, 0, 0])
    
    for idx in range(size_poses):
        if interpolate_:
            if idx != size_poses - 1:
                p0 = np.array(pose_list_to_pos_rpy_list(pose_list_array[idx]))
                pf = np.array(pose_list_to_pos_rpy_list(pose_list_array[idx+1]))
                time_step = timestamp_list[idx + 1] - timestamp_list[idx]
                t0 = 0.0
                tf = t0 + time_step
                v0 = vf
                vf = (pf - p0) / time_step
                a0 = af
                af = (vf - v0) / time_step
                interpolator.compute_interpolation_params(p0, pf, v0, vf, a0, af, t0, tf)
            curr_start_time = rospy.Time.now().to_sec()
            t = 0.0
            print("INFO! Ctrl+C to terminate. Commanding at t", time_step)
            while True:
                t = rospy.Time.now().to_sec() - curr_start_time
                if t > tf:
                    break
                pt = interpolator.get_interpolated_x(t)
                pose_msg = pos_rpy_list_to_pose_msg(pt)
                drill_handle.set_pose(pose_msg)
                # print("INFO! Commanding at t: ", t)
                time.sleep(dt)

        else:
            if idx != size_poses - 1:
                dt = timestamp_list[idx + 1] - timestamp_list[idx]

            pose_list = pose_list_array[idx]
            pose_msg = pose_list_to_pose_msg(pose_list)
            drill_handle.set_pose(pose_msg)
            time.sleep(dt)

    print("INFO! GOOD BYE")


if __name__ == "__main__":
    parser = ArgumentParser()

    # path to adfs
    parser.add_argument('--path', type=str, required=True,
                        help='Path to recorded study with HDF5 files')
    parser.add_argument('-i', dest='interpolate', type=bool, default=True,
                        help='Enable interpolation between poses')
    args = parser.parse_args()
    main(args)