"""Homework 1 simulation scene. This file is provided, do not change it.

This module provides:
- The UR5e robot and 2F85 gripper models, downloaded from the MuJoCo Menagerie on
  first use.
- The MuJoCo model and data of our working environment, shared by the notebook and the
  task files.
- The robot's home configuration.
"""

from pathlib import Path
import subprocess

import mujoco
import numpy as np

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"
MENAGERIE_PATH = Path(__file__).parent / "mujoco_menagerie"
ROBOT_DIR = "universal_robots_ur5e"
GRIPPER_DIR = "robotiq_2f85_v4"

# Download UR5e robot and 2F85 gripper MJCF files from DeepMind's MuJoCo Menagerie.
# The full repository takes up a few GB, so we only check out the models we use.
if not MENAGERIE_PATH.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         MENAGERIE_URL, str(MENAGERIE_PATH)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(MENAGERIE_PATH), "sparse-checkout", "set", ROBOT_DIR, GRIPPER_DIR],
        check=True,
    )

robot_path = MENAGERIE_PATH / ROBOT_DIR / "ur5e.xml"
gripper_path = MENAGERIE_PATH / GRIPPER_DIR / "2f85.xml"

# Define the scene XML
xml = """
<mujoco model="tableworld">
    <statistic center="0.3 0 0.4" extent="0.8"/>

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="45" elevation="-20"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
                 markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    </asset>

    <worldbody>
        <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
        <geom name="floor" pos="0.0 0.0 0.0" size="0 0 0.05" type="plane" material="groundplane"/>

        <body name="table" pos="0.3 0.0 0">
            <geom name="table_top" type="box" size=".37 .37 .01" pos="0 0 .7"/>
            <geom name="table_leg_1" type="cylinder" size=".01 .7" pos=".24 -.24 -.0" euler="0 0 0"/>
            <geom name="table_leg_2" type="cylinder" size=".01 .7" pos=".24 .24 -.0" euler="0 0 0"/>
            <geom name="table_leg_3" type="cylinder" size=".01 .7" pos="-.24 -.24 -.0" euler="0 0 0"/>
            <geom name="table_leg_4" type="cylinder" size=".01 .7" pos="-.24 .24 -.0" euler="0 0 0"/>
        </body>

        <body name="red_box" pos="0.3 -0.2 0.75">
            <freejoint name="red_box_joint"/>
            <geom name="red_box_geom" type="box" size=".03 .03 .03" rgba="1 0 0 1"/>
        </body>
        <body name="green_bin" pos="0.45 0.25 0.71">
            <geom name="green_bin_bottom" type="box" size=".08 .08 .01" rgba="0 1 0 1"/>
            <geom name="green_bin_left_wall" type="box" size=".08 .01 .08" rgba="0 1 0 1" pos="0 -0.08 0.04"/>
            <geom name="green_bin_right_wall" type="box" size=".08 .01 .08" rgba="0 1 0 1" pos="0 0.08 0.04"/>
            <geom name="green_bin_back_wall" type="box" size=".01 .08 .08" rgba="0 1 0 1" pos="-0.08 0 0.04"/>
            <geom name="green_bin_front_wall" type="box" size=".01 .08 .08" rgba="0 1 0 1" pos="0.08 0 0.04"/>
        </body>

        <body name="obstacle1" pos="0.1 0.3 1.01">
            <freejoint/>
            <geom type="box" size=".025 .025 .3"  rgba="0 0 1 1"/>
        </body>
        <body name="obstacle2" pos="0.1 -0.3 1.01">
            <freejoint/>
            <geom type="box" size=".025 .025 .3"  rgba="0 0 1 1"/>
        </body>
        <body name="obstacle3" pos="0.55 0.05 1.01">
            <freejoint/>
            <geom type="box" size=".025 .025 .3"  rgba="0 0 1 1"/>
        </body>
        <body name="obstacle4" pos="0.3 0 0.81">
            <freejoint/>
            <geom type="box" size=".025 .025 .1"  rgba="0 0 1 1"/>
        </body>

        <!--angle view-->
        <camera name="rightangleview" pos="2.182 1.724 2.590" xyaxes="-0.685 0.729 0.000 -0.456 -0.429 0.780"/>
        <camera name="leftangleview" pos="2.182 -1.724 2.590" xyaxes="0.685 0.729 0.000 -0.456 0.429 0.780"/>
        <!-- front view -->
        <camera mode="fixed" name="frontview" pos="2.5 0 1.45" quat="0.56 0.43 0.43 0.56"/>
        <!-- bird view -->
        <camera mode="fixed" name="birdview" pos="0.3 0 3.0" quat="0.7071 0 0 0.7071"/>
        <!-- side view -->
        <camera mode="fixed" name="leftsideview" pos="0.151 -2.941 0.9" xyaxes="1.000 0.000 0.000 0.000 0.000 1.000"/>
        <camera mode="fixed" name="rightsideview" pos="0.151 2.941 0.9" xyaxes="-1.000 0.000 0.000 0.000 0.000 1.000"/>

    </worldbody>

</mujoco>
"""

# load MJCF files
scene_spec = mujoco.MjSpec.from_string(xml)
robot_spec = mujoco.MjSpec.from_file(str(robot_path))
gripper_spec = mujoco.MjSpec.from_file(str(gripper_path))

# change robot actuation from position control to torque control
for act in robot_spec.actuators:
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.gainprm = [2000.0] + [0.0]*9
    act.biasprm = [0.0, 0.0, 0.0] + [0.0]*7  # makes it motor control
robot_spec.key('home').ctrl = [0.0] * len(robot_spec.actuators)

# the robot's home configuration
HOME_CONFIG = np.array(robot_spec.key('home').qpos)

# create a mount (body + site) in the parent where the robot should attach
mount = scene_spec.worldbody.add_body(name="mount", pos=[0.0, 0.0, 0.72], xyaxes=[0, -1, 0, 1, 0, 0])
mount_site = mount.add_site(name="mount_site", pos=[0, 0, 0])

# attach the entire robot spec to the parent's mount site
# The prefix renames attached elements. All robot related elements will be prefixed with "robot_"
scene_spec.attach(robot_spec, site=mount_site, prefix="robot_")

# attach the entire gripper spec to the robot's attachment site
# The prefix renames attached elements. All gripper related elements will be prefixed with "gripper_"
scene_spec.attach(gripper_spec, site="robot_attachment_site", prefix="gripper_")

# compile into a model and data
model = scene_spec.compile()
data = mujoco.MjData(model)
