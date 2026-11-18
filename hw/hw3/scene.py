"""Homework 3 simulation scene. This file is provided, do not change it.

This module provides:
- The UR5e robot and 2F85 gripper models, downloaded from the MuJoCo Menagerie on
  first use.
- The MuJoCo model and data of our working environment, shared by the notebooks and the
  task files.
"""

from pathlib import Path
import subprocess

import mujoco

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"
MENAGERIE_PATH = Path(__file__).parent / "mujoco_menagerie"
ROBOT_DIR = "universal_robots_ur5e"
GRIPPER_DIR = "robotiq_2f85"

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

        <body name="red_box" pos="0.2 -0.2 0.725">
            <freejoint name="red_box_joint"/>
            <geom name="red_box_geom" type="box" size=".03 .03 .03" rgba="1 0 0 1" friction="2.0 0.05 0.01"/>
        </body>
        <body name="green_box" pos="0.2 0.2 0.725">
            <freejoint name="green_box_joint"/>
            <geom name="green_box_geom" type="box" size=".03 .03 .03" rgba="0 1 0 1" friction="2.0 0.05 0.01"/>
        </body>
        <body name="blue_box" pos="0.4 -0.2 0.725">
            <freejoint name="blue_box_joint"/>
            <geom name="blue_box_geom" type="box" size=".03 .03 .03" rgba="0 0 1 1" friction="2.0 0.05 0.01"/>
        </body>
        <body name="yellow_box" pos="0.4 0.2 0.725">
            <freejoint name="yellow_box_joint"/>
            <geom name="yellow_box_geom" type="box" size=".03 .03 .03" rgba="1 1 0 1" friction="2.0 0.05 0.01"/>
        </body>

        <geom name="robot_mount" type="cylinder" size="0.1 0.7" pos="-0.2 0 0.0" rgba="0.1 0.1 0.1 1"/>

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

# create a mount (body + site) in the parent where the robot should attach
mount = scene_spec.worldbody.add_body(name="mount", pos=[-0.2, 0.0, 0.72], xyaxes=[0, -1, 0, 1, 0, 0])
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
