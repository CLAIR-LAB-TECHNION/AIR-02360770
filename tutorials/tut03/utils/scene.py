"""Scene construction for the Tut03 notebook (Low-level Control).

The scene is the one from the previous tutorial, a UR5e mounted on a table
with a single red box. The table world is described in an MJCF file and the
robot comes from the MuJoCo Menagerie.
"""

import mujoco

UR_PREFIX = "ur5e/"


def build_scene_spec(scene_path: str, robot_path: str) -> mujoco.MjSpec:
    """Load the table scene and attach the UR5e robot to it.

    Args:
        scene_path (str): Path to the scene's MJCF file.
        robot_path (str): Path to the robot's MJCF file.

    Returns:
        mujoco.MjSpec: The combined spec. Call `compile()` on it to get the model.
    """
    # MjSpec.from_file picks a loader by file extension and does not recognize .mjcf,
    # so we read the scene file ourselves and parse its contents
    with open(scene_path) as f:
        scene_spec = mujoco.MjSpec.from_string(f.read())
    robot_spec = mujoco.MjSpec.from_file(robot_path)

    # create a mount (body + site) in the parent where the robot should attach
    mount = scene_spec.worldbody.add_body(name="mount", pos=[0.0, 0.0, 0.72], xyaxes=[-1, 0, 0, 0, -1, 0])
    mount_site = mount.add_site(name="mount_site", pos=[0, 0, 0])

    # attach the entire robot spec to the parent's mount site (prefix renames attached elements)
    scene_spec.attach(robot_spec, site=mount_site, prefix=UR_PREFIX)

    return scene_spec
