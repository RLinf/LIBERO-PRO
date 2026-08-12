from .mounted_panda import MountedPanda
from .on_the_ground_panda import OnTheGroundPanda

# robosuite 1.5 replaced SingleArm with FixedBaseRobot; both LIBERO Panda
# variants are fixed-base single-arm robots.
from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot

ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": FixedBaseRobot,
        "OnTheGroundPanda": FixedBaseRobot,
    }
)
