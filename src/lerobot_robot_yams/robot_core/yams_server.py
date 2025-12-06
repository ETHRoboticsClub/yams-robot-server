import signal

import portal
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import GripperType


def run_robot_server(config) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    gripper_type = GripperType.from_string_name(config.gripper)
    robot = get_yam_robot(channel=config.can_port, gripper_type=gripper_type)

    server = YamsServer(robot, config.server_port)
    server.serve()


class YamsServer:
    """A simple server for a Yams robot."""

    def __init__(self, robot: Robot, port: int):
        self._robot = robot
        self._server = portal.Server(port)
        print(f"Robot Sever Binding to {port}, Robot: {robot}")

        self._server.bind("num_dofs", self._robot.num_dofs)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("command_joint_state", self._robot.command_joint_state)
        self._server.bind("get_observations", self._robot.get_observations)
        self._server.bind("get_robot_info", self._robot.get_robot_info)

    def serve(self) -> None:
        """Serve the leader robot."""
        self._server.start()
