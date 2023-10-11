from functools import reduce
import operator
from typing import Dict, List, NamedTuple, Optional, Set, Tuple
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QTransform
from . import Extent
from .util import batched


class Point(NamedTuple):
    x: float = 0
    y: float = 0

    @staticmethod
    def from_qt(qpoint: QPointF):
        return Point(qpoint.x(), qpoint.y())


body_parts = [
    "Nose",  # 0
    "Neck",  # 1
    "RShoulder",  # 2
    "RElbow",  # 3
    "RWrist",  # 4
    "LShoulder",  # 5
    "LElbow",  # 6
    "LWrist",  # 7
    "RHip",  # 8
    "RKnee",  # 9
    "RAnkle",  # 10
    "LHip",  # 11
    "LKnee",  # 12
    "LAnkle",  # 13
    "REye",  # 14
    "LEye",  # 15
    "REar",  # 16
    "LEar",  # 17
]
joint_count = len(body_parts)

bone_connection = [
    (1, 2),  # 0
    (1, 5),  # 1
    (2, 3),  # 2
    (3, 4),  # 3
    (5, 6),  # 4
    (6, 7),  # 5
    (1, 8),  # 6
    (8, 9),  # 7
    (9, 10),  # 8
    (1, 11),  # 9
    (11, 12),  # 10
    (12, 13),  # 11
    (1, 0),  # 12
    (0, 14),  # 13
    (14, 16),  # 14
    (0, 15),  # 15
    (15, 17),  # 16
]
assert len(bone_connection) == joint_count - 1

colors = [
    "ff0000",
    "ff5500",
    "ffaa00",
    "ffff00",
    "aaff00",
    "55ff00",
    "00ff00",
    "00ff55",
    "00ffaa",
    "00ffff",
    "00aaff",
    "0055ff",
    "0000ff",
    "5500ff",
    "aa00ff",
    "ff00ff",
    "ff00aa",
    "ff0055",
]
assert len(body_parts) == joint_count


class JointIndex(NamedTuple):
    person: int
    joint: int


class BoneIndex(NamedTuple):
    person: int
    bone: int


def bone_id(index: BoneIndex):
    return f"P{index.person:02d}_B{index.bone:02d}"


def joint_id(index: JointIndex):
    return f"P{index.person:02d}_J{index.joint:02d}"


def parse_id(string: str):
    if len(string) != 7 or string[0] != "P" or string[3] != "_":
        return None
    person, part = int(string[1:3]), int(string[5:7])
    if string[4] == "J":
        return JointIndex(person, part)
    elif string[4] == "B":
        return BoneIndex(person, part)
    return None


def get_connected_bones(joint_id: int):
    return [i for i, (a, b) in enumerate(bone_connection) if a == joint_id or b == joint_id]


class Shape:
    def __init__(self, name: str, position: Point):
        self._name = name
        self._position = QPointF(*position)
        self.removed = False

    def name(self):
        return self._name

    def position(self):
        return self._position - QPointF(4, 4)

    def set_position(self, x, y):
        self._position = QPointF(x, y)

    def remove(self):
        self.removed = True


class Pose:
    extent: Extent
    people: Set[int]
    joints: Dict[JointIndex, Point]

    def __init__(
        self,
        extent: Extent,
        people: Optional[Set[int]] = None,
        joint_positions: Optional[Dict[JointIndex, Point]] = None,
    ):
        self.extent = extent
        self.people = people or set()
        self.joints = joint_positions or {}

    @staticmethod
    def from_open_pose_json(pose: dict):
        # Format described at https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
        extent = Extent(pose["canvas_width"], pose["canvas_height"])

        def parse_keypoints(person: int, keypoints: List[float]):
            assert len(keypoints) // 3 == joint_count, "Invalid keypoint count in OpenPose JSON"
            return {
                JointIndex(person, joint): Point(x, y)
                for joint, (x, y, confidence) in enumerate(batched(keypoints, 3))
                if confidence > 0.1
            }

        people = pose.get("people", [])
        poses = (parse_keypoints(i, p.get("pose_keypoints_2d", [])) for i, p in enumerate(people))
        return Pose(extent, set(range(len(people))), reduce(operator.ior, poses, {}))

    def update(self, shapes: List[Shape], resolution=1.0):
        changed = set()
        bones: Dict[BoneIndex, Shape] = {}

        for shape in shapes:
            index = parse_id(shape.name())
            if index:
                self.people.add(index.person)
            if isinstance(index, JointIndex):
                pos = (shape.position() + QPointF(4, 4)) * resolution
                pos = Point.from_qt(pos)
                if not index in self.joints:
                    self.joints[index] = pos
                else:
                    last_pos = self.joints[index]
                    if pos.x != last_pos.x or pos.y != last_pos.y:
                        changed.add(index)
                        self.joints[index] = pos
            elif isinstance(index, BoneIndex):
                bones[index] = shape

        if len(changed) == 0:
            return None

        width, height = self.extent.width, self.extent.height
        new_bones = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
            f' viewBox="0 0 {width} {height}">'
        )

        for person, joint_index in changed:
            connected = get_connected_bones(joint_index)
            for connected_bone_index in connected:
                bone_index = BoneIndex(person, connected_bone_index)
                bone_shape = bones.get(bone_index)
                if bone_shape:
                    bone_joints = bone_connection[bone_index.bone]
                    joint_a = self.joints.get(JointIndex(person, bone_joints[0]))
                    joint_b = self.joints.get(JointIndex(person, bone_joints[1]))
                    if joint_a and joint_b:
                        new_bones += _draw_bone(bone_index, joint_a, joint_b)
                    bone_shape.remove()
                    del bones[bone_index]

        return new_bones + "</svg>"

    def to_svg(self):
        width, height = self.extent.width, self.extent.height
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
            f' viewBox="0 0 {width} {height}">'
        )

        for i, pos in self.joints.items():
            svg += _draw_joint(i, pos)

        for person in self.people:
            for i, bone in enumerate(bone_connection):
                beg = self.joints.get(JointIndex(person, bone[0]))
                end = self.joints.get(JointIndex(person, bone[1]))
                if beg and end:
                    svg += _draw_bone(BoneIndex(person, i), beg, end)

        return svg + "</svg>"


def _draw_bone(index: BoneIndex, a: Point, b: Point):
    return (
        f'<line id="{bone_id(index)}" x1="{a.x}" y1="{a.y}" x2="{b.x}" y2="{b.y}"'
        f' stroke="#{colors[index.bone]}" stroke-width="4" stroke-opacity="0.6"/>'
    )


def _draw_joint(index: JointIndex, pos: Point):
    return (
        f'<circle id="{joint_id(index)}" cx="{pos.x}" cy="{pos.y}" r="4"'
        f' fill="#{colors[index.joint]}"/>'
    )
