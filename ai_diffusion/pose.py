from typing import Dict, List, NamedTuple, Tuple
from PyQt5.QtCore import QPointF
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


def bone_id(person_index: int, part_index: int):
    return f"P{person_index:02d}_B{part_index:02d}"


def joint_id(person_index: int, part_index: int):
    return f"P{person_index:02d}_J{part_index:02d}_{body_parts[part_index]}"


def parse_id(bone_id: str):
    if len(bone_id) < 7 or bone_id[0] != "P" or bone_id[3] != "_":
        return None, None, None
    return bone_id[4], int(bone_id[1:3]), int(bone_id[5:7])


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
        return self._position

    def set_position(self, x, y):
        self._position = QPointF(x, y)

    def remove(self):
        self.removed = True


class Pose:
    person: int
    extent: Extent
    joints: Dict[int, Point]

    def __init__(self, person: int, extent: Extent, initial_positions: Dict[int, Point]):
        self.person = person
        self.extent = extent
        self.joints = initial_positions

    @staticmethod
    def from_open_pose_json(pose: dict):
        # Format described at https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
        extent = Extent(pose["canvas_width"], pose["canvas_height"])

        def parse_keypoints(keypoints: List[float]):
            assert len(keypoints) // 3 == joint_count, "Invalid keypoint count in OpenPose JSON"
            return {i: Point(x, y) for i, (x, y, c) in enumerate(batched(keypoints, 3)) if c > 0.1}

        people = pose.get("people", [])
        return [
            Pose(i, extent, parse_keypoints(p.get("pose_keypoints_2d", [])))
            for i, p in enumerate(people)
        ]

    def update(self, shapes: List[Shape]):
        deltas = {}
        bones: Dict[int, Shape] = {}
        for shape in shapes:
            kind, person, index = parse_id(shape.name())
            if person != self.person:
                continue
            if kind == "J":
                pos = Point.from_qt(shape.position())
                if not index in self.joints:
                    self.joints[index] = pos
                else:
                    last_pos = self.joints[index]
                    delta = (pos.x - last_pos.x, pos.y - last_pos.y)
                    if delta != (0, 0):
                        deltas[index] = (pos.x - last_pos.x, pos.y - last_pos.y)
                        self.joints[index] = pos
            elif kind == "B":
                bones[index] = shape

        if len(deltas) == 0:
            return ""

        new_bones = ""

        for joint_index, delta in deltas.items():
            connected = get_connected_bones(joint_index)
            for bone_index in connected:
                bone_shape = bones.get(bone_index)
                if bone_shape:
                    bone_shape.remove()
                    bone_joints = bone_connection[bone_index]
                    joint_a = self.joints.get(bone_joints[0])
                    joint_b = self.joints.get(bone_joints[1])
                    if joint_a and joint_b:
                        new_bones += _draw_bone(self.person, bone_index, joint_a, joint_b)
                        bones[bone_index] = bone_shape

                    del bones[bone_index]

        return new_bones

    @staticmethod
    def update_all(poses: List["Pose"], shapes: List[Shape]):
        new_bones = ""
        for pose in poses:
            new_bones += pose.update(shapes)
        if new_bones != "":
            return "<svg>" + new_bones + "</svg>"
        return None

    @staticmethod
    def to_svg(poses: List["Pose"]):
        width, height = poses[0].extent.width, poses[0].extent.height
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
            f' viewBox="0 0 {width} {height}">'
        )
        for person, pose in enumerate(poses):
            svg += _to_svg(person, pose)
        svg += "</svg>"
        return svg


def _draw_bone(person: int, index: int, a: Point, b: Point):
    return (
        f'<line id="{bone_id(person, index)}" x1="{a.x}" y1="{a.y}" x2="{b.x}" y2="{b.y}"'
        f' stroke="#{colors[index]}" stroke-width="4" stroke-opacity="0.6"/>'
    )


def _draw_joint(person: int, index: int, pos: Point):
    return (
        f'<circle id="{joint_id(person, index)}" cx="{pos.x}" cy="{pos.y}" r="4"'
        f' fill="#{colors[index]}"/>'
    )


def _to_svg(person: int, pose: Pose):
    svg = ""
    for i, pos in pose.joints.items():
        svg += _draw_joint(person, i, pos)

    for i, bone in enumerate(bone_connection):
        beg = pose.joints.get(bone[0])
        end = pose.joints.get(bone[1])
        if beg and end:
            svg += _draw_bone(person, i, beg, end)

    return svg
