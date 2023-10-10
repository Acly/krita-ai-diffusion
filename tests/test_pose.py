from ai_diffusion import Extent
from ai_diffusion.pose import (
    Pose,
    BoneIndex,
    JointIndex,
    Point,
    Shape,
    parse_id,
    joint_id,
    bone_id,
    get_connected_bones,
)


def test_bone_id():
    assert bone_id(BoneIndex(0, 0)) == "P00_B00"
    assert bone_id(BoneIndex(0, 1)) == "P00_B01"
    assert bone_id(BoneIndex(1, 0)) == "P01_B00"
    assert bone_id(BoneIndex(1, 1)) == "P01_B01"


def test_joint_id():
    assert joint_id(JointIndex(0, 0)) == "P00_J00"
    assert joint_id(JointIndex(0, 1)) == "P00_J01"
    assert joint_id(JointIndex(1, 0)) == "P01_J00"
    assert joint_id(JointIndex(1, 1)) == "P01_J01"


def test_parse_id():
    assert parse_id("P00_J00") == JointIndex(0, 0)
    assert parse_id("P00_J01") == JointIndex(0, 1)
    assert parse_id("P01_J00") == JointIndex(1, 0)
    assert parse_id("P01_J01") == JointIndex(1, 1)
    assert parse_id("P00_B00") == BoneIndex(0, 0)
    assert parse_id("P00_B01") == BoneIndex(0, 1)
    assert parse_id("P01_B00") == BoneIndex(1, 0)
    assert parse_id("P01_B01") == BoneIndex(1, 1)
    assert parse_id("") is None
    assert parse_id("P00") is None
    assert parse_id("P00_X82") is None
    assert parse_id("P00_J82_bla") is None


def test_connected_bones():
    assert get_connected_bones(0) == [12, 13, 15]
    assert get_connected_bones(1) == [0, 1, 6, 9, 12]
    assert get_connected_bones(2) == [0, 2]
    assert get_connected_bones(3) == [2, 3]
    assert get_connected_bones(4) == [3]


def test_pose_from_json():
    json = {
        "canvas_width": 123,
        "canvas_height": 456,
        "people": [
            {
                "pose_keypoints_2d": [11, 12, 1, 21, 22, 1, 31, 32, 0.5, 41, 42, 0.9, 51, 52, 0] + [
                    0 for i in range(3 * 13)
                ]
            },
            {
                "pose_keypoints_2d": [-11, -12, 1, -21, -22, 1, -31, -32, 0.5] + [
                    0 for i in range(3 * 15)
                ]
            },
        ],
    }
    pose = Pose.from_open_pose_json(json)
    assert len(pose.people) == 2
    assert pose.extent == Extent(123, 456)
    assert len(pose.joints) == 4 + 3
    assert pose.joints[JointIndex(0, 0)] == Point(11, 12)
    assert pose.joints[JointIndex(0, 1)] == Point(21, 22)
    assert pose.joints[JointIndex(0, 2)] == Point(31, 32)
    assert pose.joints[JointIndex(0, 3)] == Point(41, 42)
    assert pose.joints[JointIndex(1, 0)] == Point(-11, -12)


def test_pose_to_svg():
    joints = {
        JointIndex(0, 0): Point(11, 12),
        JointIndex(0, 1): Point(21, 22),
        JointIndex(0, 14): Point(31, 32),
    }
    pose = Pose(Extent(123, 456), set([0]), joints)
    svg = pose.to_svg()
    assert '<circle id="P00_J00" cx="11" cy="12"' in svg
    assert '<circle id="P00_J01" cx="21" cy="22"' in svg
    assert '<circle id="P00_J14" cx="31" cy="32"' in svg
    assert '<line id="P00_B12" x1="21" y1="22" x2="11" y2="12"' in svg
    assert '<line id="P00_B13" x1="11" y1="12" x2="31" y2="32"' in svg


def test_pose_update():
    joints = {
        JointIndex(0, 0): Point(11, 12),
        JointIndex(0, 1): Point(21, 22),
        JointIndex(0, 14): Point(31, 32),
    }
    pose = Pose(Extent(123, 456), set([0]), joints)
    shapes = [
        Shape("P00_J00", Point(11, 12)),
        Shape("P00_J01", Point(21, 22)),
        Shape("P00_J14", Point(31, 32)),
        Shape("P00_B12", Point(21, 22)),
        Shape("P00_B13", Point(11, 12)),
    ]
    changes = pose.update(shapes)
    assert changes is None

    shapes[2].set_position(31, 38)
    changes = pose.update(shapes)
    assert "B12" not in changes
    assert '<line id="P00_B13" x1="11" y1="12" x2="31.0" y2="38.0"' in changes
    assert not shapes[3].removed and shapes[4].removed

    shapes[0].set_position(11, 18)
    changes = pose.update(shapes)
    assert '<line id="P00_B12" x1="21" y1="22" x2="11.0" y2="18.0"' in changes
    assert '<line id="P00_B13" x1="11.0" y1="18.0" x2="31.0" y2="38.0"' in changes
    assert shapes[3].removed and shapes[4].removed
