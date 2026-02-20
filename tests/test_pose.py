import pytest

from ai_diffusion.image import Extent
from ai_diffusion.pose import (
    BoneIndex,
    JointIndex,
    Point,
    Pose,
    Shape,
    get_connected_bones,
    parse_id,
)


def test_bone_id():
    assert BoneIndex(0, 0).id == "P00_B00"
    assert BoneIndex(0, 1).id == "P00_B01"
    assert BoneIndex(1, 0).id == "P01_B00"
    assert BoneIndex(1, 1).id == "P01_B01"


def test_joint_id():
    assert JointIndex(0, 0).id == "P00_J00"
    assert JointIndex(0, 1).id == "P00_J01"
    assert JointIndex(1, 0).id == "P01_J00"
    assert JointIndex(1, 1).id == "P01_J01"


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
                "pose_keypoints_2d": [11, 12, 1, 21, 22, 1, 31, 32, 0.5, 41, 42, 0.9, 51, 52, 0]
                + [0 for i in range(3 * 13)]
            },
            {
                "pose_keypoints_2d": [-11, -12, 1, -21, -22, 1, -31, -32, 0.5]
                + [0 for i in range(3 * 15)]
            },
        ],
    }
    pose = Pose.from_open_pose_json(json)
    assert pose.people_count == 2
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
    pose = Pose(Extent(123, 456), 1, joints)
    svg = pose.to_svg()
    assert '<circle id="P00_J00" cx="11" cy="12"' in svg
    assert '<circle id="P00_J01" cx="21" cy="22"' in svg
    assert '<circle id="P00_J14" cx="31" cy="32"' in svg
    assert '<line id="P00_B12" x1="21" y1="22" x2="11" y2="12"' in svg
    assert '<line id="P00_B13" x1="11" y1="12" x2="31" y2="32"' in svg


def test_pose_update():
    pose = Pose(Extent(123, 456))
    shapes = [
        Shape("P00_J00", Point(11, 12)),
        Shape("P00_J01", Point(21, 22)),
        Shape("P00_J14", Point(31, 32)),
        Shape("P00_B12", Point(21, 22)),
        Shape("P00_B13", Point(11, 12)),
    ]
    changes = pose.update(shapes)
    assert pose.people_count == 1
    assert len(pose.joints) == 3
    assert pose.joints[JointIndex(0, 0)] == Point(11, 12)

    shapes[3].removed = shapes[4].removed = False
    shapes[2].set_position(31, 38)
    changes = pose.update(shapes)
    assert changes is not None
    assert "B12" not in changes
    assert '<line id="P00_B13" x1="11.0" y1="12.0" x2="31.0" y2="38.0"' in changes
    assert not shapes[3].removed and shapes[4].removed

    shapes[0].set_position(11, 18)
    changes = pose.update(shapes)
    assert changes is not None
    assert '<line id="P00_B12" x1="21.0" y1="22.0" x2="11.0" y2="18.0"' in changes
    assert '<line id="P00_B13" x1="11.0" y1="18.0" x2="31.0" y2="38.0"' in changes
    assert shapes[3].removed and shapes[4].removed


@pytest.mark.parametrize("scenario", ["with_bones", "without_bones"])
def test_pose_update_copy(scenario):
    pose = Pose(Extent(123, 456))
    shapes = [
        Shape("P00_J00", Point(11, 12)),
        Shape("P00_J01", Point(21, 22)),
    ]
    pose.update(shapes)
    assert pose.people_count == 1
    assert len(pose.joints) == 2

    shapes[1].set_position(21, 32)
    shapes += [
        Shape("P00_J00", Point(51, 52)),
        Shape("P00_J01", Point(61, 62)),
    ]
    if scenario == "with_bones":
        shapes += [
            Shape("P00_B12", Point(21, 22)),
            Shape("P00_B12", Point(21, 22)),
        ]
    changes = pose.update(shapes)
    assert pose.people_count == 2
    assert pose.joints[JointIndex(0, 0)] == Point(11, 12)
    assert pose.joints[JointIndex(0, 1)] == Point(21, 32)
    assert pose.joints[JointIndex(1, 0)] == Point(51, 52)
    assert pose.joints[JointIndex(1, 1)] == Point(61, 62)
    assert changes is not None
    assert '<line id="P00_B12" x1="21.0" y1="32.0" x2="11.0" y2="12.0"' in changes
    assert '<line id="P01_B12" x1="61.0" y1="62.0" x2="51.0" y2="52.0"' in changes
    assert shapes[2].name() == "P01_J00"
    assert shapes[3].name() == "P01_J01"
    if scenario == "with_bones":
        assert shapes[-1].removed and shapes[-2].removed


def test_stroke_width():
    joints = {JointIndex(0, 0): Point(100, 100), JointIndex(0, 1): Point(200, 200)}
    pose = Pose(Extent(2000, 2000), 1, joints)
    svg = pose.to_svg()
    assert 'stroke-width="11.' in svg
    assert 'r="11.' in svg
