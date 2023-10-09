from ai_diffusion import Extent
from ai_diffusion.pose import Pose, Point, Shape, parse_id, joint_id, bone_id, get_connected_bones


def test_bone_id():
    assert bone_id(0, 0) == "P00_B00"
    assert bone_id(0, 1) == "P00_B01"
    assert bone_id(1, 0) == "P01_B00"
    assert bone_id(1, 1) == "P01_B01"


def test_joint_id():
    assert joint_id(0, 0) == "P00_J00_Nose"
    assert joint_id(0, 1) == "P00_J01_Neck"
    assert joint_id(1, 0) == "P01_J00_Nose"
    assert joint_id(1, 1) == "P01_J01_Neck"


def test_parse_id():
    assert parse_id("P00_J00_Nose") == ("J", 0, 0)
    assert parse_id("P00_J01_Neck") == ("J", 0, 1)
    assert parse_id("P01_J00_Nose") == ("J", 1, 0)
    assert parse_id("P01_J01_Neck") == ("J", 1, 1)
    assert parse_id("P00_B00") == ("B", 0, 0)
    assert parse_id("P00_B01") == ("B", 0, 1)
    assert parse_id("P01_B00") == ("B", 1, 0)
    assert parse_id("P01_B01") == ("B", 1, 1)


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
    poses = Pose.from_open_pose_json(json)
    assert len(poses) == 2
    pose = poses[0]
    assert pose.extent == Extent(123, 456) and pose.person == 0
    assert len(pose.joints) == 4
    assert pose.joints[0] == Point(11, 12)
    assert pose.joints[1] == Point(21, 22)
    assert pose.joints[2] == Point(31, 32)
    assert pose.joints[3] == Point(41, 42)
    pose = poses[1]
    assert pose.extent == Extent(123, 456) and pose.person == 1
    assert len(pose.joints) == 3
    assert pose.joints[0] == Point(-11, -12)


def test_pose_to_svg():
    pose = Pose(0, Extent(123, 456), {0: Point(11, 12), 1: Point(21, 22), 14: Point(31, 32)})
    svg = Pose.to_svg([pose])
    assert '<circle id="P00_J00_Nose" cx="11" cy="12"' in svg
    assert '<circle id="P00_J01_Neck" cx="21" cy="22"' in svg
    assert '<circle id="P00_J14_REye" cx="31" cy="32"' in svg
    assert '<line id="P00_B12" x1="21" y1="22" x2="11" y2="12"' in svg
    assert '<line id="P00_B13" x1="11" y1="12" x2="31" y2="32"' in svg


def test_pose_update():
    pose = Pose(0, Extent(123, 456), {0: Point(11, 12), 1: Point(21, 22), 14: Point(31, 32)})
    shapes = [
        Shape("P00_J00_Nose", Point(11, 12)),
        Shape("P00_J01_Neck", Point(21, 22)),
        Shape("P00_J14_REye", Point(31, 32)),
        Shape("P00_B12", Point(21, 22)),
        Shape("P00_B13", Point(11, 12)),
    ]
    changes = Pose.update_all([pose], shapes)
    assert changes is None

    shapes[2].set_position(31, 38)
    changes = Pose.update_all([pose], shapes)
    assert "B12" not in changes
    assert '<line id="P00_B13" x1="11" y1="12" x2="31.0" y2="38.0"' in changes
    assert not shapes[3].removed and shapes[4].removed

    shapes[0].set_position(11, 18)
    changes = Pose.update_all([pose], shapes)
    assert '<line id="P00_B12" x1="21" y1="22" x2="11.0" y2="18.0"' in changes
    assert '<line id="P00_B13" x1="11.0" y1="18.0" x2="31.0" y2="38.0"' in changes
    assert shapes[3].removed and shapes[4].removed
