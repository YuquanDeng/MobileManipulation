import cv2
from scene_renderer import SceneRenderer
import numpy as np
import magnum as mn
import quaternion

def axisAngle2quaternion(rotation_axis, yaw_angle):
    """
    Args:
    yaw_angle: rotation angle in radian. 
    rotation_axis: (x, y, z) where x, y, z defines the axis of rotation.
    """
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    x, y, z =  tuple(rotation_axis)
    w = yaw_angle

    q_0 = np.cos(w / 2)
    q_1 = x * np.sin(w / 2)
    q_2 = y * np.sin(w / 2)
    q_3 = z * np.sin(w / 2)
    return quaternion.quaternion(np.quaternion(q_0, q_1, q_2, q_3))


def quaternion_from_axis_angle(theta, axis):
    # Convert the axis vector to a Magnum Vector3 object
    axis_magnum = mn.Vector3(*axis)

    # Create a Magnum Quaternion object representing the rotation
    q = mn.Quaternion.rotation(mn.Rad(theta), axis_magnum)

    # Return the quaternion as a tuple (x, y, z, w)
    return q.vector[0], q.vector[1], q.vector[2], q.scalar

# print(axisAngle2quaternion((0, 1, 0, np.pi/2)))
# theta = 1.57  # 90 degrees in radians
# axis = [0, 1, 0]  # Y axis
# q = quaternion_from_axis_angle(theta, axis)
# print(q)

print(axisAngle2quaternion(np.array([0, 1.0, 0]), 0.2))


raise NotImplementedError

sr = SceneRenderer(
    '/mnt/ssd1/habitat-data/versioned_data/hm3d-1.0/hm3d/example/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb',
    (256, 256), 60, 0.0, 1)

p = sr.get_random_navigable_point()
sr.set_camera_pose(0, p, (1, 0, 0, 0))
ob = sr.render(0)

cv2.imshow('', cv2.cvtColor(ob['rgb'], cv2.COLOR_RGB2BGR))
cv2.waitKey(0)


# Two ways: 
# (1) couple the SceneRenderer code with GNM policy. Use the waypoints instead of random navigable point.
#     and then denormalized the waypoint
