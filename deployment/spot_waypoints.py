"""
The function I wrote for familiaring with frame and waypoints command. 

"""

def go_to(self):
    # Lease acquiring.
    lease = self._lease_client.take()
    lease_keep = LeaseKeepAlive(self._lease_client)
    # Power on the robot and stand it up
    resp = self._robot.power_on()

    try:
        blocking_stand(self._robot_command_client)
    except CommandFailedError as exc:
        print(f'Error ({exc}) occurred while trying to stand. Check robot surroundings.')
        return False
    except CommandTimedOutError as exc:
        print(f'Stand command timed out: {exc}')
        return False
    print('Robot powered on and standing.')
    # WARNING: The mobility params depend on the Spot's current position and heading.
    mobility_params = self._get_mobility_params() 


    robot_state = self._robot_state_task.proto
    vo_tform_robot = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)
    goal_point = (-1, 0)
    waypoints = self._generate_waypoints(vo_tform_robot.x, 
                                            vo_tform_robot.y, 
                                            goal_point[0],
                                            goal_point[1],
                                            4)

    print("waypoints: ", waypoints)
    curr_idx = 0
    
    # Navigation Loop.
    while not SHUTDOWN_FLAG.value:
        if curr_idx >= len(waypoints):
            break
        goal_x, goal_y = waypoints[curr_idx]
        curr_idx += 1
        cmd = RobotCommandBuilder.trajectory_command(goal_x=goal_x,
                                                        goal_y=goal_y,
                                                        goal_heading=vo_tform_robot.rot.to_yaw(),
                                                frame_name=VISION_FRAME_NAME,
                                                params=mobility_params)
        end_time = 15.0

        self._robot_command_client.robot_command(lease=None, command=cmd,
                                            end_time_secs=time.time() + end_time)
        
def _generate_waypoints(self, x1, y1, x2, y2, n):
    print("relative distance: ", np.sqrt((x2-x1)**2 + (y2-y1)**2) / (n-1))
    waypoints = [(x1 + i*(x2 - x1)/(n - 1), y1 + i*(y2 - y1)/(n - 1)) for i in range(n)]
    return waypoints