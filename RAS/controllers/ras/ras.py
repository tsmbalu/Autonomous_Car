import cv2
from rasrobot import RASRobot
from lane_controller_simple import LaneController

import os
"""
Mission 1: Perception and Behaviour Coordination (25 marks)
Learning outcomes: 1, 2, 3

Scenario:
You are developing a controller for an autonomous vehicle. The existing controller 
can already follow the road and takes random turns at crossroads using a simple 
state machine for coordination of these behaviours.

Task: 
In addition to the current behaviour, the car should stop if it reaches traffic
lights which are red, and go once they are green.
In order to do so you will need to:
- Implement a percept based on traffic light detection
- Implement a behaviour that stops the car based on percept
- Coordinate these behaviours with the existing behaviours  

Choose a suitable approach for behaviour coordination based on unit 6 and justify 
your choice. 
Also, decide whether you use local, global, or hybrid sensing (cf. unit 3, slide 19).

Hint:
If you do not manage to implement the traffic light detector, create a mock-up 
perception module that allows you to complete all other components of this task.

"""



class MyRobot(RASRobot):
    def __init__(self):
        """
        The constructor has no parameters.
        """
        super(MyRobot, self).__init__()

    def run(self):
        """
        This function implements the main loop of the robot.
        """
        # initialise lane controller
        lc = LaneController(debug=False)  # set debug to True if you want visualisation and logs

        while self.tick():
            # get front camera from car and display it
            image = self.get_camera_image()
            name = 'camera image'
            scale = 2

            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, image.shape[1]*scale, image.shape[0]*scale)
            cv2.imshow(name, image)
            cv2.waitKey(1)
    
            # use lane controller to determine steering angle and speed
            steering_angle, speed = lc.predict_angle_and_speed(image)

            # set steering angle and speed of car (will be applied in next tick)
            self.set_steering_angle(steering_angle)
            self.set_speed(speed)


if __name__ == '__main__':
    # create a robot and let it run!
    robot = MyRobot()
    robot.run()
