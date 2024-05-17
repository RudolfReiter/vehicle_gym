import rclpy
from rclpy.node import Node
import numpy as np

from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import Odometry


class StatePublisher(Node):

    def __init__(self):
        super().__init__('state_publisher')
        self.publisher_ = self.create_publisher(Odometry, 'state', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        odometry_state_msg = Odometry()

        odometry_state_msg.pose.pose.position.x = 1.0
        odometry_state_msg.pose.pose.position.y = 2.0

        odometry_state_msg.twist.twist.linear.x = 1.0
        odometry_state_msg.twist.twist.linear.y = 2.0


        yaw = 90
        r = R.from_euler('z', yaw, degrees=True)
        yaw_quaternion = r.as_quat() # [x,y,z,w] ?
        odometry_state_msg.pose.pose.orientation.x = yaw_quaternion[0]
        odometry_state_msg.pose.pose.orientation.y = yaw_quaternion[1]
        odometry_state_msg.pose.pose.orientation.z = yaw_quaternion[2]
        odometry_state_msg.pose.pose.orientation.w = yaw_quaternion[3]

        
        self.publisher_.publish(odometry_state_msg)
        self.get_logger().info('Publishing: "%s"' % odometry_state_msg.pose.pose.orientation)
        #self.i += 1


def main(args=None):
    rclpy.init(args=args)

    state_publisher = StatePublisher()

    rclpy.spin(state_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    state_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()