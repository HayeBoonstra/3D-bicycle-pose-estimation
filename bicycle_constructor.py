import numpy as np

class Bicycle:
    def __init__(self):
        self.wheel_size = 28 * 0.0254
        self.wheel_width = 0.03
        self.frame_size = 0.55 # m (55 cm is standard frame size)
        self.seat_tube_angle = np.deg2rad(15) # degrees
        self.wheel_base = 1.05 # between 1.0 and 1.1 meters
        self.hub_raise = 0.05
        self.fork_angle = np.deg2rad(-20) # degrees
        self.fork_length = 0.0
        self.handlebar_height = 0.1 # m
        self.handlebar_width = 0.3 # m
        self.seat_tube_height = 0.0
        self.bottom_bracket = [0.0, 0.0, 0.0]
        self.rear_hub = [0.0, 0.0, 0.0]
        self.front_hub = [0.0, 0.0, 0.0]
        self.head_tube = [0.0, 0.0, 0.0]
        self.seat_tube = [0.0, 0.0, 0.0]
        self.seat_post = [0.0, 0.0, 0.0]
        self.wheel_clearance = 0.02
        self.rear_hub_distance = 0.43
        self.crank_width = 0.06
        self.crank_length = 0.175
        self.pedal_width = 0.04
        self.pedal_length = 0.06
        self.seat_height = 0.1
        self.seat_width = 0.04
        self.seat_length = 0.1
        self.seat_thickness = 0.01
        # Gear ratio: rear wheel revs per pedal rev (e.g. 44/11 = 4)
        self.gear_ratio = 1.4

        ## human variables
        self.torso_length = 0.48
        self.torso_width = 0.1
        self.torso_lean = 15 # degrees
        self.torso_mass = 10

        self.upper_arm_length = 0.4
        self.upper_arm_width = 0.016
        self.upper_arm_mass = 0.001
        self.lower_arm_length = 0.3
        self.lower_arm_width = 0.016
        self.lower_arm_mass = 0.001
        self.hand_length = 0.016
        self.hand_width = 0.016
        self.hand_mass = 0.001

    
    def create_bicycle_variables(self):
        ## fork geometry
        # X forward, Y left, Z up
        self.seat_tube_height = self.frame_size * np.sin(self.seat_tube_angle)
        self.fork_length = self.frame_size / np.cos(self.seat_tube_angle)

        ## frame geometry
        self.bottom_bracket = np.array([0.0, 0.0, 0.0])
        self.rear_hub = self.bottom_bracket + np.array([-self.rear_hub_distance, 0.0, self.hub_raise])
        self.front_hub = self.bottom_bracket + np.array([self.rear_hub[0] + self.wheel_base, 0.0, self.hub_raise])

        fork_dir = np.array([np.sin(self.fork_angle), 0.0, np.cos(self.fork_angle)])
        self.head_tube = self.front_hub + self.fork_length * fork_dir
        seat_tube_dir = np.array([-np.sin(self.seat_tube_angle), 0.0, np.cos(self.seat_tube_angle)])
        self.seat_tube = self.bottom_bracket + self.frame_size * seat_tube_dir

        seat_stay_attachment_ratio = 0.9
        self.seat_stay_attachment = self.bottom_bracket + self.seat_tube * seat_stay_attachment_ratio

        ## seat geometry
        self.seat_tube_post = self.seat_tube + np.array([-np.sin(self.seat_tube_angle), 0.0, np.cos(self.seat_tube_angle)]) * self.seat_height

        

    def create_bicycle_model(self):
        ## create the mujoco XML file
        wheel_xml = f"""
        <asset>
            <mesh name="wheel_torus" builtin="supertorus" params="64 0.04 1 1" scale="0.35 0.35 0.35"/>
        </asset>
        """

        actuator_xml = """
        <actuator>
            <motor name="pedal_drive" joint="pedals"/>
            <motor joint="steer"/>
        </actuator>
        """

        # Couple rear wheel to pedals: rear_angle = gear_ratio * pedal_angle
        equality_xml = f"""
        <equality>
            <joint joint1="rear wheel hinge" joint2="pedals" polycoef="0 {self.gear_ratio} 0 0 0"/>
            <weld site1="left hand site" site2="left handlebar site"/>
            <weld site1="right hand site" site2="right handlebar site"/>
        </equality>
        """

        frame_xml = f"""
        <worldbody>
            <body name="bicycle" pos="0 0 0.35">
                <freejoint name="bicycle_free"/>
                <body name="frame" pos="0 0 0">
                    <geom name="seat tube" type="capsule" fromto="{self.bottom_bracket[0]} {self.bottom_bracket[1]} {self.bottom_bracket[2]}  {self.seat_tube[0]} {self.seat_tube[1]} {self.seat_tube[2]}" size="0.016" contype="4" conaffinity="4"/>
                    <geom name="seat post" type="capsule" fromto="{self.seat_tube[0]} {self.seat_tube[1]} {self.seat_tube[2]}  {self.seat_tube_post[0]} {self.seat_tube_post[1]} {self.seat_tube_post[2]}" size="0.016" contype="4" conaffinity="4"/>
                    <body name="seat" pos="{self.seat_tube_post[0]} {self.seat_tube_post[1]} {self.seat_tube_post[2]}">
                        <geom name="seat" type="box" size="{self.seat_length} {self.seat_width} {self.seat_thickness}" contype="4" conaffinity="4"/>
                        <body name="torso" pos="0 0 0.01" euler="0 15 0">
                            <geom name="torso" type="capsule" fromto="0 0 0 0 0 {self.torso_length}" size="{self.torso_width}" contype="4" conaffinity="4" mass="{self.torso_mass}"/>
                                <body name="left upper arm" pos="0 {self.torso_width/2} {self.torso_length}" euler="0 -15 0">
                                    <geom name="left upper arm" type="capsule" fromto="0 0 0 0 0 -{self.upper_arm_length}" size="{self.upper_arm_width}" contype="4" conaffinity="4" mass="{self.upper_arm_mass}"/>
                                    <joint name="left upper arm flexion" type="hinge" axis="0 1 0" pos="0 0 0" limited="true" range="-120 0"/>
                                    <joint name="left upper arm abduction" type="hinge" axis="1 0 0" pos="0 0 0" limited="true" range="0 60"/>
                                    <body name="left lower arm" pos="0 0 -{self.upper_arm_length}" euler="0 0 0">
                                        <geom name="left lower arm" type="capsule" fromto="0 0 0 0 0 -{self.lower_arm_length}" size="{self.lower_arm_width}" contype="4" conaffinity="4" mass="{self.lower_arm_mass}"/>
                                        <joint name="left lower arm hinge" type="hinge" axis="0 1 0" pos="0 0 0" limited="true" range="-120 0"/>
                                        <body name="left hand" pos="0 0 -{self.lower_arm_length}" euler="0 0 0">
                                            <geom name="left hand" type="capsule" fromto="0 0 0 0 0 -{self.hand_length}" size="{self.hand_width}" contype="4" conaffinity="4" mass="{self.hand_mass}"/>
                                            <site name="left hand site" pos="0 0 0"/>
                                        </body>
                                    </body>
                                </body>
                                <body name="right upper arm" pos="0 {-self.torso_width/2} {self.torso_length}" euler="0 -15 0">
                                    <geom name="right upper arm" type="capsule" fromto="0 0 0 0 0 -{self.upper_arm_length}" size="{self.upper_arm_width}" contype="4" conaffinity="4" mass="{self.upper_arm_mass}"/>
                                    <joint name="right upper arm flexion" type="hinge" axis="0 1 0" pos="0 0 0" limited="true" range="-120 0"/>
                                    <joint name="right upper arm abduction" type="hinge" axis="1 0 0" pos="0 0 0" limited="true" range="0 60"/>
                                    <body name="right lower arm" pos="0 0 -{self.upper_arm_length}" euler="0 0 0">
                                        <geom name="right lower arm" type="capsule" fromto="0 0 0 0 0 -{self.lower_arm_length}" size="{self.lower_arm_width}" contype="4" conaffinity="4" mass="{self.lower_arm_mass}"/>
                                        <joint name="right lower arm hinge" type="hinge" axis="0 1 0" pos="0 0 0" limited="true" range="-120 0"/>
                                        <body name="right hand" pos="0 0 -{self.lower_arm_length}" euler="0 0 0">
                                            <geom name="right hand" type="capsule" fromto="0 0 0 0 0 -{self.hand_length}" size="{self.hand_width}" contype="4" conaffinity="4" mass="{self.hand_mass}"/>
                                            <site name="right hand site" pos="0 0 0"/>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    <geom name="down tube" type="capsule" fromto="{self.bottom_bracket[0]} {self.bottom_bracket[1]} {self.bottom_bracket[2]}  {self.head_tube[0]} {self.head_tube[1]} {self.head_tube[2]}" size="0.016"/>
                    <geom name="top tube" type="capsule" fromto="{self.head_tube[0]} {self.head_tube[1]} {self.head_tube[2]}  {self.seat_tube[0]} {self.seat_tube[1]} {self.seat_tube[2]}" size="0.016"/>
                    <geom name="chain stay left" type="capsule" fromto="{self.bottom_bracket[0]-0.02} {-self.wheel_clearance} {self.bottom_bracket[2]}  {self.rear_hub[0]} {self.rear_hub[1]-self.wheel_clearance} {self.rear_hub[2]}" size="0.016"/>
                    <geom name="chain stay right" type="capsule" fromto="{self.bottom_bracket[0]-0.02} {self.wheel_clearance} {self.bottom_bracket[2]}  {self.rear_hub[0]} {self.rear_hub[1]+self.wheel_clearance} {self.rear_hub[2]}" size="0.016"/>
                    <geom name="seat stay left" type="capsule" fromto="{self.rear_hub[0]} {self.rear_hub[1]-self.wheel_clearance} {self.rear_hub[2]}  {self.seat_stay_attachment[0]} {self.seat_stay_attachment[1]} {self.seat_stay_attachment[2]}" size="0.016"/>
                    <geom name="seat stay right" type="capsule" fromto="{self.rear_hub[0]} {self.rear_hub[1]+self.wheel_clearance} {self.rear_hub[2]}  {self.seat_stay_attachment[0]} {self.seat_stay_attachment[1]} {self.seat_stay_attachment[2]}" size="0.016"/>
                    <body name="rear wheel" pos="{self.rear_hub[0]} {self.rear_hub[1]} {self.rear_hub[2]}" euler="90 0 0">
                        <joint name="rear wheel hinge" type="hinge" axis="0 0 -1" pos="0 0 0" limited="false"/>
                        <geom name="rear wheel contact" type="cylinder" size="{self.wheel_size/2} {self.wheel_width/2}" rgba="0.1 0.1 0.1 0" contype="2" conaffinity="2" friction="1.0 0.005 0.0001"/>
                        <geom name="rear wheel geom" type="mesh" mesh="wheel_torus" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>
                    </body>
                </body>
                <body name="front fork" pos="{self.front_hub[0]} {self.front_hub[1]} {self.front_hub[2]}" euler="0 {np.rad2deg(self.fork_angle)} 0">
                    <joint name="steer" type="hinge" axis="0 0 1" pos="0 0 0" range="-35 35"/>
                    <geom name="head tube" type="capsule" fromto="0 0 {self.fork_length} 0 0 {self.fork_length + self.handlebar_height}" size="0.016"/>
                    <geom name="handlebar" type="capsule" fromto="0 {-self.handlebar_width/2} {self.fork_length + self.handlebar_height} 0 {self.handlebar_width/2} {self.fork_length + self.handlebar_height}" size="0.016"/>
                    <site name="left handlebar site" pos="0 {self.handlebar_width/2} {self.fork_length + self.handlebar_height}"/>
                    <site name="right handlebar site" pos="0 {-self.handlebar_width/2} {self.fork_length + self.handlebar_height}"/>
                    <geom name="fork left" type="capsule" fromto="0 {-self.wheel_clearance} 0 0 {-self.wheel_clearance} {self.fork_length}" size="0.016"/>
                    <geom name="fork right" type="capsule" fromto="0 {self.wheel_clearance} 0 0 {self.wheel_clearance} {self.fork_length}" size="0.016"/>
                    <body name="front wheel" pos="0 0 0" euler="90 0 0">
                        <joint name="front wheel hinge" type="hinge" axis="0 0 -1" pos="0 0 0" limited="false"/>
                        <geom name="front wheel contact" type="cylinder" size="{self.wheel_size/2} {self.wheel_width/2}" rgba="0.1 0.1 0.1 0" contype="2" conaffinity="2" friction="1.0 0.005 0.0001"/>
                        <geom name="front wheel geom" type="mesh" mesh="wheel_torus" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>
                    </body>
                </body>
                <body name="pedals" pos="0 0 0">
                    <joint name="pedals" type="hinge" axis="0 1 0" pos="0 0 0" limited="false"/>
                    <geom name="left_crank" type="capsule" fromto="0 -{self.crank_width} 0 0 -{self.crank_width} -{self.crank_length}" size="0.016"/>
                    <geom name="right_crank" type="capsule" fromto="0 {self.crank_width} 0 0 {self.crank_width} {self.crank_length}" size="0.016"/>
                    <body name="left_pedal" pos="0 {-self.crank_width - self.pedal_width/2 - 4*0.016} {-self.crank_length}" euler="180 0 0">
                        <joint name="left_pedal_hinge" type="hinge" axis="0 1 0" pos="0 0 0" limited="false"/>
                        <geom name="left_pedal" type="box" size="{self.pedal_width} {self.pedal_length} {0.01}"/>
                    </body>
                    <body name="right_pedal" pos="0 {self.crank_width + self.pedal_width/2 + 4*0.016} {self.crank_length}">
                        <joint name="right_pedal_hinge" type="hinge" axis="0 1 0" pos="0 0 0" limited="false"/>
                        <geom name="right_pedal" type="box" size="{self.pedal_width} {self.pedal_length} {0.01}"/>
                    </body>
                </body>

            </body>
        </worldbody>"""

        xml_file = f"""
            {wheel_xml}
            <default>
                <joint limited="true" damping="0.01"/>
            </default>

            {frame_xml}
            {equality_xml}
            {actuator_xml}

            <size njmax="100" nconmax="50"/>
        """
        return xml_file
    
    def save_bicycle_model(self, filename):
        with open(filename, "w") as f:
            f.write(self.create_bicycle_model())

if __name__ == "__main__":
    bicycle = Bicycle()
    bicycle.create_bicycle_variables()
    bicycle.create_bicycle_model()
    bicycle.save_bicycle_model("bicycle.xml")