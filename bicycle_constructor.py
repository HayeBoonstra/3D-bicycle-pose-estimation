import numpy as np

class Bicycle:
    def __init__(self):
        self.wheel_size = 28 * 0.0254
        self.wheel_width = 0.03
        self.frame_size = 0.55 # m (55 cm is standard frame size)
        self.seat_tube_angle = np.deg2rad(8) # degrees
        self.wheel_base = 1.05 # between 1.0 and 1.1 meters
        self.hub_raise = 0.03
        self.fork_angle = np.deg2rad(-20) # degrees
        self.fork_height = 0.0
        self.fork_length = 0.0
        self.handlebar_height = 0.1 # m
        self.handlebar_width = 0.3 # m
        self.seat_tube_height = 0.0
        self.bottom_bracket = [0.0, 0.0, 0.0]
        self.rear_hub = [0.0, 0.0, 0.0]
        self.front_hub = [0.0, 0.0, 0.0]
        self.head_tube = [0.0, 0.0, 0.0]
        self.seat_tube = [0.0, 0.0, 0.0]
        self.wheel_clearance = 0.02
    
    def create_bicycle_variables(self):
        ## fork geometry
        # X forward, Y left, Z up
        # the end point of the fork is dependent on the frame size and fork angle.
        # the end of the fork should be as high as the seat tube height (or frame size)
        self.seat_tube_height = self.frame_size * np.sin(self.seat_tube_angle)
        self.fork_height = self.seat_tube_height
        self.fork_length = self.frame_size * np.cos(self.seat_tube_angle)

        ## frame geometry
        self.bottom_bracket = np.array([0.0, 0.0, 0.0])
        self.rear_hub = self.bottom_bracket + np.array([-0.4, 0.0, self.hub_raise])
        self.front_hub = self.bottom_bracket + np.array([self.rear_hub[0] + self.wheel_base, 0.0, self.hub_raise])

        fork_dir = np.array([np.sin(self.fork_angle), 0.0, np.cos(self.fork_angle)])
        self.head_tube = self.front_hub + self.fork_length * fork_dir
        seat_tube_dir = np.array([-np.sin(self.seat_tube_angle), 0.0, np.cos(self.seat_tube_angle)])
        self.seat_tube = self.bottom_bracket + self.frame_size * seat_tube_dir

        

    def create_bicycle_model(self):
        ## create the mujoco XML file
        wheel_xml = f"""
        <asset>
            <mesh name="wheel_torus" builtin="supertorus" params="64 0.025 1 1" scale="0.35 0.35 0.35"/>
        </asset>
        """

        actuator_xml = """
        <actuator>
            <motor joint="rear wheel hinge"/>
            <motor joint="steer"/>
        </actuator>
        """

        frame_xml = f"""
        <worldbody>
            <body name="bicycle" pos="0 0 0.35">
                <freejoint name="bicycle_free"/>
                <body name="frame" pos="0 0 0">
                    <geom name="seat tube" type="capsule" fromto="{self.bottom_bracket[0]} {self.bottom_bracket[1]} {self.bottom_bracket[2]}  {self.seat_tube[0]} {self.seat_tube[1]} {self.seat_tube[2]}" size="0.016"/>
                    <geom name="down tube" type="capsule" fromto="{self.bottom_bracket[0]} {self.bottom_bracket[1]} {self.bottom_bracket[2]}  {self.head_tube[0]} {self.head_tube[1]} {self.head_tube[2]}" size="0.016"/>
                    <geom name="top tube" type="capsule" fromto="{self.head_tube[0]} {self.head_tube[1]} {self.head_tube[2]}  {self.seat_tube[0]} {self.seat_tube[1]} {self.seat_tube[2]}" size="0.016"/>
                    <geom name="chain stay left" type="capsule" fromto="{self.bottom_bracket[0]-0.02} {-self.wheel_clearance} {self.bottom_bracket[2]}  {self.rear_hub[0]} {self.rear_hub[1]-self.wheel_clearance} {self.rear_hub[2]}" size="0.016"/>
                    <geom name="chain stay right" type="capsule" fromto="{self.bottom_bracket[0]-0.02} {self.wheel_clearance} {self.bottom_bracket[2]}  {self.rear_hub[0]} {self.rear_hub[1]+self.wheel_clearance} {self.rear_hub[2]}" size="0.016"/>
                    <geom name="seat stay left" type="capsule" fromto="{self.rear_hub[0]} {self.rear_hub[1]-self.wheel_clearance} {self.rear_hub[2]}  {self.seat_tube[0]} {self.seat_tube[1]} {self.seat_tube[2]}" size="0.016"/>
                    <geom name="seat stay right" type="capsule" fromto="{self.rear_hub[0]} {self.rear_hub[1]+self.wheel_clearance} {self.rear_hub[2]}  {self.seat_tube[0]} {self.seat_tube[1]} {self.seat_tube[2]}" size="0.016"/>
                    <body name="rear wheel" pos="{self.rear_hub[0]} {self.rear_hub[1]} {self.rear_hub[2]}" euler="90 0 0">
                        <joint name="rear wheel hinge" type="hinge" axis="0 0 -1" pos="0 0 0" limited="false"/>
                        <geom name="rear wheel geom" type="mesh" mesh="wheel_torus" rgba="0.1 0.1 0.1 1" contype="2" conaffinity="2"/>
                    </body>
                </body>
                <body name="front fork" pos="{self.front_hub[0]} {self.front_hub[1]} {self.front_hub[2]}" euler="0 {np.rad2deg(self.fork_angle)} 0">
                    <joint name="steer" type="hinge" axis="0 0 1" pos="0 0 0" range="-35 35"/>
                    <geom name="head tube" type="capsule" fromto="0 0 {self.fork_length} 0 0 {self.fork_length + self.handlebar_height}" size="0.016"/>
                    <geom name="handlebar" type="capsule" fromto="0 {-self.handlebar_width/2} {self.fork_length + self.handlebar_height} 0 {self.handlebar_width/2} {self.fork_length + self.handlebar_height}" size="0.016"/>
                    <geom name="fork left" type="capsule" fromto="0 {-self.wheel_clearance} 0 0 {-self.wheel_clearance} {self.fork_length}" size="0.016"/>
                    <geom name="fork right" type="capsule" fromto="0 {self.wheel_clearance} 0 0 {self.wheel_clearance} {self.fork_length}" size="0.016"/>
                    <body name="front wheel" pos="0 0 0" euler="90 0 0">
                        <joint name="front wheel hinge" type="hinge" axis="0 0 -1" pos="0 0 0" limited="false"/>
                        <geom name="front wheel geom" type="mesh" mesh="wheel_torus" rgba="0.1 0.1 0.1 1" contype="2" conaffinity="2"/>
                    </body>
                </body>
            </body>
        </worldbody>"""

        xml_file = f"""
        <mujoco model="bicycle">
            <compiler angle="degree" coordinate="local"/>
            <option timestep="0.01" gravity="0 0 -9.81"/>
            {wheel_xml}
            <default>
                <joint limited="true" damping="0.01"/>
                <geom friction="0.8 0.1 0.1" rgba="0.8 0.6 0.4 1"/>
            </default>

            {frame_xml}
            {actuator_xml}

            <size njmax="100" nconmax="50"/>
        </mujoco>
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