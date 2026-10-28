import mujoco
from bicycle_constructor import Bicycle

class World:
    def __init__(self):
        bicycle = Bicycle()
        bicycle.create_bicycle_variables()
        bicycle.create_bicycle_model()
        bicycle.save_bicycle_model("bicycle.xml")
    
    def create_world(self):
        world_xml = f"""
            <mujoco model="bicycle world">
            <compiler angle="degree" coordinate="local"/>
            <option timestep="0.005" gravity="0 0 -9.81" noslip_iterations="15"/>
            <include file="bicycle.xml"/>

            <statistic center="0 0 0.55" extent="1.1"/>

            <visual>
                <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
                <rgba haze="0.15 0.25 0.35 1"/>
                <global azimuth="150" elevation="-20"/>
            </visual>

            <asset>
                <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
                <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
                markrgb="0.8 0.8 0.8" width="300" height="300"/>
                <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
            </asset>

            <worldbody>
                <light pos="0 0 3" dir="0 0 -1" directional="true" diffuse="1.5 1.5 1.5"/>
                <geom name="floor" size="0 0 .125" type="plane" material="groundplane" contype="3" conaffinity="15" condim="3"/>
            </worldbody>
            </mujoco>"""
        return world_xml
    
    def save_world(self, filename):
        with open(filename, "w") as f:
            f.write(self.create_world())