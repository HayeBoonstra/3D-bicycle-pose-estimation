import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)

while True:
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.01)