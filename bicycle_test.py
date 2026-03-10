import mujoco
import mujoco.viewer
import time
import numpy as np

model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)

t = 0.1
data.qvel[0] = 3
while True:
    t += 0.3
    mujoco.mj_step(model, data)
    viewer.sync()
    data.ctrl[0] = 10
    data.ctrl[1] = np.sin(t) * 4
    time.sleep(0.01)