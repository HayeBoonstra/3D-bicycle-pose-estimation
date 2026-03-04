import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("bicycle.xml")
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)

while True:
    mujoco.mj_step(model, data)
    viewer.sync()