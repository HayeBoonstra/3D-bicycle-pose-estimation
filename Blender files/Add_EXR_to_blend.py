import bpy
import os
# Set this to your EXR file path. for use within blender interface.
BLENDER_HDRIS_DIR = "~/home/" + os.getenv("USER") + "/3D-bicycle-pose-estimation/Blender files/HDRIs/"
EXR_PATH = BLENDER_HDRIS_DIR + "modern_evening_street_4k.exr"

# Optional controls.
BACKGROUND_STRENGTH = 1.0
SCENE_EXPOSURE = 0.0


def set_exr_world_background(exr_path: str, strength: float = 1.0) -> None:
    scene = bpy.context.scene

    # Ensure a World datablock exists and is used by the scene.
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")

    world = scene.world
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    # Clean existing nodes for a predictable setup.
    nodes.clear()

    out = nodes.new(type="ShaderNodeOutputWorld")
    out.location = (250, 0)

    bg = nodes.new(type="ShaderNodeBackground")
    bg.location = (0, 0)
    bg.inputs["Strength"].default_value = strength

    env = nodes.new(type="ShaderNodeTexEnvironment")
    env.location = (-300, 0)
    env.image = bpy.data.images.load(exr_path, check_existing=True)

    links.new(env.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    # Make sure the world background is visible and contributes light.
    scene.render.film_transparent = False


def main() -> None:
    set_exr_world_background(EXR_PATH, BACKGROUND_STRENGTH)
    bpy.context.scene.view_settings.exposure = SCENE_EXPOSURE
    print(f"World EXR applied: {EXR_PATH}")


if __name__ == "__main__":
    main()
