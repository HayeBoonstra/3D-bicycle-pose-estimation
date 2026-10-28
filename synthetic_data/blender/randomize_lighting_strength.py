import os
import random

import bpy

MIN_STRENGTH = 0.1
MAX_STRENGTH = 1.0


def _resolve_background_node(world):
    """Return the world Background node used by World Output if possible."""
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    output = None
    for node in nodes:
        if node.bl_idname == "ShaderNodeOutputWorld" and node.is_active_output:
            output = node
            break
    if output is None:
        for node in nodes:
            if node.bl_idname == "ShaderNodeOutputWorld":
                output = node
                break
    if output is None:
        output = nodes.new(type="ShaderNodeOutputWorld")
        output.location = (250, 0)

    surface_input = output.inputs.get("Surface")
    if (
        surface_input is not None
        and surface_input.is_linked
        and surface_input.links[0].from_node.bl_idname == "ShaderNodeBackground"
    ):
        return surface_input.links[0].from_node

    for node in nodes:
        if node.bl_idname == "ShaderNodeBackground":
            return node

    background = nodes.new(type="ShaderNodeBackground")
    background.location = (0, 0)
    if surface_input is not None:
        links.new(background.outputs["Background"], surface_input)
    return background


def randomize_world_background_strength(min_strength=MIN_STRENGTH, max_strength=MAX_STRENGTH):
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world

    seed = os.environ.get("LIGHTING_SEED")
    if seed not in {None, ""}:
        random.seed(int(seed))

    background = _resolve_background_node(world)
    sampled_strength = random.uniform(min_strength, max_strength)
    background.inputs["Strength"].default_value = sampled_strength
    os.environ["LIGHTING_STRENGTH"] = f"{sampled_strength:.6f}"
    return sampled_strength


def main():
    sampled_strength = randomize_world_background_strength()


if __name__ == "__main__":
    main()

