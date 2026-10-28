import os
import random

import bpy

FOG_OBJECT_NAME = "Fog Cube"
MIN_DENSITY = 0.0
MAX_DENSITY = 0.02
NO_FOG_PROBABILITY = 0.5


def _seed_rng() -> None:
    seed_raw = os.environ.get("CAMERA_SEED")
    if seed_raw not in {None, ""}:
        # Offset the seed so fog randomness is deterministic but independent
        # from other randomization scripts that also use CAMERA_SEED.
        random.seed(int(seed_raw) + 100_003)


def _ensure_principled_volume(material: bpy.types.Material) -> bpy.types.ShaderNodeVolumePrincipled:
    node_tree = material.node_tree
    if node_tree is None:
        raise RuntimeError(
            "Fog material has no node tree. Create/assign a node-based fog material first."
        )

    output_node = None
    volume_node = None
    for node in node_tree.nodes:
        if node.type == "OUTPUT_MATERIAL":
            output_node = node
        elif node.type == "VOLUME_PRINCIPLED":
            volume_node = node

    if output_node is None:
        output_node = node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    if volume_node is None:
        volume_node = node_tree.nodes.new(type="ShaderNodeVolumePrincipled")

    # Ensure the volume output is connected to the material volume input.
    volume_input = output_node.inputs.get("Volume")
    if volume_input is None:
        raise RuntimeError("Material output node has no Volume input.")
    if not any(link.from_node == volume_node for link in volume_input.links):
        for link in list(volume_input.links):
            node_tree.links.remove(link)
        node_tree.links.new(volume_node.outputs["Volume"], volume_input)

    return volume_node


def main() -> None:
    _seed_rng()

    fog_obj = bpy.data.objects.get(FOG_OBJECT_NAME)
    if fog_obj is None:
        raise RuntimeError(f'Fog object "{FOG_OBJECT_NAME}" was not found in the scene.')
    if fog_obj.type != "MESH":
        raise RuntimeError(f'Fog object "{FOG_OBJECT_NAME}" is not a mesh object.')

    if fog_obj.active_material is not None:
        fog_material = fog_obj.active_material
    elif fog_obj.data.materials:
        fog_material = fog_obj.data.materials[0]
    else:
        fog_material = bpy.data.materials.new(name="Fog Volume Material")
        fog_obj.data.materials.append(fog_material)

    volume_node = _ensure_principled_volume(fog_material)
    if random.random() < NO_FOG_PROBABILITY:
        density = 0.0
    else:
        density = random.uniform(MIN_DENSITY, MAX_DENSITY)
    volume_node.inputs["Density"].default_value = float(density)

    # Remove animation/drivers that might override the sampled value.
    if fog_material.node_tree and fog_material.node_tree.animation_data is not None:
        fog_material.node_tree.animation_data_clear()

    fog_material.update_tag()
    bpy.context.view_layer.update()


if __name__ == "__main__":
    main()
