import bpy

# Reuse existing fog cube if present; otherwise create it.
fog_cube = bpy.data.objects.get("Fog Cube")
if fog_cube is None:
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.0))
    fog_cube = bpy.context.active_object
    fog_cube.name = "Fog Cube"
else:
    fog_cube.location = (0.0, 0.0, 0.0)

fog_cube.scale = (50.0, 50.0, 50.0)

# Reuse existing material if present so reruns update the same datablock.
fog_material = bpy.data.materials.get("Fog Volume Material")
if fog_material is None:
    fog_material = bpy.data.materials.new(name="Fog Volume Material")
nodes = fog_material.node_tree.nodes
links = fog_material.node_tree.links

nodes.clear()
output_node = nodes.new(type="ShaderNodeOutputMaterial")
volume_node = nodes.new(type="ShaderNodeVolumePrincipled")
volume_node.inputs["Density"].default_value = 0.0

links.new(volume_node.outputs["Volume"], output_node.inputs["Volume"])

# Remove old animation/drivers that could override density on frame change.
if fog_material.node_tree.animation_data is not None:
    fog_material.node_tree.animation_data_clear()

fog_cube.data.materials.clear()
fog_cube.data.materials.append(fog_material)
fog_cube.active_material = fog_material

# Force dependency graph refresh so the value updates immediately.
fog_material.update_tag()
bpy.context.view_layer.update()

