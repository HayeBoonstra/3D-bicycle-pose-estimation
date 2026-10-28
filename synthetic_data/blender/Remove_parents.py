# this script is used to remove the annoying parenting of imported bicycles

import bpy

BICYCLE_NAME = "Bicycle"


def collect_descendants(obj):
    """Return every descendant object under obj."""
    descendants = []
    for child in obj.children:
        descendants.append(child)
        descendants.extend(collect_descendants(child))
    return descendants


def parent_depth(obj):
    """Return hierarchy depth by walking up parents."""
    depth = 0
    current = obj.parent
    while current is not None:
        depth += 1
        current = current.parent
    return depth


def unparent_and_rename_from_parent(root_name):
    root = bpy.data.objects.get(root_name)
    if root is None:
        raise ValueError(f"Object '{root_name}' was not found in this scene.")

    descendants = collect_descendants(root)
    if not descendants:
        print(f"No descendants found under '{root_name}'.")
        return

    # Process deepest nodes first so parent references are still valid when needed.
    descendants.sort(key=parent_depth, reverse=True)

    renamed_count = 0
    for obj in descendants:
        parent = obj.parent
        if parent is None:
            continue

        # Keep the current world transform before removing parenting.
        world_matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world_matrix

        obj.name = parent.name
        renamed_count += 1

    print(f"Unparented and renamed {renamed_count} objects under '{root_name}'.")

def set_origin_to_object_all():
    """Set origin to geometry for all objects that support it."""
    view_layer = bpy.context.view_layer
    original_active = view_layer.objects.active
    original_selection = list(bpy.context.selected_objects)

    eligible_types = {"MESH", "CURVE", "SURFACE", "META", "FONT", "GPENCIL"}
    changed_count = 0

    for obj in bpy.context.scene.objects:
        if obj.type not in eligible_types:
            continue

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        view_layer.objects.active = obj

        try:
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
            changed_count += 1
        except RuntimeError:
            # Skip objects that cannot be processed in current context.
            continue

    bpy.ops.object.select_all(action="DESELECT")
    for obj in original_selection:
        if obj.name in bpy.data.objects:
            obj.select_set(True)
    if original_active and original_active.name in bpy.data.objects:
        view_layer.objects.active = original_active

    print(f"Set origin to geometry for {changed_count} objects.")


# unparent_and_rename_from_parent(BICYCLE_NAME)
set_origin_to_object_all()
