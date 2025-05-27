from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import get_world_transform_matrix

mesh_prim_path = "/World/MyObject/MyXform/Mesh"
mesh_prim = get_prim_at_path(mesh_prim_path)

world_transform = get_world_transform_matrix(mesh_prim)
world_position = world_transform[:3, 3]
print("Mesh world position:", world_position)

