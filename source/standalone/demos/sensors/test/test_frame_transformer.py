from omni.isaac.orbit.utils.frame_transformer import FrameTransformer

# Initialize the frame transformer
ft = FrameTransformer()

# Define the prims to track
# These MUST be the paths to the prims with the RigidBodyAPI
transforms_to_track = {
    "object_1": "/World/MyAssembledObject/rigid_body_1",
    "object_2": "/World/MyAssembledObject/rigid_body_2",
    "object_3": "/World/MyAssembledObject/rigid_body_3"
}

# Add the transforms to the tracker
ft.add_transforms(list(transforms_to_track.keys()), list(transforms_to_track.values()))

# In your loop, you can now get their individual poses
ft.update()
pos, rot = ft.get_transform("object_1", "world")