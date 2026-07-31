class Config:
    """Configuration for simulating the neural vector-field model."""
    def __init__(self):

        self.local_size = 61
        self.world_size = 256
        self.start_pose = (self.local_size * 2, self.local_size * 2)
        self.max_steps = 300