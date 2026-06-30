class Config:
    """Configuration for training the neural vector-field model."""
    def __init__(self):
        self.grid_size = 21
        self.obstacle_prob = 0.2
        self.threshold = 0.05
        self.num_samples = 200
        self.val_fraction = 0.2
        self.batch_size = 16
        self.epochs = 500
        self.lr = 1e-3
        self.model_channels = 32
        self.seed = 0