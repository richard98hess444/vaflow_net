class Config:
    """Configuration for training the neural vector-field model."""
    def __init__(self):
        self.grid_size = 21
        self.obstacle_prob = 0.2
        self.threshold = 0.05
        self.num_samples = 200
        self.val_fraction = 0.2
        self.num_queries = 16
        self.batch_size = 16
        self.epochs = 500
        self.lr = 1e-3
        self.latent_dim = 64
        self.hidden_dim = 128
        self.seed = 0