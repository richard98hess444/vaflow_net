class Config:
    """Configuration for training the neural vector-field model."""

    def __init__(self):
        self.grid_size = 21
        self.obstacle_prob = 0.2
        self.threshold = 0.05
        self.num_samples = 20000
        self.val_fraction = 0.2
        self.batch_size = 16
        self.epochs = 100
        self.lr = 1e-3
        self.model_channels = 32
        self.seed = 0
        
        self.check_reachable = True
        self.use_dataset_cache = True  # generate once, then reuse from disk
        self.dataset_cache_dir = "training/cache"
