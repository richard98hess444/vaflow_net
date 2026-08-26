class Config:
    """Configuration for training the neural vector-field model."""

    def __init__(self):
        self.grid_size = 61
        self.obstacle_prob = 0.2
        self.threshold = 0.05
        self.num_samples = 20000
        self.val_fraction = 0.2
        self.batch_size = 8
        self.epochs = 100
        self.lr = 1e-3
        self.model_channels = 32
        self.seed = 0

        self.use_amp = True  # mixed-precision training on CUDA

        self.check_reachable = True
        self.use_dataset_cache = True  # generate once, then reuse from disk

        self.query_points = 256
        self.query_loss_weight = 1.0
        self.query_fourier_bands = 6

        # 2D or 3D pipeline. Selects the dataset/model/predict fn.
        self.dataset = "3d"

        # Ground-truth guidance method: "eik", "bfs", or "ns" (Navier-Stokes drag).
        self.method = "ns"

        self.dataset_dir_2d = "cache/dataset_2d"
        self.dataset_dir_3d = "cache/dataset_3d"
        self.cache_chunks = 1
        self.chunk_size = 1000  # samples per on-disk chunk (3D generation)
        self.gpu_batch_size = 128  # method="ns": GPU sub-batch size (None = auto)
        self.num_workers = 4
        self.pin_memory = True

    def dataset_subdir(self, dataset_key=None):
        """Per-config folder name that uniquely identifies this dataset.

        Encodes every dataset-defining value so different configs never share a
        directory (e.g. ``flow_ds_3d_ns_n61_p0.2_s200_seed0_reach1``). Both the
        Trainer and ``scripts/generate_dataset_3d.py`` build the dataset path
        from this, so generated chunks land exactly where training looks for
        them. ``dataset_key`` overrides ``self.dataset`` (the generate script
        passes ``"3d"`` explicitly since it only produces 3D data).
        """
        key = self.dataset if dataset_key is None else dataset_key
        return (
            f"flow_ds_{key}_{self.method}_n{self.grid_size}_p{self.obstacle_prob}"
            f"_s{self.num_samples}_seed{self.seed}"
            f"_reach{int(self.check_reachable)}"
        )
