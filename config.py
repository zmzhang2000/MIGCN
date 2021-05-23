class Config:
    # override default hyper-parameters using user arguments
    def __init__(self, user_args):
        for k, v in user_args.items():
            if v is not None:
                if k == "window_widths":
                    setattr(self, k, tuple(v))
                else:
                    setattr(self, k, v)

        print('user config: ')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


class ActivitynetC3D(Config):
    def __init__(self, user_args):
        self.dataset = "activitynet"
        self.feature = "c3d"
        self.feature_path = "data/activitynet/activitynet.c3d.hdf5"
        self.train_path = "data/activitynet/train.pkl"
        self.val_path = "data/activitynet/val.pkl"
        self.model_load_path = None

        self.epochs = 30
        self.batch_size = 128
        self.lr = 0.0003
        self.weight_decay = 0.00001
        self.dropout = 0.5
        self.alpha = 1
        self.beta = 1e-3

        self.frame_feature_dim = 500
        self.word_feature_dim = 300
        self.node_dim = 512
        self.max_frames_num = 300
        self.max_words_num = 50
        self.gcn_layers_num = 1
        self.window_widths = (6, 12, 24, 48, 96, 192, 288)
        self.window_stride = 3
        super().__init__(user_args)


class CharadesC3D(Config):
    def __init__(self, user_args):
        self.dataset = "charades"
        self.feature = "c3d"
        self.feature_path = "data/charades/charades.c3d.hdf5"
        self.train_path = "data/charades/train.pkl"
        self.val_path = "data/charades/val.pkl"
        self.model_load_path = None

        self.epochs = 30
        self.batch_size = 128
        self.lr = 0.001
        self.weight_decay = 0.00001
        self.dropout = 0.5
        self.alpha = 1e-1
        self.beta = 1e-3

        self.frame_feature_dim = 4096
        self.word_feature_dim = 300
        self.node_dim = 512
        self.max_frames_num = 75
        self.max_words_num = 10
        self.gcn_layers_num = 1
        self.window_widths = (6, 12, 18, 24, 30, 36)
        self.window_stride = 3
        super().__init__(user_args)


class CharadesI3D(Config):
    def __init__(self, user_args):
        self.dataset = "charades"
        self.feature = "i3d"
        self.feature_path = "data/charades/charades.i3d.hdf5"
        self.train_path = "data/charades/train.pkl"
        self.val_path = "data/charades/val.pkl"
        self.model_load_path = None

        self.epochs = 30
        self.batch_size = 128
        self.lr = 0.001
        self.weight_decay = 0.00001
        self.dropout = 0.5
        self.alpha = 1e-1
        self.beta = 1e-3

        self.frame_feature_dim = 1024
        self.word_feature_dim = 300
        self.node_dim = 256
        self.max_frames_num = 75
        self.max_words_num = 10
        self.gcn_layers_num = 1
        self.window_widths = (6, 12, 18, 24, 30, 36)
        self.window_stride = 3
        super().__init__(user_args)


class CharadesTwostream(Config):
    def __init__(self, user_args):
        self.dataset = "charades"
        self.feature = "two-stream"
        self.feature_path = "data/charades/charades.two-stream.hdf5"
        self.train_path = "data/charades/train.pkl"
        self.val_path = "data/charades/val.pkl"
        self.model_load_path = None

        self.epochs = 30
        self.batch_size = 128
        self.lr = 0.0003
        self.weight_decay = 0.00001
        self.dropout = 0.5
        self.alpha = 1e-1
        self.beta = 1e-3

        self.frame_feature_dim = 8192
        self.word_feature_dim = 300
        self.node_dim = 512
        self.max_frames_num = 75
        self.max_words_num = 10
        self.gcn_layers_num = 1
        self.window_widths = (6, 12, 18, 24, 30, 36)
        self.window_stride = 3
        super().__init__(user_args)


config = {
    ('activitynet', 'c3d'): ActivitynetC3D,
    ('charades', 'c3d'): CharadesC3D,
    ('charades', 'i3d'): CharadesI3D,
    ('charades', 'two-stream'): CharadesTwostream
}


def get_config(user_args):
    return config[(user_args.dataset, user_args.feature)](user_args.__dict__)
