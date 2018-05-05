from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten
# by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    data_dir = '/media/khtao/data/DataSets/pathology_classifer/'
    num_workers = 8
    test_num_workers = 8

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3
    use_adam = True

    # preset
    model_name = 'inceptionv4'
    pretrained_model = 'imagenet'

    # training
    epoch = 40
    train_batch_size = 40
    test_batch_size = 40
    plot_every = 20

    use_drop = True  # use dropout in RoIHead

    test_num = 10000
    # model
    load_path = None

    def _parse(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
