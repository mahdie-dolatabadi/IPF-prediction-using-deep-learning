class Parameters:
    def __init__(self, model_type):

        if model_type == "slope_train_vit_simple":
            self.seed = 2024
            self.data_folder = '..' # one level up
            self.ct_tab_feature_csv = 'train_data_ct_tab.csv' # some extra features
            self.strip_ct = 0.35 # strip this amount of ct slices before randomly choosing
            self.n_tab = 20 # number of tabular features used

            self.cnn_dim_s = 768
            self.cnn_dim = 1000
            self.fc_dim = 64 # 20 40 32 16 128

            # select which models to train
            self.train_models = ['vit_b_16'] # 'vit-base-patch16-224-in21k_lung_and_colon_cancer', 'resnet152', resnext101, efnb3
            # ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101', 'efnb0', 'efnb1', 'efnb2', 'efnb3', 'efnb4', 'efnb5', 'efnb6', 'efnb7'] 
            self.gpu_index = 0
            self.num_workers = 0 # 0 for bug fix/docker
            self.results_dir = "results_slopes"
            self.nfold = 5
            self.n_epochs = 5 # 100
            self.batch_size = 8
            self.final_lr = 0.0002

        elif model_type == "slope_train_vit_simple_without_tab":
            self.seed = 2024
            self.data_folder = '..' # one level up
            self.ct_tab_feature_csv = 'train_data_ct_tab.csv' # some extra features
            self.strip_ct = 0.35 # strip this amount of ct slices before randomly choosing
            self.n_tab = 4 # number of tabular features used

            self.cnn_dim_s = 768
            self.cnn_dim = 1000
            self.fc_dim = 64 # 20 40 32 16 128

            # select which models to train
            self.train_models = ['vit_b_16'] # 'vit-base-patch16-224-in21k_lung_and_colon_cancer', 'resnet152', resnext101, efnb3
            # ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101', 'efnb0', 'efnb1', 'efnb2', 'efnb3', 'efnb4', 'efnb5', 'efnb6', 'efnb7'] 
            self.gpu_index = 0
            self.num_workers = 0 # 0 for bug fix/docker
            self.results_dir = "results_slopes"
            self.nfold = 5
            self.n_epochs = 5 # 100
            self.batch_size = 8
            self.final_lr = 0.0002

        elif model_type == "slope_train_vit_simple_without_cnn":
            self.seed = 2024
            self.data_folder = '..' # one level up
            self.ct_tab_feature_csv = 'train_data_ct_tab.csv' # some extra features
            self.strip_ct = 0.35 # strip this amount of ct slices before randomly choosing
            self.n_tab = 20 # number of tabular features used

            self.cnn_dim_s = 768
            self.cnn_dim = 0
            self.fc_dim = 64 # 20 40 32 16 128

            # select which models to train
            self.train_models = ['vit_b_16'] # 'vit-base-patch16-224-in21k_lung_and_colon_cancer', 'resnet152', resnext101, efnb3
            # ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101', 'efnb0', 'efnb1', 'efnb2', 'efnb3', 'efnb4', 'efnb5', 'efnb6', 'efnb7'] 
            self.gpu_index = 0
            self.num_workers = 0 # 0 for bug fix/docker
            self.results_dir = "results_slopes"
            self.nfold = 5
            self.n_epochs = 5 # 100
            self.batch_size = 8
            self.final_lr = 0.0002

        elif model_type == "slope_train_vit_simple_without_vit":
            self.seed = 2024
            self.data_folder = '..' # one level up
            self.ct_tab_feature_csv = 'train_data_ct_tab.csv' # some extra features
            self.strip_ct = 0.35 # strip this amount of ct slices before randomly choosing
            self.n_tab = 20 # number of tabular features used

            self.cnn_dim_s = 0
            self.cnn_dim = 1000
            self.fc_dim = 64 # 20 40 32 16 128

            # select which models to train
            self.train_models = ['vit_b_16'] # 'vit-base-patch16-224-in21k_lung_and_colon_cancer', 'resnet152', resnext101, efnb3
            # ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101', 'efnb0', 'efnb1', 'efnb2', 'efnb3', 'efnb4', 'efnb5', 'efnb6', 'efnb7'] 
            self.gpu_index = 0
            self.num_workers = 0 # 0 for bug fix/docker
            self.results_dir = "results_slopes"
            self.nfold = 5
            self.n_epochs = 5 # 100
            self.batch_size = 8
            self.final_lr = 0.0002

        elif model_type == "slope_train_vit_simple24April":
            self.seed = 1997
            self.data_folder = '..' # one level up
            self.ct_tab_feature_csv = 'train_data_ct_tab.csv' # some extra features
            self.strip_ct = 0.35 # strip this amount of ct slices before randomly choosing
            self.n_tab = 20 # number of tabular features used

            self.cnn_dim_s = 768
            self.cnn_dim = 1000
            self.fc_dim = 64 # 20 40 32 16 128

            # select which models to train
            self.train_models = ['vit_b_16'] # 'vit-base-patch16-224-in21k_lung_and_colon_cancer', 'resnet152', resnext101, efnb3
            # ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101', 'efnb0', 'efnb1', 'efnb2', 'efnb3', 'efnb4', 'efnb5', 'efnb6', 'efnb7'] 
            self.gpu_index = 0
            self.num_workers = 0 # 0 for bug fix/docker
            self.results_dir = "results_slopes"
            self.nfold = 5
            self.n_epochs = 2 # 100
            self.batch_size = 2
            self.final_lr = 0.0002
