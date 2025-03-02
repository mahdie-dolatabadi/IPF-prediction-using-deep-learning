import argparse

class HyperParameters:
    def __init__(self, model_type):
        parser = argparse.ArgumentParser(description="Hyperparameter Configuration")
        parser.add_argument("--seed", type=int, default=2024)
        parser.add_argument("--data_folder", type=str, default="..")
        parser.add_argument("--ct_tab_feature_csv", type=str, default="train_data_ct_tab.csv")
        parser.add_argument("--strip_ct", type=float, default=0.35)
        parser.add_argument("--n_tab", type=int, default=20)
        parser.add_argument("--cnn_dim_s", type=int, default=768)
        parser.add_argument("--cnn_dim", type=int, default=1000)
        parser.add_argument("--fc_dim", type=int, default=64)
        parser.add_argument("--train_models", type=str, nargs="+", default=["vit_b_16"])
        parser.add_argument("--gpu_index", type=int, default=0)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--results_dir", type=str, default="results_slopes")
        parser.add_argument("--nfold", type=int, default=5)
        parser.add_argument("--n_epochs", type=int, default=5)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--final_lr", type=float, default=0.0002)
        
        args, _ = parser.parse_known_args()
        
        if model_type == "slope_train_vit_simple_without_tab":
            args.n_tab = 4
        elif model_type == "slope_train_vit_simple_without_cnn":
            args.cnn_dim = 0
        elif model_type == "slope_train_vit_simple_without_vit":
            args.cnn_dim_s = 0
        
        self.__dict__.update(vars(args))


# params = HyperParameters("slope_train_vit_simple")
# print(params.n_epochs)  # Example access
