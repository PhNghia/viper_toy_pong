def get_oracle_path(args):
    if args.oracle_path is not None:
        return args.oracle_path
    return "./log/oracle_" + args.log_prefix  + args.env_name


def get_viper_path(args):
    ccp_alpha = str(args.ccp_alpha) if args.ccp_alpha is not None else "0.0001"
    n_leaves = str(args.max_leaves) if args.max_leaves is not None else "all-leaves"
    max_depth = str(args.max_depth) if args.max_depth is not None else "all-depth"
    return "./log/viper_" + args.log_prefix + args.env_name + "_" + ccp_alpha + "_" + n_leaves + "_" + max_depth + ".joblib"

def get_viper_dataset_path(args):
    ccp_alpha = str(args.ccp_alpha) if args.ccp_alpha is not None else "0.0001"
    n_leaves = str(args.max_leaves) if args.max_leaves is not None else "all-leaves"
    max_depth = str(args.max_depth) if args.max_depth is not None else "all-depth"
    return "./dataset/viper_" + args.log_prefix + args.env_name + "_" + ccp_alpha + "_" + n_leaves + "_" + max_depth + ".joblib"

def get_viper_pruned_path(ccp_alpha, max_depth=None, max_leaves=None):
    ccp_alpha = str(ccp_alpha) if ccp_alpha is not None else "0.0001"
    n_leaves = str(max_leaves) if max_leaves is not None else "all-leaves"
    max_depth = str(max_depth) if max_depth is not None else "all-depth"
    return "./log/viper_ToyPong-v0_pruned_" + ccp_alpha + "_" + n_leaves + "_" + max_depth + ".joblib"

def get_viper_pruned_path_2(ccp_alpha, max_depth=None, max_leaves=None):
    ccp_alpha = str(ccp_alpha) if ccp_alpha is not None else "0.0001"
    n_leaves = str(max_leaves) if max_leaves is not None else "all-leaves"
    max_depth = str(max_depth) if max_depth is not None else "all-depth"
    return "./log/viper_ToyPong-v0_pruned_2_" + ccp_alpha + "_" + n_leaves + "_" + max_depth + ".joblib"