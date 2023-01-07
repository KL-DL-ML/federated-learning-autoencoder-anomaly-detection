best_configs = {}
num_epochs = 100
lr = 1e-3

def get_best_config(algo_name):
    best_configs["USAD"] = {
        "num_epochs": 15,
        "num_hidden": 16,
        "latent": 5,
        "learning_rate": 0.0001,
        "weight_decay": 1e-5,
        "num_window": 15,
    }
    
    best_configs["AE"] = {
        "num_epochs": 15,
        "learning_rate": 0.0001,
        "weight_decay": 1e-5,
        "num_window": 15,
    }

    best_configs["DAGMM"] = {
        "beta": 0.01,
        "embedding_dim": 16,
        "num_epochs": 15,
        "num_hidden": 16,
        "latent": 8,
        "learning_rate": 0.0001,
        "weight_decay": 1e-5,
        "num_window": 15,
    }

    best_configs["MAD_GAN"] = {
        "num_epochs": 15,
        "num_hidden": 16,
        "learning_rate": 0.0001,
        "weight_decay": 1e-5,
        "num_window": 15,
    }

    best_configs["OmniAnomaly"] = {
        "beta": 0.01,
        "num_epochs": 15,
        "num_hidden": 32,
        "latent": 8,
        "learning_rate": 0.002,
        "weight_decay": 1e-5,
    }

    best_configs["MSCRED"] = {
        "num_epochs": 15,
        "learning_rate": 0.0001,
        "weight_decay": 1e-5,
    }

    return best_configs[algo_name]