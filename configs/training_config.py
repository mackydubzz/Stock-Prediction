# configs/training_config.py
model_configs = {
    'rnn': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 1000,  # as per paper
        'patience': 50
    },
    'lstm': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 1000,
        'patience': 50
    },
    'cnn': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 1000,
        'patience': 50
    }
}
