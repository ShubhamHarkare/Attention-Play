from pathlib import Path

class TrainingConfig:
    '''
    All configurations required for the models to train
    '''

    DATA_DIR = Path('output/data')
    OUTPUT_DIR = Path('output')
    MODEL_DIR = OUTPUT_DIR / "models_greatlakes"
    VIZ_DIR = OUTPUT_DIR / "visualizations"

    #? Model Hyperparameters - Enhanced for better performance
    EMBEDDING_DIM = 256  # Increased from 128 for richer representations
    HIDDEN_DIM = 512     # Increased from 256 for more capacity
    NUM_HEADS = 8        # Good for 256-dim embeddings (256/8=32 per head)
    NUM_LAYERS = 4       # Increased from 2 for deeper model
    DROPOUT = 0.3        # Reduced from 0.5 to prevent underfitting
    MAX_SEQ_LENGTH = 50

    #? Training hyperparameters - Optimized
    BATCH_SIZE = 128     # Increased for more stable gradients
    LEARNING_RATE = 0.0005  # Reduced for better convergence
    NUM_EPOCHS = 50      # More epochs with early stopping
    PATIENCE = 8         # Early stopping patience
    WARMUP_EPOCHS = 3    # Learning rate warmup
    LABEL_SMOOTHING = 0.1  # Label smoothing for regularization


    TOP_K_VALUES = [1,5,10,20]

    #? System
    NUM_WORKERS = 4
    RANDOM_SEED = 42

    def __init__(self):
        """Initialize output directories"""
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.VIZ_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)