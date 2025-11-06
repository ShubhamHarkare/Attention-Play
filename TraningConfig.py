from pathlib import Path

class TrainingConfig:
    '''
    All configurations required for the models to train
    '''

    DATA_DIR = Path('output/data')
    OUTPUT_DIR = Path('output')
    MODEL_DIR = OUTPUT_DIR / "models"
    VIZ_DIR = OUTPUT_DIR / "visualizations"

    #? Model Hyperparameters
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 2
    DROPOUT = 0.5
    MAX_SEQ_LENGTH = 50

    #? Training hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30
    PATIENCE = 10


    TOP_K_VALUES = [1,5,10,20]

    #? System
    NUM_WORKERS = 4
    RANDOM_SEED = 4