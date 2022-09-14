from colossalai.amp import AMP_TYPE

# BATCH_SIZE = 128
# NUM_EPOCHS = 200

# fp16=dict(
#     mode = AMP_TYPE.NAIVE  
# )

# parallel = dict(
#     pipeline=2,
#     tensor=dict(size=4, mode='2d')
# )

# BATCH_SIZE = 512
# LEARNING_RATE = 2e-3
# WEIGHT_DECAY = 3e-2

# TENSOR_PARALLEL_SIZE = 8
# TENSOR_PARALLEL_MODE = '3d'

# NUM_EPOCHS = 200
# WARMUP_EPOCHS = 40

# parallel = dict(
#     pipeline=1,
#     tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
# )

BATCH_SIZE = 512
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 3e-2

TENSOR_PARALLEL_SIZE = 4
TENSOR_PARALLEL_MODE = '2d'

NUM_EPOCHS = 200
WARMUP_EPOCHS = 40

parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)
