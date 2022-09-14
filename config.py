from colossalai.amp import AMP_TYPE

BATCH_SIZE = 128
NUM_EPOCHS = 200

fp16=dict(
    mode=AMP_TYPE.TORCH
)

parallel = dict(
    pipeline=2,
    tensor=dict(size=4, mode='2d')
)
