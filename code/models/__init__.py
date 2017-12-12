from .abstract_model import AbstractModel
from .resnet import ResNet
from .tf_utils.layers import ResidualBlockKind

def get_wide_resnet(n, k, input_shape, class_count, dim_increase='conv1'):
    block_kind = ResidualBlockKind(
        ksizes=[3, 3],
        dropout_locations=[0],
        dropout_rate=0.3,
        dim_increase=dim_increase)
    group_count = 3
    blocks_per_group = (n - 4) // (group_count * len(block_kind.ksizes))
    print("group count: {}, blocks per group: {}".format(
        group_count, blocks_per_group))
    model = ResNet(
        input_shape=input_shape,
        class_count=class_count,
        batch_size=128,
        learning_rate_policy={
            'boundaries': [60, 120, 160],
            'values': [1e-1 * 0.2**i for i in range(4)]
        },
        block_kind=block_kind,
        group_lengths=[blocks_per_group] * group_count,
        widening_factor=k,
        weight_decay=5e-4,
        training_log_period=50)
    assert n == model.zagoruyko_depth, "invalid depth (n={}!={})".format(
        n, model.zagoruyko_depth)
    return model