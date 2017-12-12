import tensorflow as tf
import numpy as np

# Original source: CleverHans - https://github.com/tensorflow/cleverhans/


def fgsm(x, predictions, eps=0.3, clip_min=None, clip_max=None):
    # https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py
    return fgs(
        x,
        predictions,
        y=None,
        eps=eps,
        ord=np.inf,
        clip_min=clip_min,
        clip_max=clip_max)


def fgs(x, preds, y=None, eps=0.3, ord=np.inf, clip_min=None, clip_max=None, targeted=False):
    """
    TensorFlow implementation of the Fast Gradient Method.
    Original code: https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py
    :param x: the input placeholder
    :param preds: the model's output tensor (the attack expects the
        probabilities, i.e., the output of the softmax)
    :param y: A placeholder for the model labels. If targeted is true, then
        provide the target label. Otherwise, only provide this parameter if
        you'd like to use true labels when crafting adversarial samples. 
        Otherwise, model predictions are used as labels to avoid the "label 
        leaking" effect (explained here: https://arxiv.org/abs/1611.01236). 
        Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: Order of the norm. Possible values (like in Numpy): np.inf, 1 or
        2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
        default, will try to make the label incorrect. Targeted will instead try
        to move in the direction of being more like y.
    :return: a tensor for the adversarial example
    """
    # https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = utils_tf.model_loss(y, preds, mean=False)
    if targeted: loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        # Take sign of gradient
        normalized_grad = tf.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `normalized_grad` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 1:
        red_ind = list(xrange(1, len(x.get_shape())))
        normalized_grad = grad / tf.reduce_sum(
            tf.abs(grad), reduction_indices=red_ind, keep_dims=True)
    elif ord == 2:
        red_ind = list(xrange(1, len(x.get_shape())))
        square = tf.reduce_sum(
            tf.square(grad), reduction_indices=red_ind, keep_dims=True)
        normalized_grad = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Multiply by constant epsilon
    scaled_grad = eps * normalized_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + scaled_grad

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x
