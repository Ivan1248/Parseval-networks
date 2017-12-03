import os
import math

import numpy as np
import skimage as ski
import skimage.io


def forward_pass(net, inputs):
  output = inputs
  for layer in net:
    output = layer.forward(output)
  return output


def backward_pass(net, loss, x, y):
  grads = []
  grad_out = loss.backward_inputs(x, y)
  if loss.has_params:
    grads += loss.backward_params()
  for layer in reversed(net):
    grad_inputs = layer.backward_inputs(grad_out)
    if layer.has_params:
      grads += [layer.backward_params(grad_out)]
    grad_out = grad_inputs
  return grads

def sgd_update_params(grads, config):
  lr = config['lr']
  for layer_grads in grads:
    for i in range(len(layer_grads) - 1):
      params = layer_grads[i][0]
      grads = layer_grads[i][1]
      #print(layer_grads[-1], " -> ", grads.sum())
      params -= lr * grads
      

def draw_conv_filters(epoch, step, layer, save_dir):
  C = layer.C
  w = layer.weights.copy()
  num_filters = w.shape[0]
  k = int(np.sqrt(w.shape[1] / C))
  w = w.reshape(num_filters, C, k, k)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  #for i in range(C):
  for i in range(1):
    img = np.zeros([height, width])
    for j in range(num_filters):
      r = int(j / cols) * (k + border)
      c = int(j % cols) * (k + border)
      img[r:r+k,c:c+k] = w[j,i]
    filename = '%s_epoch_%02d_step_%06d_input_%03d.png' % (layer.name, epoch, step, i)
    ski.io.imsave(os.path.join(save_dir, filename), img)


def train(train_x, train_y, valid_x, valid_y, net, loss, config):
  lr_policy = config['lr_policy']
  batch_size = config['batch_size']
  max_epochs = config['max_epochs']
  save_dir = config['save_dir']
  num_examples = train_x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  for epoch in range(1, max_epochs+1):
    if epoch in lr_policy:
      solver_config = lr_policy[epoch]
    cnt_correct = 0
    #for i in range(num_batches):
    # shuffle the data at the beggining of each epoch
    permutation_idx = np.random.permutation(num_examples)
    train_x = train_x[permutation_idx]
    train_y = train_y[permutation_idx]
    #for i in range(100):
    draw_conv_filters(0, 0, net[0], save_dir)
    for i in range(num_batches):
      # store mini-batch to ndarray
      batch_x = train_x[i*batch_size:(i+1)*batch_size, :]
      batch_y = train_y[i*batch_size:(i+1)*batch_size, :]
      logits = forward_pass(net, batch_x)
      loss_val = loss.forward(logits, batch_y)
      # compute classification accuracy
      yp = np.argmax(logits, 1)
      yt = np.argmax(batch_y, 1)
      cnt_correct += (yp == yt).sum()
      grads = backward_pass(net, loss, logits, batch_y)
      sgd_update_params(grads, solver_config)

      if i % 5 == 0:
        print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, loss_val))
      if i % 100 == 0:
        draw_conv_filters(epoch, i*batch_size, net[0], save_dir)
        #draw_conv_filters(epoch, i*batch_size, net[3])
      if i > 0 and i % 50 == 0:
        print("Train accuracy = %.2f" % (cnt_correct / ((i+1)*batch_size) * 100))
    print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
    evaluate("Validation", valid_x, valid_y, net, loss, config)
  return net


def evaluate(name, x, y, net, loss, config):
  print("\nRunning evaluation: ", name)
  batch_size = config['batch_size']
  num_examples = x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  cnt_correct = 0
  loss_avg = 0
  for i in range(num_batches):
    batch_x = x[i*batch_size:(i+1)*batch_size, :]
    batch_y = y[i*batch_size:(i+1)*batch_size, :]
    logits = forward_pass(net, batch_x)
    yp = np.argmax(logits, 1)
    yt = np.argmax(batch_y, 1)
    cnt_correct += (yp == yt).sum()
    loss_val = loss.forward(logits, batch_y)
    loss_avg += loss_val
    #print("step %d / %d, loss = %.2f" % (i*batch_size, num_examples, loss_val / batch_size))
  valid_acc = cnt_correct / num_examples * 100
  loss_avg /= num_batches
  print(name + " accuracy = %.2f" % valid_acc)
  print(name + " avg loss = %.2f\n" % loss_avg)

