#  The MIT License (MIT)
#
#  Copyright (c) 2016 Yu Ishihara
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

import tensorflow as tf
import deepqnetwork as dqn
import random
import numpy as np
import gflags
import sys
from ale_environment import AleInterface

# Command line args
FLAGS = gflags.FLAGS
gflags.DEFINE_string('train_checkpoint', '', 'checkpoint file of train network')
gflags.DEFINE_string('target_checkpoint', '', 'checkpoint file of target network')

# Network parameters
BATCH_SIZE = 32
IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
NUM_CHANNELS = 4  # dqn inputs 4 image at same time as state

NUM_ACTIONS = 4

MAX_EPOCHS = 200
ITERATIONS_PER_EPOCH = 10000

EPSILON_DECAYING_FRAMES = 1000000
MIN_EPSILON = 0.1

SKIPPING_FRAME_NUM = 4

MAX_MEMORY_SIZE = 50000
MIN_MEMORY_SIZE = 50000
replay_memory = []

NETWORK_UPDATE_FREQUENCY = 10000

# Q-learning parameters
GAMMA = 0.99

graph = tf.Graph()
with graph.as_default():
  tf_train_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS))
  tf_train_target = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_ACTIONS))
  tf_filter_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_ACTIONS))
  train_network = dqn.DeepQNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 'train')

  tf_target_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS))
  target_network = dqn.DeepQNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 'target')
  target_q_values = target_network.q_values(tf_target_input)

  loss = train_network.clipped_loss(tf_train_input, tf_train_target, tf_filter_input)
  optimizer = tf.train.RMSPropGravesOptimizer(learning_rate=0.00025, decay=0.95, momentum=0.0, epsilon=1e-2).minimize(loss)

  tf_action_selection_input = tf.placeholder(tf.float32, shape=(1, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS))
  action_q_values = train_network.q_values(tf_action_selection_input)

# Tensorflow session configs
config = tf.ConfigProto()
# config.log_device_placement = True

# Output to tensorboard
summary_writer = tf.train.SummaryWriter('summary' + '/train', graph=graph)

def train_with(session, memories):
  state_batch = [memory[0] for memory in memories]
  action_batch = [memory[1] for memory in memories]
  reward_batch = [memory[2] for memory in memories]
  next_state_batch = [memory[3] for memory in memories]

  target_qs = target_q_values.eval(feed_dict={tf_target_input: next_state_batch})
  target = np.zeros(shape=(BATCH_SIZE, NUM_ACTIONS), dtype=np.float32)
  q_value_filter = np.zeros(shape=(BATCH_SIZE, NUM_ACTIONS), dtype=np.float32)
  for i in range(BATCH_SIZE):
    end_state = memories[i][4]
    action_index = action_batch[i]
    reward = reward_batch[i]
    target[i][action_index] = reward if end_state else reward + GAMMA * target_qs[i][action_index]
    q_value_filter[i][action_index] = 1.0

  session.run(optimizer, feed_dict={tf_train_input: state_batch,
                                    tf_train_target: target,
                                    tf_filter_input: q_value_filter})

def is_greedy(epsilon):
  return epsilon < random.random()


def select_greedy_action_from(state, available_actions):
  q_values = action_q_values.eval(feed_dict={tf_action_selection_input: [state]})
  index = np.argmax(q_values, 1)[0]  # return value of argmax is a list!
  return available_actions[index]


def select_random_action_from(available_actions):
  return random.choice(available_actions)


def update_target_network(session):
  train_network.copy_network_to(target_network, session)


def calculate_epsilon(frame_num):
  if EPSILON_DECAYING_FRAMES <= frame_num:
    return MIN_EPSILON
  else:
    return 1.0 - ((1.0 - MIN_EPSILON) / EPSILON_DECAYING_FRAMES) * frame_num


def has_enough_memory():
  return MIN_MEMORY_SIZE <= len(replay_memory)


def sample_replay_memory(batch_size):
  return random.sample(replay_memory, batch_size)


def save_replay_memory(memory):
  if memory is None:
    return
  if MAX_MEMORY_SIZE < len(replay_memory):
    del replay_memory[0]
  replay_memory.append(memory)


def execute_one_episode_with(session, environment, iterations, frame_num):
  actions = environment.available_actions()
  environment.reset()

  # Play randomly and get first NUM_CHANNEL input frames to provide to the network
  screens = []
  next_screen = None
  while environment.is_end_state() is False:
    action = select_random_action_from(actions)
    for i in range(SKIPPING_FRAME_NUM):
      reward, next_screen = environment.act(action)
    screens.append(next_screen)
    frame_num += 1
    if len(screens) is NUM_CHANNELS:
      break

  # Start training
  state = np.stack(screens, axis=-1)
  while environment.is_end_state() is False:
    assert state.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)
    epsilon = calculate_epsilon(frame_num)
    if is_greedy(epsilon):
      action = select_greedy_action_from(state, actions)
    else:
      action = select_random_action_from(actions)

    reward = 0.0
    for i in range(SKIPPING_FRAME_NUM):
      intermediate_reward, next_screen = environment.act(action)
      reward += np.clip([intermediate_reward], -1, 1)[0]

    next_state = np.reshape(next_screen, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    next_state = np.append(state[:, :, 1:], next_state, axis=-1)

    memory = (state, action, reward, next_state, environment.is_end_state())
    save_replay_memory(memory)

    if has_enough_memory():
      if iterations % NETWORK_UPDATE_FREQUENCY == 0:
        update_target_network(session)
      memories = sample_replay_memory(BATCH_SIZE)
      train_with(session, memories)
      iterations += 1

    state = next_state
    frame_num += 1

    # print 'epsilon:', epsilon, ' frame_num:', frame_num, ' reward:', reward
  return iterations, frame_num


def evaluate_network(environment):
  actions = environment.available_actions()
  environment.reset()

  # Play randomly and get first NUM_CHANNEL input frames to provide to the network
  screens = []
  next_screen = None
  while environment.is_end_state() is False:
    action = select_random_action_from(actions)
    for i in range(SKIPPING_FRAME_NUM):
      reward, next_screen = environment.act(action)
    screens.append(next_screen)
    if len(screens) is NUM_CHANNELS:
      break

  # Start training
  state = np.stack(screens, axis=-1)
  total_reward = 0
  while environment.is_end_state() is False:
    assert state.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)
    epsilon = 0.05
    if is_greedy(epsilon):
      action = select_greedy_action_from(state, actions)
    else:
      action = select_random_action_from(actions)

    reward = 0.0
    for i in range(SKIPPING_FRAME_NUM):
      intermediate_reward, next_screen = environment.act(action)
      reward += np.clip([intermediate_reward], -1, 1)[0]
    total_reward += reward

    next_state = np.reshape(next_screen, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    state = np.append(state[:, :, 1:], next_state, axis=-1)

  return total_reward

def save_network_parameters(session, iterations):
  train_network.save_parameters(session, 'checkpoint/dqn_train_network', iterations)
  target_network.save_parameters(session, 'checkpoint/dqn_target_network', iterations)

def parse_iteration_num(string):
  candidates = string.split('-')
  for candidate in candidates:
    if candidate.isdigit():
      return int(candidate)
  return 0

def summaries(score):
  score_summary = tf.scalar_summary('score', score)
  return tf.merge_summary([score_summary])


def start_training(argv, session):
  try:
    argv = FLAGS(argv)
  except gflags.FlagsError:
    print 'Incompatible flags were specified'
    return
  print 'train checkpoint: ', FLAGS.train_checkpoint
  print 'target checkpoint: ', FLAGS.target_checkpoint
  train_network.restore_parameters(session, FLAGS.train_checkpoint)
  target_network.restore_parameters(session, FLAGS.target_checkpoint)

  environment = AleInterface('breakout.bin', record_display=False)
  iterations = parse_iteration_num(FLAGS.train_checkpoint)
  frame_num = iterations
  epoch = iterations / ITERATIONS_PER_EPOCH

  print 'start from iteration: ', iterations
  while epoch < MAX_EPOCHS:
    random.seed()
    iterations, frame_num = execute_one_episode_with(session, environment, iterations, frame_num)
    if epoch < (iterations / ITERATIONS_PER_EPOCH):
      save_network_parameters(session, iterations)
      score = evaluate_network(environment)
      summary_writer.add_summary(session.run(summaries(score)), epoch)
      print 'evaluation result of epoch: ', epoch, ' score: ', score
      epoch += 1

def debug_play_randomly():
  environment = AleInterface('breakout.bin')
  environment.reset()
  actions = environment.available_actions()
  for i in range(5):
    while environment.is_end_state() is False:
      environment.act(select_random_action_from(actions))
    environment.reset()

def debug_take_video(session, train_checkpoint):
  environment = AleInterface('breakout.bin')
  for i in range(5):
    random.seed()
    train_network.restore_parameters(session, train_checkpoint)
    score = evaluate_network(environment)

with tf.Session(graph=graph, config=config) as session:
  tf.initialize_all_variables().run()
  update_target_network(session)
  if __name__ == '__main__':
    start_training(sys.argv, session)
