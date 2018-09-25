import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables, generate_random_images, encode, generate_image_from_seed, generate_walk_in_latent_space, generate_continuous_random_interps, generate_continuous_interps_from_json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
# MEEE custom flags
flags.DEFINE_string("input_seed_path", None, "Path to the json file to be inputted to generator.")
flags.DEFINE_integer("walk_rand_seed", None, "Seed for PRNG to be inputted (to recreate previous film)")
flags.DEFINE_integer("walk_num", 2700, "Number of frames of walk in latent space.")
flags.DEFINE_integer("generation_mode", 1, "Generation mode used in testing. Please refer to README.txt")
flags.DEFINE_string("checkpoint_name", None, "Name of the checkpoint file to load from e.g. DCGAN.model-183502")
flags.DEFINE_string("interp_json", None, "Path to json file which contains the info needed to generate mode 10.")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    if FLAGS.dataset == 'mnist':
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          y_dim=10,
          z_dim=FLAGS.generate_test_images,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          data_dir=FLAGS.data_dir)
    else:
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          z_dim=FLAGS.generate_test_images,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          data_dir=FLAGS.data_dir,
        # MEEE ATOM options
          checkpoint_name=FLAGS.checkpoint_name)

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
      mode = FLAGS.generation_mode
      if mode == 1: # Generate 300 random images and their seed value json files
        generate_random_images(sess, dcgan, FLAGS, 500)
      elif mode == 2: # Generate 1.5 min random num of frames per interpolation. With cut: A - B | C - D
        generate_continuous_random_interps(sess, dcgan, FLAGS, 2700, True, True)
      elif mode == 3: # Generate 1.5 min 32 frames per interpolation. With cut: A - B | C - D
        generate_continuous_random_interps(sess, dcgan, FLAGS, 2700, True, False)
      elif mode == 4: # Generate 1.5 min random num of frames per interpolation. With cut: A - B - C
        generate_continuous_random_interps(sess, dcgan, FLAGS, 2700, False, True)
      elif mode == 5: # Generate 1.5 min 32 frames per interpolation. With cut: A - B - C
        generate_continuous_random_interps(sess, dcgan, FLAGS, 2700, False, False)
      # NOTE: for walk in latent space, it is required to pass in --input_seed_path <filename>.json
      elif mode == 6: # Walk in latent space, velocity/acceleration with clamp mode
        generate_walk_in_latent_space(sess, dcgan, FLAGS, 6)
      elif mode == 7: # Walk in latent space, velocity/acceleration with wrap mode
        generate_walk_in_latent_space(sess, dcgan, FLAGS, 7)
      elif mode == 8: # Walk in latent space, default mode (not velocity/acceleration)
        generate_walk_in_latent_space(sess, dcgan, FLAGS, 8)
      elif mode == 9: # Walk in latent space, velocity/acceleration with reverse mode
        generate_walk_in_latent_space(sess, dcgan, FLAGS, 9)
      elif mode == 10: # Generate continuous interpretation from a json file
        generate_continuous_interps_from_json(sess, dcgan, FLAGS)
      elif mode == 11: # Walk in latent space, velocity/acceleration wrap mode, only update 50 out of 100 values
        generate_walk_in_latent_space(sess, dcgan, FLAGS, 11)
      elif mode == 12: # Change one value out of 100
        generate_single_value_changes(sess, dcgan, FLAGS)



      # Generate
      # generate_image_from_seed(sess, dcgan, FLAGS)
      # encode(sess, dcgan, FLAGS)


    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
    # OPTION = 0
    OPTION = 1
    # visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
