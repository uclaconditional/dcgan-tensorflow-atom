"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange

# MEEE
import json
from colorama import init, Fore, Back, Style

import tensorflow as tf
import tensorflow.contrib.slim as slim

init(autoreset=True)

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  print("MEEE in merge, images shape: " + str(images.shape))
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width,
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc",
            "sy": 1, "sx": 1,
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv",
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def encode(sess, dcgan, config):
  idx = 354
  batch_files = dcgan.data[idx*config.batch_size:(idx+1)*config.batch_size]
  batch = [
    get_image(batch_file,
      input_height=dcgan.input_height,
      input_width=dcgan.input_width,
      resize_height=dcgan.output_height,
      resize_width=dcgan.output_width,
      crop=dcgan.crop,
      grayscale=dcgan.grayscale) for batch_file in batch_files]
  if dcgan.grayscale:
    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
  else:
    batch_images = np.array(batch).astype(np.float32)

  batch_z = np.random.uniform(-1, 1, [config.batch_size, dcgan.z_dim]) \
          .astype(np.float32)

  encoded = sess.run(dcgan.D, feed_dict={dcgan.inputs: batch_images})

  print("MEEE Encode encoded shape: " + str(encoded.shape))

      # sample_files = dcgan.data[0:dcgan.sample_num]
      # sample = [
      #     get_image(sample_file,
      #               input_height=dcgan.input_height,
      #               input_width=dcgan.input_width,
      #               resize_height=dcgan.output_height,
      #               resize_width=dcgan.output_width,
      #               crop=dcgan.crop,
      #               grayscale=dcgan.grayscale) for sample_file in sample_files]
      # if (dcgan.grayscale):
      #   sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      # else:
      #   sample_inputs = np.array(sample).astype(np.float32)


def generate_random_images(sess, dcgan, config, time_stamp, cut, count):
  num_images = cut["num_frame_num"]
  # print("MEEE image_frame_dim: " + str(image_frame_dim))
  idx = 0
  # time_stamp = strftime("%Y%m%d-%H%M%S", gmtime())

  while 1:
    values = np.arange(0, 1, 1./config.batch_size)
    z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
    print("MEEE first z_sample: " + str(z_sample[0, :5]))
    print("MEEE z_sample shape: " + str(z_sample.shape))
    # for kdx, z in enumerate(z_sample): # Why many times sess.run(z_sample) ?
    # print("MEEE kdx: " + str(kdx) + " z shape: " + str(z.shape))
    # z[idx % config.batch_size] = values[kdx]
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    print("MEEE samples shape: " + str(samples.shape))

    for n in range(config.batch_size):
        if idx + 1 > num_images:
            return

        save_name = '{}_{}_{:05d}'.format(config.dataset, time_stamp , count)
        count += 1
        img_path = config.sample_dir + "/" + save_name + '.png'
        json_path = config.sample_dir + "/" + save_name + '.json'
        print("img path rand gen: " + img_path);
        # save_images(samples[0, :, :, :], [1, 1], './samples/test_single%s.png' % (0))
        scipy.misc.imsave(img_path, samples[n, :, :, :])
        rand_seed = z_sample[n, :].tolist()
        # Save seed into json
        with open(json_path, 'w') as outfile:
            json.dump(rand_seed, outfile)
        idx += 1
    return count

def generate_image_from_seed(sess, dcgan, config):
    json_path = config.input_seed_path
    json_file_name = json_path.split("/")[-1]
    json_file_name = json_file_name.split(".")[0]
    seed = []
    if json_path:
        with open(json_path, 'r') as f:
            seed = json.load(f)
        print("MEEE seed read: " + str(seed))
    else:
        print(Fore.RED + "MEEE WARNING: Input seed path is None.")
        return
    z_sample_list = []
    for i in range(config.batch_size):
        z_sample_list.append(seed)

    z_sample = np.asarray(z_sample_list, dtype=np.float32)
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_name = 'GenFrom_{}'.format(json_file_name)
    img_path = './samples/' + save_name + '.png'
    # save_images(samples[0, :, :, :], [1, 1], './samples/test_single%s.png' % (0))
    scipy.misc.imsave(img_path, samples[0, :, :, :])
    print(Fore.CYAN + "MEEE seed image generated: " + img_path)

def generate_single_value_changes(sess, dcgan, config, time_stamp, cut, count):
    change_idx_num = cut["change_idx_num"]
    # time_stamp = strftime("%Y%m%d-%H%M%S", gmtime())
    starting_image_path = cut["starting_image"]
    # json_path = config.input_seed_path
    seed = [] # Will be reassigned before use
    if starting_image_path:
        with open(starting_image_path, 'r') as f:
            seed = json.load(f)
        print("MEEE seed read: " + str(seed))
    else:
        print(Fore.RED + "MEEE WARNING: Input seed path is None.")
    z_sample_list = []
    # for i in range(config.batch_size):
    step = 0.00001
    for i in range(5):
        for j in range(10):
            z_sample_list.append(seed[:])
            for k in range(change_idx_num):
                seed[k] += step
            print("seed[0]: " + str(seed[0]) + " step: " + str(step))
        step *= 10
        # Reset json
        if starting_image_path:
            with open(starting_image_path, 'r') as f:
                seed = json.load(f)
            print("MEEE seed read: " + str(seed))
        else:
            print(Fore.RED + "MEEE WARNING: Input seed path is None.")

    while len(z_sample_list) < config.batch_size:
        z_sample_list.append(seed)
    for i in range(50):
        print("double check: " + str(z_sample_list[i][0]))

    z_sample = np.asarray(z_sample_list, dtype=np.float32)
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    # Generate batch images
    saved_idx = 0
    for i in range(5):
        for j in range(10):
            # save_name = '{}_{}_{}_{:02d}'.format(time_stamp, change_idx_num, str(5-i), j)
            save_name = '{}_{}_{:05d}'.format(config.dataset, time_stamp , count)
            count += 1
            img_path = config.sample_dir + "/" + save_name + '.png'
            scipy.misc.imsave(img_path, samples[saved_idx, :, :, :])
            print(Fore.CYAN + "MEEE mode12 image generated: " + img_path)
            saved_idx+=1
    return count

def generate_sin_cycle_all_100(sess, dcgan, config, time_stamp, cut, count):
    # time_stamp = strftime("%Y%m%d-%H%M%S", gmtime())
    starting_image_path = cut["starting_image"]
    # json_path = config.input_seed_path
    seed = [] # Will be reassigned before use
    if starting_image_path:
        with open(starting_image_path, 'r') as f:
            seed = json.load(f)
        print("MEEE seed read: " + str(seed))
    else:
        print(Fore.RED + "MEEE WARNING: Input seed path is None.")
    orig_seed = seed[:]
    # sin_cycle_json_path = config.sin_cycle_json
    # sin_cycle_json_path = cut["params"]
    z_sample_list = []
    num_cycles = 2
    # frames_per_cycle = 30 * 6# PARAM one cycle in 6 seconds
    frames_per_cycle = cut["params"]["frames_per_cycle"]
    num_total_frames = frames_per_cycle * 100
    # num_frames_per_number = frames_per_cycle * num_cycles
    sin_step = (2 * math.pi) / frames_per_cycle
    saved_frame = 0
    curr_frame = 0

    while curr_frame < num_total_frames:
        z_sample_list = []
        for i in range(config.batch_size):
            z_sample_list.append(seed[:])
            # Update seed with sin
            seed = orig_seed[:]
            seed_idx = int(int(curr_frame) / int(frames_per_cycle)) % 100 # %100 to prevent going over 100
            print("seed_idx: " + str(seed_idx) + " : " + str(curr_frame) + " / " + str(frames_per_cycle) + " % " + str(num_total_frames))
            seed[seed_idx] = math.sin((curr_frame % frames_per_cycle) * sin_step)
            curr_frame+=1


        z_sample = np.asarray(z_sample_list, dtype=np.float32)
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        for i in range(config.batch_size):
            # save_name = 'Sin_cycle_all_100_{}_{:05d}'.format(time_stamp, saved_frame)
            save_name = '{}_{}_{:05d}'.format(config.dataset, time_stamp , count)
            img_path = config.sample_dir + "/" + save_name + '.png'
            # scipy.misc.imsave(img_path, samples[i, :, :, :])
            print(Fore.CYAN + "MEEE sin cycle all 100 image generated: " + img_path)
            saved_frame += 1
            if saved_frame >= num_total_frames:
                return count
    return count

def generate_sin_cycle(sess, dcgan, config, time_stamp, cut, count):
    num_cycles = cut["num_cycles"]
    seconds_per_cycle = cut["seconds_per_cycle"]
    mode = cut["mode_num"]
    starting_image_path = cut["starting_image"]
    # time_stamp = strftime("%Y%m%d-%H%M%S", gmtime())
    # json_path = config.input_seed_path
    seed = [] # Will be reassigned before use
    if starting_image_path:
        with open(starting_image_path, 'r') as f:
            seed = json.load(f)
        print("MEEE seed read: " + str(seed))
    else:
        print(Fore.RED + "MEEE WARNING: Input seed path is None.")
    z_sample_list = []
    frames_per_cycle = 30 * seconds_per_cycle  # PARAM one cycle in 10 seconds
    num_total_frames = frames_per_cycle * num_cycles
    sin_step = (2 * math.pi) / frames_per_cycle
    saved_frame = 0
    curr_frame = 0

    # sin_cycle_json_path = config.sin_cycle_json
    # cycle_data = cut["cycle_data"]
    if mode == 14:
        # if "cycle_data" in cut:
        #     # with open(sin_cycle_json_path, 'r') as f:
        #     #     cycle_json = json.load(f)
        #     print("MEEE cycle data read: " + str(cycle_data))
        # else:
        #     print(Fore.RED + "MEEE WARNING: sin cycle data is None.")
        num_total_frames = cut["overall_length"]
        cycle_data = cut["data"]


    while curr_frame < num_total_frames:
        z_sample_list = []
        for i in range(config.batch_size):
            z_sample_list.append(seed[:])
            # Update seed with sin
            for sin in cycle_data:
                sin_step = (2*math.pi) / sin["framesPerCycle"]
                seed[sin["idx"]] = math.sin(curr_frame * sin_step + sin["phaseShift"] * 2 * math.pi)
            curr_frame+=1


        z_sample = np.asarray(z_sample_list, dtype=np.float32)
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        for i in range(config.batch_size):
            # save_name = 'Sin_cycle_withJson_{}_{:05d}'.format(time_stamp, saved_frame)
            save_name = '{}_{}_{:05d}'.format(config.dataset, time_stamp , count)
            count += 1
            img_path = config.sample_dir + "/" + save_name + '.png'
            scipy.misc.imsave(img_path, samples[i, :, :, :])
            print(Fore.CYAN + "MEEE sin cycle image generated: " + img_path)
            saved_frame += 1
            if saved_frame >= num_total_frames:
                return count
    return count



def generate_walk_in_latent_space(sess, dcgan, config, time_stamp, cut, count):
    walk_num = cut["total_frame_num"]
    mode = cut["mode_num"]
    starting_image_path = cut["starting_image"]
    max_vector_length = cut["params"]["max_vector_length"]
    # time_stamp = strftime("%Y%m%d-%H%M%S", gmtime())
    # json_path = config.input_seed_path
    # json_file_name = starting_image_path.split("/")[-1]
    # json_file_name = json_file_name.split(".")[0]
    seed = [] # Will be reassigned before use
    if starting_image_path:
        with open(starting_image_path, 'r') as f:
            seed = json.load(f)
        print("MEEE seed read: " + str(seed))
    else:
        print(Fore.RED + "MEEE WARNING: Input seed path is None.")
        return

    if config.walk_rand_seed == None:
        rand_seed = random.randint(0, 10000)
    else:
        rand_seed = config.walk_rand_seed
    print(Fore.RED + "MEEE rand seed + 1: " + str(rand_seed + 1))
    random.seed(rand_seed)

    # rand_state_json_path = './samples/Walk_{}_randState.json'.format(time_stamp)
    # with open(rand_state_json_path, 'w') as outfile:
    #   json.dump(walk_rand_state, outfile)
    #   print(Fore.CYAN + "MEEE saved rand state json: " + rand_state_json_path)

    walked = 0
    # max_vector_length = 0.003 #0.005 # PARAM
    vector = np.random.uniform(-max_vector_length, max_vector_length, size=(1, dcgan.z_dim))[0]
    # Zero out half the vector if mode 11 and move 50/100 vectors
    if mode == 11:
        vector[50:] = np.zeros(50)
    temp_max_val = 0.0
    while walked < walk_num:
        z_sample_list = []
        for i in range(config.batch_size):
            z_sample_list.append(seed)
            # seed = walk_seed(seed)
            if mode == 6:
                seed, vector, temp_max_val = vector_walk_seed(seed, vector, 6, 0.0003, None)
            elif mode == 7:
                seed, vector = vector_walk_seed(seed, vector, 7, 0.0003, None)
            elif mode == 8:
                seed = walk_seed(seed)
            elif mode == 9:
                seed, vector = vector_walk_seed(seed, vector, 9, 0.0003, None)
            elif mode == 11:
                seed, vector = vector_walk_seed(seed, vector, 11, 0.0003, None)
            elif mode == 16:
                seed, vector, temp_max_val = vector_walk_seed(seed, vector, 16, config.max_jump_step, config.min_jump_step, temp_max_val)

                # seed = walk_seed(seed, config.max_jump_step)


              # Generate batch images
        z_sample = np.asarray(z_sample_list, dtype=np.float32)
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

        for i in range(config.batch_size):
            if mode == 16:
                save_name = 'Jump_randSeed{}_{}_max{}_min{}_{:05d}'.format(rand_seed, time_stamp, cut["max_jump_step"], cut["min_jump_step"], walked)
            else:
                save_name = '{}_{}_{}_{:05d}'.format(config.dataset, rand_seed, time_stamp , count)
            img_path = config.sample_dir + "/" + save_name + '.png'
            scipy.misc.imsave(img_path, samples[i, :, :, :])
            print(Fore.CYAN + "MEEE walk image generated: " + img_path)
            walked += 1
            if walked >= walk_num:
                print("Final temp_max_val: " + str(temp_max_val))
                return count
    return count

# Walk with a vector
def vector_walk_seed(seed, vector, walk_mode, max_step, min_step, temp_max_val=None):
    # maxVectorWalkStep = max_step#0.0003 #0.0005  # PARAM
    result_vector = []
    result_seed = []
    max_vel = max_step * 27.3 # PARAM
    max_vel_perc = 0.2 # PARAM
    for idx in range(len(seed)):
        if min_step == None:
            vectorWalkStep = random.uniform(-max_step, max_step)
        else:
            diff = max_step - min_step
            rand_step = random.uniform(-diff, diff)
            vectorWalkStep = rand_step + (rand_step / abs(rand_step)) * min_step
            # vectorWalkStep = random.uniform(-1.0, 1.0)

        if walk_mode == 11:
            if idx > 49:
                vectorWalkStep = 0.0
        vector_cell = vector[idx] + vectorWalkStep

        if walk_mode == 16: # Cap velocity and push towards the center if around max/min
            vel_bound = max_vel * (1.0-max_vel_perc)
            if vector_cell > vel_bound: # If near upper bound, needs to go lower dir
                vector_cell = vector[idx] - abs(vectorWalkStep)
            elif vector_cell < -vel_bound: # If near lower bound, needs to go higher dir
                vector_cell = vector[idx] + abs(vectorWalkStep)

        # Debug
        if abs(vector_cell) > temp_max_val and temp_max_val != None:
          temp_max_val = abs(vector_cell)
        print("Vector curr val: " + str(vector_cell))

        cell = seed[idx] + vector_cell
          
        if walk_mode == 6: # clamp mode
            cell = max(min(cell, 1.0), -1.0)
        elif walk_mode == 7 or walk_mode == 11 or walk_mode == 16:
            if cell > 1: # Wrap mode
                print("MEEE vector walk cell before: " + str(cell))
                frac, whole = math.modf(cell)
                cell = -1 + frac
                print("MEEE vector walk cell after: " + str(cell))
            elif cell < -1:
                print("MEEE vector walk cell before: " + str(cell))
                frac, whole = math.modf(cell)
                cell = 1 + frac
                print("MEEE vector walk cell after: " + str(cell))
        elif walk_mode == 9:
            if cell > 1 or cell < -1:
                vector_cell = -vector_cell

        result_vector.append(vector_cell)
        result_seed.append(cell)

    return result_seed, result_vector, temp_max_val

# Walk a single step for all 100 numbers in a seed
def walk_seed(seed, max_step=0.035):
    # maxWalkStep = 0.035 # PARAM
    maxWalkStep = max_step # PARAM
    result_seed = []
    for idx in range(len(seed)):
        random_cell = random.uniform(-maxWalkStep, maxWalkStep)
        print("MEEE random cell: " + str(random_cell))
        cell = seed[idx] + random_cell
        # print("MEEE updated cell: " + str(cell))
        # cell = np.clip(-1.0, 1.0, cell)
        cell = max(min(cell, 1.0), -1.0)
        # print("MEEE after clip: " + str(cell))
        result_seed.append(cell)
    np_result_seed = np.asarray(result_seed, dtype=np.float32)
    np_seed = np.asarray(seed, dtype=np.float32)
    # print("MEEE walk seed diff: " + str(np_result_seed - np_seed))
    return result_seed

def generate_continuous_random_interps(sess, dcgan, config, time_stamp, cut, count):
    total_frame_num = cut["total_frame_num"]
    mode = cut["mode_num"]
    if mode == 2:
      is_cut, is_rand_steps_per_interp = True, True
    elif mode == 3:
      is_cut, is_rand_steps_per_interp = True, False
    elif mode == 4:
      is_cut, is_rand_steps_per_interp = False, True
    elif mode == 5:
      is_cut, is_rand_steps_per_interp = False, False
    else:
      is_cut, is_rand_steps_per_interp = None, None

    # steps_per_interp = 32 # 16   # PARAM
    steps_per_interp = cut["steps_per_interp"]  # 16   # PARAM
    stored_images = 0
    num_queued_images = 0
    # time_stamp = strftime("%Y%m%d-%H%M%S", gmtime())
    rand_batch_z = np.random.uniform(-1, 1, size=(2 , dcgan.z_dim))
    z1 = np.asarray(rand_batch_z[0, :])
    z2 = np.asarray(rand_batch_z[1, :])
    while stored_images < total_frame_num:
        batch_idx = 0
        batch_seeds = np.zeros(shape=(config.batch_size, 100))

        while batch_idx < config.batch_size:
            interp_idx = num_queued_images % steps_per_interp
            print("interp_idx: " + str(interp_idx))
            # for i, ratio in enumerate(np.linspace(0, 1, steps_per_interp)):
            ratio = np.linspace(0, 1, steps_per_interp)[interp_idx]
            # print("i: " + str(i) + " ratio: " + str(ratio))
            print(" ratio: " + str(ratio))

            slerped_z = slerp(ratio, z1, z2)
            # print("MEEE ratio: " + str(ratio) + " z1: " + str(z1.shape) + " z2: " + str(z2.shape))
            # batch_seeds = np.append(batch_seeds, [slerped_z], axis=0)
            print("MEEE batch_idx: " + str(batch_idx))
            batch_seeds[batch_idx] = slerped_z
            # print("MEEE batch_seeds: " + str(batch_seeds.shape) + " , slerped_z: " + str(slerped_z.shape))
            batch_idx += 1
            num_queued_images += 1

                # if batch_idx >= config.batch_size:
                #     break

            if num_queued_images % steps_per_interp == 0:
                interp_frame_nums = [8, 16, 32, 8, 25, 36, 85, 7, 16, 10, 40, 10, 30, 20, 30, 34, 50, 25, 50, 100, 120, 250, 300, 512]
                if is_rand_steps_per_interp:
                    steps_per_interp = interp_frame_nums[random.randint(0, len(interp_frame_nums)-1)]
                    num_queued_images = 0
                rand_batch_z = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
                if is_cut:
                    z1 = np.asarray(rand_batch_z[1, :]) #PARAM A - B - C or A - B | C - D
                else:
                    z1 = z2
                z2 = np.asarray(rand_batch_z[0, :])
                print("MEEE newly assigned z1: " + str(z1))
                print("MEEE newly gen uniform z2: " + str(z2))

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: batch_seeds})

        # Naming
        for i in range(config.batch_size):
            save_name = '{}_{}_{:05d}'.format(config.dataset, time_stamp , count)
            img_path = config.sample_dir + "/" + save_name + '.png'
            scipy.misc.imsave(img_path, samples[i, :, :, :])
            print(Fore.CYAN + "MEEE Continuous random interp image generated: " + img_path)
            stored_images += 1
            if stored_images >= total_frame_num:
                return count
    return count

def generate_continuous_interps_from_json(sess, dcgan, config, base_dir, time_stamp, cut, count):

    # Read interp json
    # with open(config.interp_json, 'r') as f:
    #     interp_data = json.load(f)
    interp_data = cut["mode_data"]

    steps_per_interp = interp_data[0][2] # 16   # PARAM
    stored_images = 0
    num_queued_images = 0
    # time_stamp = strftime("%Y%m%d-%H%M%S", gmtime())

    # base_json_path = cut["base_dir"] # NOTE: Should pass in from main?
    base_json_path = base_dir
    seedA = []
    seedB = []
    with open(base_json_path + '/' + interp_data[0][0] + ".json", 'r') as f:
        seedA = json.load(f)
    with open(base_json_path + '/' + interp_data[0][1] + ".json", 'r') as f:
        seedB = json.load(f)

    total_frame_num = 0
    for i in range(len(interp_data)):
        total_frame_num += interp_data[i][2]

    # print("MEEE total frame num : " + str(total_frame_num))
    # z_sample_list = []
    # for i in range(config.batch_size):
        # z_sample_list.append(seed)

    # z_sample = np.asarray(z_sample_list, dtype=np.float32)
    curr_cut_idx = 0

    rand_batch_z = np.random.uniform(-1, 1, size=(2 , dcgan.z_dim))
    # z1 = np.asarray(rand_batch_z[0, :])
    # z2 = np.asarray(rand_batch_z[1, :])
    z1 = np.asarray(seedA, dtype=np.float32)
    z2 = np.asarray(seedB, dtype=np.float32)
    # print("z1: " + str(z1))
    while stored_images < total_frame_num:
    # for i in range(len(interp_data["data"])):
        batch_idx = 0
        batch_seeds = np.zeros(shape=(config.batch_size, 100), dtype=np.float32)
        # batch_seeds = []

        while batch_idx < config.batch_size:
            interp_idx = num_queued_images % steps_per_interp
            # print("interp_idx: " + str(interp_idx))
            # for i, ratio in enumerate(np.linspace(0, 1, steps_per_interp)):
            ratio = np.linspace(0, 1, steps_per_interp)[interp_idx]
            ratio = np.float32(ratio)
            # print("linspace:")
            # print(str(np.linspace(0, 1, steps_per_interp)))
            # print("i: " + str(i) + " ratio: " + str(ratio))
            print(" ratio: " + str(ratio))

            slerped_z = slerp(ratio, z1, z2)
            # slerped_z = z1 * (1.0 - ratio) + z2 * ratio
            # slerped_z = seedA
            # print("slerped_z: " + str(slerped_z))
            # print("MEEE ratio: " + str(ratio) + " z1: " + str(z1.shape) + " z2: " + str(z2.shape))
            # batch_seeds = np.append(batch_seeds, [slerped_z], axis=0)
            # print("MEEE batch_idx: " + str(batch_idx))
            batch_seeds[batch_idx] = slerped_z
            # batch_seeds.append(slerped_z[:])
            # print("MEEE batch_seeds: " + str(batch_seeds.shape) + " , slerped_z: " + str(slerped_z.shape))
            batch_idx += 1
            num_queued_images += 1

                # if batch_idx >= config.batch_size:
                #     break

            if num_queued_images % steps_per_interp == 0:
                # interp_frame_nums = [8, 16, 32, 8, 25, 36, 85, 7, 16, 10, 40, 10, 30, 20, 30, 34, 50, 25, 50, 100, 120, 250, 300, 512]
                curr_cut_idx += 1
                # print("loading curr cur idx: " + str(curr_cut_idx))
                # print("num_queued_images: " + str(num_queued_images))
                if curr_cut_idx >= len(interp_data):
                    continue
                steps_per_interp = interp_data[curr_cut_idx][2]
                num_queued_images = 0
                # if is_rand_steps_per_interp:
                    # steps_per_interp = interp_frame_nums[random.randint(0, len(interp_frame_nums)-1)]
                rand_batch_z = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
                # Read new json to z1

                with open(base_json_path + '/' + interp_data[curr_cut_idx][0] + ".json", 'r') as f:
                    seedA = json.load(f)
                with open(base_json_path + '/' + interp_data[curr_cut_idx][1] + ".json", 'r') as f:
                    seedB = json.load(f)

                z1 = np.asarray(seedA, dtype=np.float32)
                z2 = np.asarray(seedB, dtype=np.float32)
                # print("z1: " + str(z1) + " type z1: " + str(type(z1)))


                # if is_cut:
                    # z1 = np.asarray(rand_batch_z[1, :]) #PARAM A - B - C or A - B | C - D
                # else:
                    # z1 = z2
                # z2 = np.asarray(rand_batch_z[0, :])
                # print("MEEE newly assigned z1: " + str(z1))
                # print("MEEE newly gen uniform z2: " + str(z2))

        # np_batch_seeds = np.asarray(batch_seeds, dtype=np.float32)
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: batch_seeds})
        # samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: np_batch_seeds})

        # Naming
        for i in range(config.batch_size):
            save_name = '{}_{}_{:05d}'.format(config.dataset, time_stamp , count)
            count += 1
            # TODO: Create timestampt dir
            img_path = config.sample_dir + "/" + save_name + '.png'
            scipy.misc.imsave(img_path, samples[i, :, :, :])
            print(Fore.CYAN + "MEEE Continuous random interp image generated: " + img_path)
            stored_images += 1
            # print("stored images: " + str(stored_images))
            # print("total framenum: " + str(total_frame_num))
            if stored_images >= total_frame_num:
                print("MEEE Should return!!")
                return count
    return count

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    # print("MEEE slepr low: " + str(low.shape) + ", high: " + str(high.shape))
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    print("omega: " + str(omega) + " so: " + str(so) + " val: " + str(val))
    print("typeomega: " + str(type(omega)) + " typeso: " + str(type(so)) + " val type: " + str(type(val)))
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    result = np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high
    if val <= 0.0:
      print(str(result - low))
    return result



def visualize(sess, dcgan, config, option):
  image_frame_dim = int(math.ceil(config.batch_size**.5))
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        print("MEEE samples shape: " + str(samples.shape))

      save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, dcgan.z_dim - 1) for _ in xrange(dcgan.z_dim)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      try:
        make_gif(samples, './samples/test_gif_%s.gif' % (idx))
      except:
        save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)


def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  # MEEE to account for change in batch size
  manifold_w = int(8)
  manifold_h = int(num_images/8)
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w
