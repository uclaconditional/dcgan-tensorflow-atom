## Generation mode for REAS_MI
The python code for generating images is the same with starting training (`main.py`.) The images are generated in directory specified by `--sample_dir` option. More on that below.  

### Note on models
Trained models are located in directory `DCGAN-tensorflow/checkpoints`.

### Generating video with DCGAN
#### `main.py` options
##### --dataset :
Name of the directory (not path to that directory) that contains the images to train on. This option needs to be defined also when in generating images after training. It should be the directory that contains the images the model was originally trained on.
e.g. "--dataset=Frames-all-scaleTranslate" if the images are in /media/conditionalstudio/REAS_MI_2/Persona/Frames-all-scaleTranslate.
##### --data_dir :
Path to the directory that contains the images to train on.  Same as `—-dataset` this option should be defined when generating images.
e.g. "--data_dir=/media/conditionalstudio/REAS_MI_2/Persona" if the images are in /media/conditionalstudio/REAS_MI_2/Persona/Frames-all-scaleTranslate.
##### --input_fname_pattern :
Format of the input images. Same as above, this option needs to be defined when generating images after training.
```
--input_fname_pattern="Persona*.jpg"
```
##### --input_height --input_width:
Dimension of the images to be trained on. If the actual image dimension does not match this number, it will be scaled to match during DCGAN training.
e.g. --input_height=64
##### --output_height --output_width:
Dimension of the output (generated) images. It can be different from --input_height --input_width.
e.g. --output_width=64
##### --sample_dir :
Directory that samples generated during training are saved. If not supplied, by default saves in <DCGAN_root>/samples.
##### --batch_size :
Number of images to process per iteration. The higher the number the faster the training. It depends on memory size of the GPU. With 64x64 to 64x64 default batch_size should be fine. With 128x128 to 128x128 batch_size 16 is the fastest we can do on Zotac.
##### --gen_json :
Path to the json file that defines the order and parameters of consecutive cuts. More on that below.
##### Example command:
In root directory of DCGAN, 
```
python main.py --dataset=Frames-all-scaleTranslate-mag2 --data_dir=/media/conditionalstudio/REAS_MI_2/Persona --input_fname_pattern="Persona-*.jpg" --input_height=128 --input_width=128 --batch_size=16 --crop --output_height=128 --output_width=128 --checkpoint_name="DCGAN.model-183502" --generation_mode=1
```
##### The commands does:
Loads trained model "DCGAN.model-183502" of training of images in "/media/conditionalstudio/REAS_MI_2/Persona/Frames-all-scaleTranslate-mag2" with dimension 128x128 to 128x128. Generate output defined as mode 1 (more on that below).

### Generating video with consecutive cuts with DCGAN
The command above in conjunction with `--gen_json` can be used to provide parameters for videos with multiple cuts.

### Example JSON file format
```
{
                "trained_model" : "checkpoint/Contempt_Export_DCGAN_128_16_128_128/DCGAN.model-192501",
                "base_dir" : "samples/UntitledFilmStills-Contempt-source",
                "rand_seed" : 1234,
                "data" : [
                                { "mode_num" : 1,
                                 "mode_data" : [["RandGen_20180918-181322_00119", "RandGen_20180918-181322_00233", 48],
                                                                ["RandGen_20180918-181322_00083", "RandGen_20180918-181322_00337", 48]]
                                },
                                { "mode_num" : 2,
                                 "mode_data" : [["RandGen_20180918-181322_00119", "RandGen_20180918-181322_00233", 48],
                                                                ["RandGen_20180918-181322_00083", "RandGen_20180918-181322_00337", 48]]
                                },
                                { "mode_num" : 3,
                                  "start_image" : "RandGen_20180918-181322_00119",
                                  "total_frame_num" : 120,
                                  "amplitude" : 0.1,
                                  "speed" : 0.01
                                },
                                { "mode_num" : 4,
                                  "start_image" : "RandGen_20180918-181322_00119",
                                  "total_frame_num" : 120,
                                  "max_speed" : 0.01
                                }
                ]
}
```

### Mode descriptions
#### Mode 1:
Use spherical linear interpolation, slerp (implementation by Tom White [https://github.com/soumith/dcgan.torch/issues/14](https://github.com/soumith/dcgan.torch/issues/14) ), to interpolate between a pair of frames. (For e.g. From frame A to frame B, then from frame C to frame D). Slerp is generally smoother than ‘lerp’.
#### Mode 2:
Use spherical linear interpolation, slerp (implementation by Tom White [https://github.com/soumith/dcgan.torch/issues/14](https://github.com/soumith/dcgan.torch/issues/14) ), to interpolate between a sequence of frames. (For e.g. From frame A to frame B to frame C to frame D). Slerp is generally smoother than ‘lerp’.
#### Mode 3:
Displace each of the 100 numbers with sine waves. While the amplitude and the frequency across all the numbers are the same, their phase differ and is randomly generated. This phase difference is eased in to create a continuous animation from the starting image.
#### Mode 4:
Randomly walk around the latent space. If any of the 100 numbers reach the boundary of the space, wrap around to the other end.
#### Mode 5:
Randomly walk around the latent space. If any of the 100 numbers reach the boundary of the space, clamp it at the boundary.
#### Mode 6:
Linearly interpolate between a sequence of frames (e.g. From frame A to frame B, then from frame C to frame D)
#### Mode 7:
Linearly interpolate between a sequence of frames (e.g. From frame A to frame B, then from frame C to frame D), but wrap around the boundary if wrap distance is shorter than regular interpolation distance.
#### Mode 8:
Randomly jump around the space within a certain distance from the starting image. Generates a flicker-like effect.
#### Mode 9:
Exponential ease in or ease out when interpolating between two key frames. Easing speed and whether to ease in or out is controlled through json params.
#### Mode 10:
Sinusoidally ease in or out when interpolating between two key frames.
#### Mode 11:
Randomly jump around the space within a certain distance from the lerp position from A - B.
#### Mode 12:
Exponentially ease in and then out when interpolating between two key frames (A | slow - fast - slow | B). Easing speed is controllable through param “power”.
#### Mode 13:
Exponential ease in and then out with flicker. Combination of mode 12 and 8.
#### Mode 14:
Slerp with flicker. Combination of mode 1 and 8.
#### Mode 15:
Traverse all latent vectors. Linearly step through each of the 100 latent vectors.

### JSON file global parameters
```
"base_dir" : Directory where seed JSON files being used in this video is located.
"trained_model" : Path to trained model to be used for this video.
"rand_seed" : Random seed to be used to generate this video. "rand_seed" is used per video, not per cut.
"data" : List of JSON objects containing parameters for each cut.
```
### Mode specific parameters:
```
Common:
"mode_num" : Mode number of current cut.

if mode_num == 1:  # A - B | B - C, slerp
  "interp_data" : A list of the lists ["nameFrameA", "nameFrameB", number_of_frames_for_interp]
if mode_num == 2:  # A - B - C, slerp
  Same as mode 1.
if mode_num == 3:  # Sinusoidal oscillation
  "start_image" : JSON file name of the starting image.
  "total_frame_num" : Number of total frames to generate for this cut.
  "amplitude" : Amplitude of the sinusoidal motion.
  "speed" : Speed of the sinusoidal motion.
  "easing" : Easing factor to control how smoothly the current phase transitions into target phase.
if mode_num == 4:  # Random walk, wrap
  "start_image" : JSON file name of the starting image.
  "total_frame_num" : Number of total frames to generate for this cut.
  "max_speed" : Maximum of the random speed.
if mode_num == 5:  # Random walk, clamp
  "start_image" : JSON file name of the starting image.
  "total_frame_num" : Number of total frames to generate for this cut.
  "max_speed" : Maximum of the random speed.
  "clamp_boundary" : Seed value does not exceed the range [-clamp_boundary, clamp_boundary].
if mode_num == 6:  # A - B - C, lerp
  Same as mode 1.
if mode_num == 7:  # A - B - C, lerp with wrap if closer
  "interp_data" : List of lists containing ["seedAjson", "seedBjson", num_frames_to_interp]
  "usual_cutoff" : Cutoff point for points within range [-usual_cutoff, usual_cutoff]
  "overflow_buffer" : Cutoff point for points outside range usual_cutoff is [-abs(orig_pos)-overflow, +abs(orig_pos)+overflow]
if mode_num == 8:  # Flicker
  "start_image" : JSON file name of the starting image.
  "total_frame_num" : Number of total frames to generate for this cut.
  "max_step" : Maximun step per number from the original frame.
if mode_num == 9:  # Exponential easing in or out
  "interp_data" : List of lists containing ["seedAjson", "seedBjson", num_frames_to_interp]
  "is_ease_in" : "true" for easing in, "false" for easing out.
  "power" : Exponent of animation. Higher is faster.
if mode_num == 10: # Sinusoidal easing interp.
  "interp_data" : List of lists containing ["seedAjson", "seedBjson", num_frames_to_interp]
  "is_ease_in" : "true" for easing in, "false" for easing out.
if mode_num == 11:  # Flicker lerp
  "interp_data" : List of lists containing ["seedAjson", "seedBjson", num_frames_to_interp]
  "max_step" : Maximun step per number from the original frame.
if mode_num == 12:  # Exponential ease inout
  "interp_data" : List of lists containing ["seedAjson", "seedBjson", num_frames_to_interp]
  "power" : Exponent of animation. Higher is faster.
if mode_num == 13:  # Flicker +  Exponential ease inout A - B | B - C
  "interp_data" : List of lists containing ["seedAjson", "seedBjson", num_frames_to_interp]
  "power" : Exponent of animation. Higher is faster.
  "max_step" : Maximun step per number from the original frame.
if mode_num == 14:  # Slerp + flicker
  "interp_data" : List of lists containing ["seedAjson", "seedBjson", num_frames_to_interp]
  "max_step" : Maximun step per number from the original frame.
```
---
### Legacy modes documentation:
#### Example command:
In root directory of DCGAN, 
```
python main.py --dataset=Frames-all-scaleTranslate-mag2 --data_dir=/media/conditionalstudio/REAS_MI_2/Persona --input_fname_pattern="Persona-*.jpg" --input_height=128 --input_width=128 --batch_size=16 --crop --output_height=128 --output_width=128 --checkpoint_name="DCGAN.model-183502" --generation_mode=1
```
#### The commands does:
Loads trained model "DCGAN.model-183502" of training of images in "/media/conditionalstudio/REAS_MI_2/Persona/Frames-all-scaleTranslate-mag2" with dimension 128x128 to 128x128. Generate output defined as mode 1 (more on that below).
#### Options:
##### --generation_mode :
```
mode == 1: # Generate 300 random images and their seed value json files
mode == 2: # Generate 1.5 min random num of frames per interpolation. With cut: A - B | C - D 
mode == 3: # Generate 1.5 min 32 frames per interpolation. With cut: A - B | C - D 
mode == 4: # Generate 1.5 min random num of frames per interpolation. With cut: A - B - C 
mode == 5: # Generate 1.5 min 32 frames per interpolation. With cut: A - B - C 
mode == 6: # Walk in latent space, velocity/acceleration with clamp mode
mode == 7: # Walk in latent space, velocity/acceleration with wrap mode
mode == 8: # Walk in latent space, default mode (not velocity/acceleration)
mode == 9: # Walk in latent space, velocity/acceleration with reverse mode
mode == 10: # Generate continuous interpretation from a json file
mode == 11: # Walk in latent space, velocity/acceleration wrap mode, only update 50 out of 100 values
mode == 12:  # 10th to 100000th digit change for 1st number of seed
mode == 13: # Sinusoidal cycling of first value, 2 cycles, 10 seconds per cycle
mode == 14: # Sinusoidal cycling of values specified by json (use --sin_cycle_json)
mode == 15: # Sinusoidal cycling through all 100 numbers, 6s percycle
mode == 16: # Jumps in latent space, velocity/acceleration with wrap mode
```
e.g. --generation_mode=6
##### More on generation_mode:
For generation_mode 6-9, a json file name generated by mode 1 will need to be inputted. For e.g. this can be done by:
```
--input_seed_path="samples/RandGen_20180924-181557_00060.json"
```
A PRNG random seed can be inputted for mode 6-9 to "walk the same path" as another walk. If you want to match the sequence "Walk_randSeed4567_20180904-201711_?????.png", input:
```
--walk_rand_seed=4567
```
##### --checkpoint_name :
Name of the model to use for generating. If none supplied the latest saved model (latest checkpoing) is used. To determine which model to use go to <DCGAN_room>/samples (or --sample_dir if explicitly defined) to see samples outputed throughout the training. If the desired image is train_24_7043_iter00183301.png, then --checkpoint_name="DCGAN.model-183301".
##### --sample_dir :
Directory that generated images will be saved. For e.g. If not supplied, by default saved in <DCGAN_root>/samples.
```
--sample_dir=samples/Persona-all-scaleTranslate-128_128
```
##### Mode 10 and 14
A json file of a set format needs to be inputted for these modes. For more information please refer to wiki page on json.

##### Mode 16
The maximum and minimum possible change in velocity can be set through `--max_jump_step` and `--min_jump_step`.

### To combine the generated images into a video file:
#### Example command:
In the directory where image files are saved, 
```
ffmpeg -r 24 -i ContinuousInterp_20180904-201110_%05d.png -crf 0 ContinuousInterp_20180904-201110-mode3.mp4
``` 
ContinuousInterp_20180904-201110-mode3.mp4 is the name of the to-be-outputted mp4 file.
#### The command does:
Generates a mp4 file from images ContinuousInterp_20180904-201110_%05d.png (%05d selects all from 00000->00001->00002... until sequence ends). mp4 will be 24 fps with no compression.
##### -r :
fps number.
##### -i :
Format of input image sequence.
##### -crf :
Compression setting. 0 is lossless, 23 default and 51 is worst quality possible.

## Directories:  
These are the directories where the images used/generated by DCGAN for MI_REAS are stored on Zotac.
### Inputs:
Also contains the original frames and others used by Pix2pix.
```
/media/conditionalstudio/REAS_MI_1/Persona
/media/conditionalstudio/REAS_MI_1/Garden
```
### Outputs/Generated:
Each directory inside the directory below holds images generated by the trained model indicated by the directory name. The generated include images and videos from mode 1-9 as well as samples outputted when training.
```
/media/conditionalstudio/REAS_MI_2/DCGAN-tensorflow/samples
```

-------------------------
# DCGAN in Tensorflow

Tensorflow implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) which is a stabilize Generative Adversarial Networks. The referenced torch code can be found [here](https://github.com/soumith/dcgan.torch).

![alt tag](DCGAN.png)

* [Brandon Amos](http://bamos.github.io/) wrote an excellent [blog post](http://bamos.github.io/2016/08/09/deep-completion/) and [image completion code](https://github.com/bamos/dcgan-completion.tensorflow) based on this repo.
* *To avoid the fast convergence of D (discriminator) network, G (generator) network is updated twice for each D network update, which differs from original paper.*


## Online Demo

[<img src="https://raw.githubusercontent.com/carpedm20/blog/master/content/images/face.png">](http://carpedm20.github.io/faces/)

[link](http://carpedm20.github.io/faces/)


## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)
- (Optional) [moviepy](https://github.com/Zulko/moviepy) (for visualization)
- (Optional) [Align&Cropped Images.zip](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) : Large-scale CelebFaces Dataset


## Usage

First, download dataset with:

    $ python download.py mnist celebA

To train a model with downloaded dataset:

    $ python main.py --dataset mnist --input_height=28 --output_height=28 --train
    $ python main.py --dataset celebA --input_height=108 --train --crop

To test with an existing model:

    $ python main.py --dataset mnist --input_height=28 --output_height=28
    $ python main.py --dataset celebA --input_height=108 --crop

Or, you can use your own dataset (without central crop) by:

    $ mkdir data/DATASET_NAME
    ... add images to data/DATASET_NAME ...
    $ python main.py --dataset DATASET_NAME --train
    $ python main.py --dataset DATASET_NAME
    $ # example
    $ python main.py --dataset=eyes --input_fname_pattern="*_cropped.png" --train

If your dataset is located in a different root directory:

    $ python main.py --dataset DATASET_NAME --data_dir DATASET_ROOT_DIR --train
    $ python main.py --dataset DATASET_NAME --data_dir DATASET_ROOT_DIR
    $ # example
    $ python main.py --dataset=eyes --data_dir ../datasets/ --input_fname_pattern="*_cropped.png" --train
    

## Results

![result](assets/training.gif)

### celebA

After 6th epoch:

![result3](assets/result_16_01_04_.png)

After 10th epoch:

![result4](assets/test_2016-01-27%2015:08:54.png)

### Asian face dataset

![custom_result1](web/img/change5.png)

![custom_result1](web/img/change2.png)

![custom_result2](web/img/change4.png)

### MNIST

MNIST codes are written by [@PhoenixDai](https://github.com/PhoenixDai).

![mnist_result1](assets/mnist1.png)

![mnist_result2](assets/mnist2.png)

![mnist_result3](assets/mnist3.png)

More results can be found [here](./assets/) and [here](./web/img/).


## Training details

Details of the loss of Discriminator and Generator (with custom dataset not celebA).

![d_loss](assets/d_loss.png)

![g_loss](assets/g_loss.png)

Details of the histogram of true and fake result of discriminator (with custom dataset not celebA).

![d_hist](assets/d_hist.png)

![d__hist](assets/d__hist.png)


## Related works

- [BEGAN-tensorflow](https://github.com/carpedm20/BEGAN-tensorflow)
- [DiscoGAN-pytorch](https://github.com/carpedm20/DiscoGAN-pytorch)
- [simulated-unsupervised-tensorflow](https://github.com/carpedm20/simulated-unsupervised-tensorflow)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
