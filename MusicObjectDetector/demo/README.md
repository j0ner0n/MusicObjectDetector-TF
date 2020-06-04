# Running this demo

1. Build the docker image and tag it. In this case we will call it saty/muscima.

   ```bash
   docker build MusicObjectDetector/demo/ -t saty/muscima
   ```

2. All hand writen music sheets should then be put in a single directory. In this example the directory will be `/sheet_music`.
3. Run the docker image with a volume mounted to the directory mapped to `/workspace`.

   ```bash
   # For invoking nvidia-docker
   nvidia-docker run --rm -t -v /sheet_music/:/workspace saty/muscima

   # For the new method of invoking the nvidia-docker runtime
   docker run --gpus all --rm -t -v /sheet_music/:/workspace saty/muscima
   ```

   | Argument                        | Description                                                       |
   |---------------------------------|-------------------------------------------------------------------|
   | `run`                           | Runs an image                                                     |
   | `nvidia-docker` or `--gpus all` | Use the nvidia runtime and allow access to GPUs                   |
   | `--rm`                          | Automatically removes the container once the command has finished |
   | `-t`                            | Allocates a pseudo-tty                                            |
   | `-v /sheet_music/:/workspace`   | Mounts `/sheet_music/` to `/workspace` inside the container       |

4. Inference will now be performed on all image files in the `/sheet_music` directory. The results will be returned to `/sheet_music/output`

## Using optional arguments

The following arguments are available:

| Argument                                                | Description                                                                                                                |
|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `-h`, `--h`                                             | Shows a help message and exit                                                                                              |
| `--detection_inference_graph DETECTION_INFERENCE_GRAPH` | Specify a path to the frozen inference graph                                                                               |
| `--input_directory INPUT_DIRECTORY`                     | Specify an input directory. In this use case, this must be the mapping used for the mounted volume                         |
| `--detection_label_map DETECTION_LABEL_MAP`             | Specify a path to the detection label map                                                                                  |
| `--output_dir OUTPUT_DIR`                               | Specify a path for the output. If the path does not point to somewhere within the mounted volume, the results will be lost |

To use these arguments, simply append the following docker commands:

```bash
docker run -rm -v /sheet_music/:/workspace saty/muscima [OPTIONS]
```

Note that there are no default arguments. The usage of any argument will thus require the use of all arguments

Here, we assume that you want to still perform inference on files in the `/sheet_music` directory. The `/sheet_music` directory is then mapped to `/workspace/` in the docker container.

For further reference on running optional arguments, refer to the [official documentation](https://docs.docker.com/engine/reference/run/#entrypoint-default-command-to-execute-at-runtime) for `docker run`.

## Downloading the trained model

A model pretrained on MUSCIMA++ can be downloaded from the [releases page](https://github.com/apacha/MusicObjectDetector-TF/releases/tag/full-page-detection-v2)