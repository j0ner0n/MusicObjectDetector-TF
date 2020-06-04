# Running this demo

1. Build the docker image and tag it. In this case we will call it saty/muscima.
   ```bash
   docker build MusicObjectDetector/demo/ -t saty/muscima
   ```
2. All hand writen music sheets should then be put in a single directory. In this example the directory will be `/sheet_music`
3. Run the docker image with a volume mounted to the directory mapped to `/workspace`
   ```bash
   nvidia-docker run --rm -t -v /sheet_music/:/workspace saty/muscima
   ```
4. Inference will now be performed on all image files in the `/sheet_music` directory. The results will be returned to `/sheet_music/output`

## Using optional arguments

The following arguments are available:
```
usage: standalone_inference_over_image.py [-h]
                    [--detection_inference_graph DETECTION_INFERENCE_GRAPH]
                    [--input_directory INPUT_DIRECTORY]
                    [--detection_label_map DETECTION_LABEL_MAP]
                    [--output_dir OUTPUT_DIR]

Performs detection over input image given a trained detector.

optional arguments:
  -h, --help            show this help message and exit
  --detection_inference_graph DETECTION_INFERENCE_GRAPH
                        Path to the frozen inference graph.
  --input_directory INPUT_DIRECTORY
                        Path to the input image.
  --detection_label_map DETECTION_LABEL_MAP
                        Path to the label map, which maps each categoryname to
                        a unique number. Must be a simple text-file with one
                        mapping per line in the form of:"<number> <label>",
                        e.g. "1 barline".
  --output_dir OUTPUT_DIR
                        where to output results to.
```

To use these arguments, use the following docker commands:
```bash
docker run -rm -v /sheet_music/:/workspace --entrypoint /code/standalone_inference_over_image.py saty/muscima [OPTIONS]
```
Here, we assume that you want to still perform inference on files in the `/sheet_music` directory. The `/sheet_music` directory is then mapped to `/workspace/` in the docker container.
For further reference on running optional arguments, refer to the [docker reference](https://docs.docker.com/engine/reference/run/#entrypoint-default-command-to-execute-at-runtime) for `docker run` 
