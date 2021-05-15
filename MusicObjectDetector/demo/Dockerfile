# This is a known working version of tensorflow
ARG TENSORFLOW="1.5.1"
ARG CUDA="9.0"
ARG CUDNN="7"

# Get the base images
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

# Set environment settings for LANG and LD_LIBRARY_PATH
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/targets/x86_64-linux/lib/

# Install required packages
RUN apt-get update && apt-get install -y libgtk2.0.0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    wget \
    bzip2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3.6 \
    && conda update conda \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

# Then install ML stuff
RUN pip install opencv-python~=4.2.0 \
    pillow==5.3.0 \
    tensorflow==1.8 \
    tensorflow-gpu==1.8 \
    tqdm

# Get default weights and move scripts/code
RUN mkdir -p /code \
    && wget -O /code/model.pb https://github.com/apacha/MusicObjectDetector-TF/releases/download/full-page-detection-v2/2018-07-30_faster-rcnn_inception-resnet-v2_full-page_muscima.pb

# RUN mkdir -p /code \
#     && wget -O /code/model.pb https://github.com/apacha/MusicObjectDetector-TF/releases/download/crop-images-detection/2018-01-24_faster-rcnn_inception-resnet-v2_crop-image-object-detection_muscima-pp.pb

COPY standalone_inference_over_image.py /code/standalone_inference_over_image.py
COPY category_mapping_fixed.txt /code/category_mapping.txt

# Set the entry point
ENTRYPOINT [ "python", "/code/standalone_inference_over_image.py" ]
CMD [ "--detection_inference_graph", "/code/model.pb", \
      "--input_directory", "/workspace", \
      "--detection_label_map", "/code/category_mapping.txt", \
      "--output_dir", "/workspace/output" ]
