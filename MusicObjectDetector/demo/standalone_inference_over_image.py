import argparse
import os
import os.path as osp

# Ignore future warnings from numpy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image, ImageColor


COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections',
                'detection_boxes',
                'detection_scores',
                'detection_classes'
            ]:
                tensor_name = key + ':0'

                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            return output_dict


def load_detection_graph(path_to_checkpoint):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def build_map(path_to_labelmap):
    int2category = {}
    lines = open(path_to_labelmap, 'r').read().splitlines()

    for line in lines:
        integer, category = line.split()
        int2category[int(integer)] = category

    return int2category


def parse_args():
    parser = argparse.ArgumentParser(
        description='Performs detection over input image given a trained '
                    'detector.'
    )
    parser.add_argument('--detection_inference_graph', type=str,
                        help='Path to the frozen inference graph.')
    parser.add_argument('--input_directory', type=str,
                        help='Path to the input image.')
    parser.add_argument('--detection_label_map', type=str,
                        default="category_mapping.txt",
                        help='Path to the label map, which maps each category'
                             'name to a unique number. Must be a simple '
                             'text-file with one mapping per line in the form '
                             'of:"<number> <label>", e.g. "1 barline".')
    parser.add_argument('--output_dir', type=str,
                        help='where to output results to.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # The next line forces Tensorflow on Windows to run the computation on CPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Make sure the output dir exists
    os.makedirs(args.output_dir)

    # Build category map
    detection_cat_mapping = build_map(args.detection_label_map)
    class_idx_mapping = {value: key
                         for key, value in detection_cat_mapping.items()}

    # Read frozen graphs
    detection_graph = load_detection_graph(args.detection_inference_graph)

    # Get all files in the input directory
    images = os.listdir(args.input_directory)
    valid_extensions = ('.jpg', '.png', '.jpeg', '.tiff', '.bmp', '.gif')
    valid_images = []
    for img in images:
        if img.lower().endswith(valid_extensions):
            valid_images.append(img)

    # start the session.
    with detection_graph.as_default():
        with tf.Session() as sess:
            default_graph = tf.get_default_graph()
            ops = default_graph.get_operations()
            all_tensor_names = {output.name
                                for op in ops
                                for output in op.outputs}
            tensor_dict = {}
            keys_to_check = ('num_detections', 'detection_boxes',
                             'detection_scores', 'detection_classes')
            for key in keys_to_check:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[
                        key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            # Now iterate through images
            print("running detection")
            image_arrays = []
            raw_images = []
            for img_name in tqdm(valid_images):
                # Open each file as a PIL
                img_fp = osp.join(args.input_directory, img_name)

                # Opencv Image (draw)
                image_cv = cv2.imread(img_fp)
                # Resize everything to the same size. This size is A4 at 144 DPI
                image_cv = cv2.resize(image_cv, (1190, 1684))
                height, width, _ = image_cv.shape
                raw_images.append({
                    'image_name': img_name,
                    'image': image_cv,
                    'height': height,
                    'width': width
                })

                # Numpy image
                image_arrays.append(np.array(image_cv))

                if len(image_arrays) < 1:
                    continue

                # Otherwise prepare for detection
                if len(image_arrays) > 1:
                    stacked_array = np.stack(image_arrays)
                else:
                    stacked_array = np.expand_dims(image_arrays[0], 0)
                image_arrays = []  # Reset the image_arrays list

                # Actual detection
                image_tensor = default_graph.get_tensor_by_name(
                    'image_tensor:0'
                )

                session_result = sess.run(
                    tensor_dict,
                    feed_dict={image_tensor: stacked_array}
                )

                # Post process detections
                processed_dictionaries = []
                for i in range(len(session_result['num_detections'])):
                    processed_dictionaries.append({
                        'num_detections':
                            int(session_result['num_detections'][i]),
                        'detection_classes':
                            session_result['detection_classes'][i]
                                .astype(np.uint8),
                        'detection_boxes':
                            session_result['detection_boxes'][i],
                        'detection_scores':
                            session_result['detection_scores'][i]
                    })

                output_lines = ['x1,y1,x2,y2,det_class']

                for raw_image, out_dict in zip(raw_images,
                                               processed_dictionaries):
                    image_cv = raw_image['image']
                    img_name = raw_image['image_name']
                    image_height = raw_image['height']
                    image_width = raw_image['width']
                    for idx in range(out_dict['num_detections']):
                        if out_dict['detection_scores'][idx] > 0.5:
                            # Only draw rectangles with a score above 0.5
                            y1, x1, y2, x2 = out_dict['detection_boxes'][idx]

                            y1 = y1 * image_height
                            y2 = y2 * image_height
                            x1 = x1 * image_width
                            x2 = x2 * image_width
                            detected_class = detection_cat_mapping[
                                out_dict['detection_classes'][idx]
                            ]

                            output_line = "{:.3f},{:.3f},{:.3f},{:.3f};{}"\
                                .format(x1, y1, x2, y2, detected_class)
                            output_lines.append(output_line)

                            # Draw the bbox onto the image
                            color_idx = class_idx_mapping[detected_class] % len(COLORS)
                            color_name = COLORS[color_idx]
                            color_rgb = ImageColor.getrgb(color_name)
                            cv2.rectangle(image_cv, (int(x1), int(y1)),
                                          (int(x2), int(y2)), color_rgb, 3)

                        else:
                            # Once we get to a score below 0.5, the rest will presumably
                            # also be below 0.5 so we stop
                            break

                    cv2.imwrite(osp.join(args.output_dir, img_name), image_cv)

                    text_file_fp = osp.join(args.output_dir,
                                            osp.splitext(img_name)[0] + '.csv')

                    with open(text_file_fp, "w") as output_file:
                        output_file.write("\n".join(output_lines))

                raw_images = []
