import os
from os.path import isfile, join
import argparse

labels = ",landmark_0.x,landmark_0.y,landmark_0.z,landmark_1.x,landmark_1.y,landmark_1.z,landmark_2.x,landmark_2.y,landmark_2.z,landmark_3.x,landmark_3.y,landmark_3.z,landmark_4.x,landmark_4.y,landmark_4.z,landmark_5.x,landmark_5.y,landmark_5.z,landmark_6.x,landmark_6.y,landmark_6.z,landmark_7.x,landmark_7.y,landmark_7.z,landmark_8.x,landmark_8.y,landmark_8.z,landmark_9.x,landmark_9.y,landmark_9.z,landmark_10.x,landmark_10.y,landmark_10.z,landmark_11.x,landmark_11.y,landmark_11.z,landmark_12.x,landmark_12.y,landmark_12.z,landmark_13.x,landmark_13.y,landmark_13.z,landmark_14.x,landmark_14.y,landmark_14.z,landmark_15.x,landmark_15.y,landmark_15.z,landmark_16.x,landmark_16.y,landmark_16.z,landmark_17.x,landmark_17.y,landmark_17.z,landmark_18.x,landmark_18.y,landmark_18.z,landmark_19.x,landmark_19.y,landmark_19.z,landmark_20.x,landmark_20.y,landmark_20.z,world_landmark_0.x,world_landmark_0.y,world_landmark_0.z,world_landmark_1.x,world_landmark_1.y,world_landmark_1.z,world_landmark_2.x,world_landmark_2.y,world_landmark_2.z,world_landmark_3.x,world_landmark_3.y,world_landmark_3.z,world_landmark_4.x,world_landmark_4.y,world_landmark_4.z,world_landmark_5.x,world_landmark_5.y,world_landmark_5.z,world_landmark_6.x,world_landmark_6.y,world_landmark_6.z,world_landmark_7.x,world_landmark_7.y,world_landmark_7.z,world_landmark_8.x,world_landmark_8.y,world_landmark_8.z,world_landmark_9.x,world_landmark_9.y,world_landmark_9.z,world_landmark_10.x,world_landmark_10.y,world_landmark_10.z,world_landmark_11.x,world_landmark_11.y,world_landmark_11.z,world_landmark_12.x,world_landmark_12.y,world_landmark_12.z,world_landmark_13.x,world_landmark_13.y,world_landmark_13.z,world_landmark_14.x,world_landmark_14.y,world_landmark_14.z,world_landmark_15.x,world_landmark_15.y,world_landmark_15.z,world_landmark_16.x,world_landmark_16.y,world_landmark_16.z,world_landmark_17.x,world_landmark_17.y,world_landmark_17.z,world_landmark_18.x,world_landmark_18.y,world_landmark_18.z,world_landmark_19.x,world_landmark_19.y,world_landmark_19.z,world_landmark_20.x,world_landmark_20.y,world_landmark_20.z,handedness.score,handedness.label,letter"


def load_data_paths(dataset_path):
    images = [join(dataset_path, f) for f in os.listdir(
        dataset_path) if isfile(join(dataset_path, f))]
    return sorted(images)


def read_lines(dataset):
    content = None
    with open(dataset, 'r') as f:
        content = f.read().splitlines()
        if content[0][0] == ',' or content[0][1] == ',':
            content = content[1:]
    return content


def write_dataset(output_path, dataset_path):
    idx = 0
    datasets = load_data_paths(dataset_path)
    with open(output_path, 'w') as output:
        output.write(f"{labels}\n")
        for dataset in datasets:
            output_data = read_lines(dataset)
            for line in output_data:
                line = line.split(',', maxsplit=1)[-1]
                output.write(f"{idx},{line}\n")
                idx += 1


def main(directory, output_file):
    print(f"Directory of dataset: {directory}")
    print(f"Output File: {output_file}")
    write_dataset(output_file, directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process files in a directory and write to an output file")
    parser.add_argument("directory", nargs="?", default='dataset',
                        help="path to the directory of files")
    parser.add_argument("output", nargs="?",
                        default='data_latest.csv', help="path to the output file")
    args = parser.parse_args()
    main(args.directory, args.output)
