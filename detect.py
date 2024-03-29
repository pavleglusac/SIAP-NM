import argparse
import io
import json
import time
from collections import defaultdict
import boto3
import cv2
import imageio.v2 as imageio
import numpy as np
import torch

s3 = boto3.client('s3')
model = None


def date_before(date1, date2):
    # 2023-12-03T09:05:00
    date1 = date1.split('-')
    date2 = date2.split('-')
    for i in range(3):
        if date1[i] < date2[i]:
            return True
        elif date1[i] > date2[i]:
            return False
    return False


def find_done_files(start_date, area='belgrade'):
    done_files = []
    suffix = '' if area == 'belgrade' else '/horgos'
    for obj in s3.list_objects(Bucket='siap-data', Prefix='detection' + suffix)['Contents']:
        obj_date = obj['Key'].split('/')[1].split('.')[0]
        if date_before(obj_date, start_date):
            continue
        done_files.append(obj['Key'])
    return done_files


def find_done_timestamps(done_files):
    done_timestamps = set()
    for file in done_files:
        if not file.endswith('.json'):
            continue
        # load file
        print(f"Loading {file}")
        obj = s3.get_object(Bucket='siap-data', Key=file)
        detections = json.loads(obj['Body'].read())
        # iterate over timestamps
        for timestamp in detections:
            done_timestamps.add(timestamp)
    return done_timestamps


def find_missing_timestamps(done_timestamps, n_detections=288, start_date='2021-01-01', area='belgrade'):
    missing = []
    paginator = s3.get_paginator('list_objects_v2')

    if area == 'belgrade':
        search_in = 'belgrade/takovska'
    else:
        search_in = 'horgos/entry'

    for page in paginator.paginate(Bucket='siap-cameras', Prefix=search_in):
        for obj in page['Contents']:
            timestamp = obj['Key'].split('/')[2].rsplit('.', 1)[0]
            print(timestamp)
            if date_before(timestamp, start_date):
                continue
            if timestamp not in done_timestamps:
                print("Appending")
                missing.append(timestamp)
            if len(missing) >= n_detections:
                return missing

    return missing


def get_model():
    global model
    if model is None:
        model = torch.hub.load(
            'ultralytics/yolov5',
            'yolov5l', pretrained=True
        )
    return model


def process_timestamp(image_path):
    model = get_model()
    image = download_image(image_path)
    prediction = model(image)
    detections = prediction.xyxy[0]
    output = []
    for *xyxy, conf, cls in detections:
        bbox = torch.tensor(xyxy).tolist()
        detection = {
            'class': cls.item(),
            'confidence': conf.item(),
            'bbox': [round(coord, 3) for coord in bbox]
        }
        output.append(detection)
    return output


def download_image(path):
    print(f"Downloading {path}")
    obj = s3.get_object(Bucket='siap-cameras', Key=path)
    image_data = obj['Body'].read()
    image_stream = io.BytesIO(image_data)
    if path.endswith(".ts"):
        image = imageio.imread(image_stream)
    else:
        image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def process_missing(missing_timestamps, area='belgrade'):
    result = {}
    for timestamp in missing_timestamps:
        print(f"Processing {timestamp}")
        if area == 'belgrade':
            image_path = f"belgrade/takovska/{timestamp}.jpg"
        else:
            image_path = f"horgos/entry/{timestamp}.ts"
        timestamp_result = process_timestamp(image_path)
        result[timestamp] = timestamp_result
    return result


def merge_with_s3(timestamps_per_date, area='belgrade'):
    for date in timestamps_per_date:
        _merge_with_s3(timestamps_per_date, date, area)


def _merge_with_s3(timestamps_per_date, date, area='belgrade'):
    detections = timestamps_per_date[date]
    detections = {timestamp: detections_inner for timestamp, detections_inner in detections}
    if area == 'belgrade':
        obj_key = f"detection/{date}.json"
    else:
        obj_key = f"detection/horgos/{date}.json"

    try:
        obj = s3.get_object(Bucket='siap-data', Key=obj_key)
        existing_detections = json.loads(obj['Body'].read())
        detections = detections | existing_detections
    except:
        pass
    # sort by timestamp
    detections = {k: detections[k] for k in sorted(detections)}
    # print(obj_key)
    s3.put_object(
        Bucket='siap-data',
        Key=obj_key, Body=json.dumps(detections, indent=4)
    )


def main(n_detections, start_date='2021-01-01'):
    area = 'horgos'
    done_files = find_done_files(start_date, area)
    done_timestamps = find_done_timestamps(done_files)
    missing_timestamps = find_missing_timestamps(
        done_timestamps, n_detections, start_date, area
    )
    print(f"Done: {len(done_timestamps)}")
    print(f"Missing: {len(missing_timestamps)}")
    result = process_missing(missing_timestamps, area)
    timestamps_per_date = defaultdict(list)
    for timestamp in result:
        date = timestamp.split('T')[0]
        timestamps_per_date[date].append((timestamp, result[timestamp]))

    merge_with_s3(timestamps_per_date, area)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--start_date', help='Start date',
        type=str, default='2023-12-02'
    )
    parser.add_argument(
        "--detections", help="Number of detections to process", type=int, default=288
    )
    args = parser.parse_args()

    print(f'Starting detections at {time.ctime()}')
    start = time.time()
    main(args.detections, args.start_date)
    print(f'Done in {time.time() - start}')
