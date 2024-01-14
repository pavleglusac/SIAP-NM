import boto3
import json
import argparse
import torch
import numpy as np
import cv2
from collections import defaultdict
import io
import time

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


def find_done_files(start_date):
    done_files = []
    for obj in s3.list_objects(Bucket='siap-data', Prefix='detection')['Contents']:
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


def find_missing_timestamps(done_timestamps, n_detections=288, start_date='2021-01-01'):
    missing = []
    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket='siap-cameras', Prefix='belgrade/takovska'):
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
        model = torch.hub.load('ultralytics/yolov5',
                               'yolov5l', pretrained=True)
    return model


def process_timestamp(timestamp):
    model = get_model()
    image = download_image(timestamp)
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


def download_image(timestamp):
    image_path = f"belgrade/takovska/{timestamp}.jpg"
    print(f"Downloading {image_path}")
    obj = s3.get_object(Bucket='siap-cameras', Key=image_path)
    image_data = obj['Body'].read()
    image_stream = io.BytesIO(image_data)
    image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def process_missing(missing_timestamps):
    result = {}
    for timestamp in missing_timestamps:
        print(f"Processing {timestamp}")
        timestamp_result = process_timestamp(timestamp)
        result[timestamp] = timestamp_result
    return result


def merge_with_s3(timestamps_per_date):
    for date in timestamps_per_date:
        _merge_with_s3(timestamps_per_date, date)


def _merge_with_s3(timestamps_per_date, date):
    detections = timestamps_per_date[date]
    detections = {timestamp: detections_inner for timestamp, detections_inner in detections}
    try:
        obj = s3.get_object(Bucket='siap-data', Key=f"detection/{date}.json")
        existing_detections = json.loads(obj['Body'].read())
        detections = detections | existing_detections
    except:
        pass
    # sort by timestamp
    detections = {k: detections[k] for k in sorted(detections)}
    s3.put_object(Bucket='siap-data',
                  Key=f"detection/{date}.json", Body=json.dumps(detections, indent=4))


def main(n_detections, start_date='2021-01-01'):
    done_files = find_done_files(start_date)
    done_timestamps = find_done_timestamps(done_files)
    missing_timestamps = find_missing_timestamps(
        done_timestamps, n_detections, start_date)
    print(f"Done: {len(done_timestamps)}")
    print(f"Missing: {len(missing_timestamps)}")
    result = process_missing(missing_timestamps)
    timestamps_per_date = defaultdict(list)
    for timestamp in result:
        date = timestamp.split('T')[0]
        timestamps_per_date[date].append((timestamp, result[timestamp]))

    merge_with_s3(timestamps_per_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', help='Start date',
                        type=str, default='2023-12-20')
    parser.add_argument(
        "--detections", help="Number of detections to process", type=int, default=288*5)
    args = parser.parse_args()

    print(f'Starting detections at {time.ctime()}')
    start = time.time()
    main(args.detections, args.start_date)
    print(f'Done in {time.time() - start}')
