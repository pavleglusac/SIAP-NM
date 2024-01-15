import boto3
import json
import argparse
import numpy as np
from collections import defaultdict
import io
import time
from datetime import datetime, timedelta
import pandas as pd

s3 = boto3.client('s3')


def adjust_timestamp(timestamp, minutes, operation='add'):
    timestamp = timestamp.split('.')[0]
    timestamp_dt = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S')
    if operation == 'add':
        new_timestamp = timestamp_dt + timedelta(minutes=minutes)
    elif operation == 'subtract':
        new_timestamp = timestamp_dt - timedelta(minutes=minutes)
    else:
        raise ValueError("Operation must be 'add' or 'subtract'.")

    return new_timestamp.strftime('%Y-%m-%dT%H:%M:%S')


def clean_timestamp(timestamp):
    timestamp = timestamp.split('.')[0]
    timestamp_dt = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S')
    timestamp_dt = timestamp_dt.replace(second=0, microsecond=0)
    ts = timestamp_dt.strftime('%Y-%m-%dT%H:%M:%S')
    return ts


def combine_data(detections_df, weather_df):
    setic = {clean_timestamp(row['Timestamp']): row for index, row in weather_df.iterrows()}
    clean_data = []
    for index, row in detections_df.iterrows():
        timestamp = clean_timestamp(row['Timestamp'])
        detections = row['Detections']
        for delta in range(0, 3):
            adjusted_timestamp_plus = adjust_timestamp(timestamp, delta, 'add')
            adjusted_timestamp_minus = adjust_timestamp(timestamp, delta, 'subtract')

            if adjusted_timestamp_plus in setic:
                timestamp_to_add = adjusted_timestamp_plus
            elif adjusted_timestamp_minus in setic:
                timestamp_to_add = adjusted_timestamp_minus
            else:
                continue

            target = setic[timestamp_to_add]
            datum = {'Timestamp': timestamp, 'Detections': detections, **target.to_dict()}
            clean_data.append(datum)
            break

    df = pd.DataFrame(clean_data)
    print(df.head(20))
    return df


def download_weather_data():
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('Belgrade')

    response = table.scan()
    data = response['Items']
    max_items = 200

    while 'LastEvaluatedKey' in response and max_items > 0:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])

    df = pd.DataFrame(data).sort_values('Timestamp').reset_index()
    return df


def transform_detections(detections):
    clean_detections = {}
    for timestamp, detections_ts in detections.items():
        count = 0
        for detection in detections_ts:
            if int(detection['class']) in [2, 3, 5, 6, 7]:
                count += 1
        clean_detections[timestamp] = count

    df = pd.DataFrame.from_dict(clean_detections, orient='index')
    df.reset_index(inplace=True)
    df.columns = ['Timestamp', 'Detections']
    return df


def download_car_detections():
    paginator = s3.get_paginator('list_objects_v2')
    data = {}
    for page in paginator.paginate(Bucket='siap-data', Prefix='detection'):
        leave = False
        for obj in page['Contents']:
            obj = s3.get_object(Bucket='siap-data', Key=obj['Key'])
            body = obj['Body'].read()
            if not body:
                continue
            detections = json.loads(body)
            data.update(detections)
        if leave:
            break
    return data


def save_data(df):
    df.to_csv('./train.csv')


def main():
    detections: dict = download_car_detections()
    detections_df: pd.DataFrame = transform_detections(detections)
    weather_df: pd.DataFrame = download_weather_data()
    df: pd.DataFrame = combine_data(detections_df, weather_df)
    save_data(df)


if __name__ == '__main__':
    main()
