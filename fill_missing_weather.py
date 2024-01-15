import boto3

import boto3
import urllib3
from datetime import datetime, timedelta
import time

region_name = 'eu-north-1'
table_names = ['Belgrade', 'Horgos']

dynamodb = boto3.resource('dynamodb', region_name=region_name)

city_names = ['horgos, rs', 'belgrade, rs']
secret_name = "openweather_api_key"
region_name = "eu-north-1"

session = boto3.session.Session()
secret_client = session.client(
    service_name='secretsmanager',
    region_name=region_name
)

api_key = eval(secret_client.get_secret_value(SecretId=secret_name)['SecretString'])['OPENWEATHER_API_KEY']

base_url = "http://api.openweathermap.org/data/2.5/weather"


def get_timestamp(timestamp):
    http = urllib3.PoolManager()
    unixtime = time.mktime(timestamp.timetuple())
    for i in range(len(city_names)):
        table = dynamodb.Table(table_names[i])

        params = {
            'q': city_names[i],
            'appid': api_key,
            'units': 'metric',
            'dt': unixtime
        }
        response = http.request('GET', base_url, fields=params)

        if response.status == 200:
            weather_data = response.data.decode('utf-8')
            item = {'Timestamp': timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                    'WeatherDescription': eval(weather_data)['weather'][0]['description'],
                    'WeatherMain': eval(weather_data)['weather'][0]['main'],
                    'Temperature': str(eval(weather_data)['main']['temp'])
                    }
            try:
                response = table.put_item(Item=item)
                print(f'Item inserted successfully: {item}')
            except Exception as e:
                print(f'Error inserting item: {e}')
        else:
            print(f"Failed to retrieve weather data. Status code: {response.status}")


def adjust_timestamp(timestamp_dt, minutes, operation='add'):
    if operation == 'add':
        new_timestamp = timestamp_dt + timedelta(minutes=minutes)
    elif operation == 'subtract':
        new_timestamp = timestamp_dt - timedelta(minutes=minutes)
    else:
        raise ValueError("Operation must be 'add' or 'subtract'.")

    return new_timestamp


start = '2023-12-02T00:00:00'
end = '2023-12-10T00:00:00'

start_ts = datetime.strptime(start, '%Y-%m-%dT%H:%M:%S')
end_ts = datetime.strptime(end, '%Y-%m-%dT%H:%M:%S')

while start_ts < end_ts:
    print(f'Getting timestamp {start_ts}')
    get_timestamp(start_ts)
    start_ts = adjust_timestamp(start_ts, 5, 'add')
    time.sleep(0.5)
