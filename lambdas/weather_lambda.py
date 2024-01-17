import boto3
import urllib3
from datetime import datetime


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


def lambda_handler(event, context):
    timestamp = datetime.utcnow().isoformat()
    http = urllib3.PoolManager()
    for i in range (len(city_names)):
        table = dynamodb.Table(table_names[i])

        params = {
            'q': city_names[i],
            'appid': api_key,
            'units': 'metric'
        }
        response = http.request('GET', base_url, fields=params)

        if response.status == 200:
            weather_data = response.data.decode('utf-8')
            item = {'Timestamp' : timestamp,
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
