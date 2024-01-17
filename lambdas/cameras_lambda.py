import urllib3
import boto3
import datetime
import random

def lambda_handler(event, context):
    print("entered handler")
    http = urllib3.PoolManager()

    urls = [
        {"url": "https://stream.uzivobeograd.rs/live/cam_13.jpg", "ext": ".jpg", "random": True, "s3_path": "belgrade/takovska/"},
        {"url": "https://kamere.amss.org.rs/horgos1/horgos11.ts", "ext": ".ts", "random": False, "s3_path": "horgos/entry/"},
        {"url": "https://kamere.amss.org.rs/horgos2/horgos21.ts", "ext": ".ts", "random": False, "s3_path": "horgos/exit/"}
    ]

    for url_info in urls:
        url = url_info["url"]
        print("Doing url", url_info)
        if url_info["random"]:
            url += f"?rand={random.random()}"

        response = http.request('GET', url)
        if response.status != 200:
            print(f"Failed to download image from {url}: {response.status}")
            continue

        filename = datetime.datetime.now().isoformat() + url_info["ext"]
        s3 = boto3.client('s3')
        bucket_name = 'siap-cameras'
        object_name = url_info["s3_path"] + filename

        try:
            s3.put_object(Body=response.data, Bucket=bucket_name, Key=object_name)
            print(f"Image successfully uploaded as {object_name}")
        except Exception as e:
            print(f"Error uploading to S3: {str(e)}")

    return "Processing completed"
