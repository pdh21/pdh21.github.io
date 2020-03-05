---
layout: post
mathjax: true
title:  "Met Office API and Automating on Google Cloud Platform."
date:   2020-03-06 15:25:01 +0100
categories: jekyll update
description: A blog post on how to use the Met Office API and Google Cloud Platform to automate the collection of a forecast training dataset
---
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

As part of my ongoing project around my new solar panel installation, I want to create my own PV forecasts. So far I have looked at [Solcast](https://solcast.com/), who generate PV forecasts for every 30 minutes. There is also the [pvlib](https://pvlib-python.readthedocs.io/en/stable/index.html) Python module that can be used to model PV power output and generate PV forecasts using the Global Forecast System (GFS) weather data. As we saw in the other blog, the simple pvlib model did not perform as well as the Solcast predictions and I think the limiting factor was the weather predictions. Solcast use Satellite data to generate their 30 minute forecasts, however the GFS data only has a resolution of 3 hours and low spatial resolution.

What I would like to do is see if I can come up with my own model that comes close to our beats the Solcast predictions. The first step is to get more accurate and higher time resolution weather forecasts than the GFS data.

The Met Office provide hourly forecasts for 5 days ahead and so is probably a good place to start. They also provide an API through their new [DataHub](https://metoffice.apiconnect.ibmcloud.com/metoffice/production/) which is free to use, with certian restrictions. Unfortunetly, they do not archive the forecasts and so I need to start storing their forecast data to get an initial training set.

I could do this manually by calling the API each day on my Macbook, but I thought this would be a good opportunity to automate the data collection on [Google Cloud Platform](https://console.cloud.google.com/). This is a blog post about how I did it.

## The Met Office API call
To get access to the Met Office API, you need to create an account, register an application and then subscribe to one of the APIs. Its fairly straightforward and they have easy to follow [guide](https://metoffice.apiconnect.ibmcloud.com/metoffice/production/start).

Having got set up, you will be given a `client ID` and `client secret` as authentication. I have stored this in a separate file and read them in so don't accidently publish them online for all to see.

To start, lets just call the API locally.


```python
#import modules 
import http
import json
```


```python
#open config file to get username and password
with open('./config') as fp:
    for line in fp:
        ls = line.split(' ')
        if ls[0] == 'client_id':
            username = ls[1].split('\n')[0]
        elif ls[0] == 'client_secret':
            password = ls[1]
```


```python
#set up connection
conn = http.client.HTTPSConnection("api-metoffice.apiconnect.ibmcloud.com")

#some boolean variables in request
meta_data='False' # Set True if you want to exclude meta data from result
location='True' # Set False if you don't want to include location name

# co-ordinates for where you want forecast
L_long='0.115206'
L_lat='50.778356'

# header information to pass to request
headers = {
    'x-ibm-client-id': client_id,
    'x-ibm-client-secret': client_secret,
    'accept': "application/json"
    }

# Make API call
conn.request("GET", "/metoffice/production/v0/forecasts/point/hourly?excludeParameterMetadata="+meta_data+"&includeLocationName="+location+"&latitude="+L_lat+"&longitude="+L_long, headers=headers)

# get request
res = conn.getresponse()
data = res.read()

```

The output is a json string, so I use json to convert it.


```python
j=json.loads(data.decode("utf-8"))
```

I can also save th json locally if I wish, using the `modelRunDate` property as part of the filename


```python
with open('./'+j['features'][0]['properties']['modelRunDate']+'.json', 'w') as outfile:
    json.dump(j, outfile)
```

You can also use Pandas to convert the json time series data to a Pandas DataFrame


```python
from pandas.io.json import json_normalize
import pandas as pd
```


```python
df=json_normalize(j['features'][0]['properties']['timeSeries'])
#convert time column to pandas datetime format
df['time']=pd.to_datetime(df['time'])
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feelsLikeTemperature</th>
      <th>max10mWindGust</th>
      <th>maxScreenAirTemp</th>
      <th>minScreenAirTemp</th>
      <th>mslp</th>
      <th>precipitationRate</th>
      <th>probOfPrecipitation</th>
      <th>screenDewPointTemperature</th>
      <th>screenRelativeHumidity</th>
      <th>screenTemperature</th>
      <th>significantWeatherCode</th>
      <th>time</th>
      <th>totalPrecipAmount</th>
      <th>totalSnowAmount</th>
      <th>uvIndex</th>
      <th>visibility</th>
      <th>windDirectionFrom10m</th>
      <th>windGustSpeed10m</th>
      <th>windSpeed10m</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.23</td>
      <td>10.046704</td>
      <td>7.024138</td>
      <td>6.999104</td>
      <td>98880</td>
      <td>3.78</td>
      <td>98</td>
      <td>6.62</td>
      <td>97.92</td>
      <td>7.01</td>
      <td>15</td>
      <td>2020-03-05T11:00Z</td>
      <td>1.78</td>
      <td>0</td>
      <td>1</td>
      <td>3234</td>
      <td>57</td>
      <td>9.26</td>
      <td>6.58</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.49</td>
      <td>11.326755</td>
      <td>7.080564</td>
      <td>7.047948</td>
      <td>98740</td>
      <td>3.84</td>
      <td>97</td>
      <td>6.37</td>
      <td>95.87</td>
      <td>7.06</td>
      <td>15</td>
      <td>2020-03-05T12:00Z</td>
      <td>3.38</td>
      <td>0</td>
      <td>1</td>
      <td>3274</td>
      <td>50</td>
      <td>8.75</td>
      <td>6.02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.97</td>
      <td>12.610963</td>
      <td>6.942079</td>
      <td>6.797396</td>
      <td>98620</td>
      <td>5.03</td>
      <td>98</td>
      <td>5.76</td>
      <td>93.31</td>
      <td>6.84</td>
      <td>15</td>
      <td>2020-03-05T13:00Z</td>
      <td>3.38</td>
      <td>0</td>
      <td>1</td>
      <td>3096</td>
      <td>35</td>
      <td>11.32</td>
      <td>6.69</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.60</td>
      <td>12.198853</td>
      <td>6.685241</td>
      <td>6.645752</td>
      <td>98610</td>
      <td>3.06</td>
      <td>97</td>
      <td>5.46</td>
      <td>92.58</td>
      <td>6.66</td>
      <td>15</td>
      <td>2020-03-05T14:00Z</td>
      <td>2.91</td>
      <td>0</td>
      <td>1</td>
      <td>5398</td>
      <td>19</td>
      <td>11.83</td>
      <td>7.10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.31</td>
      <td>13.182303</td>
      <td>6.507196</td>
      <td>6.439559</td>
      <td>98620</td>
      <td>3.03</td>
      <td>97</td>
      <td>5.34</td>
      <td>92.75</td>
      <td>6.50</td>
      <td>15</td>
      <td>2020-03-05T15:00Z</td>
      <td>1.75</td>
      <td>0</td>
      <td>1</td>
      <td>6728</td>
      <td>17</td>
      <td>13.37</td>
      <td>7.36</td>
    </tr>
  </tbody>
</table>
</div>



Lets plot the feel like temperature for the length of the forecast.


```python
import pylab as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
%matplotlib inline
plt.figure(figsize=(10,5))
plt.plot(df['time'],df['feelsLikeTemperature'])
plt.xticks(rotation=90)
plt.xlabel('Time (mm-dd hh)')
plt.ylabel(r'Feels like Temp. ($^\circ$C)')
```




    Text(0, 0.5, 'Feels like Temp. ($^\\circ$C)')




![png](/Figures/2020-03-05-Met_Office_API_Automated/output_14_1.png)


## Automating with Google Cloud Platform
Having got the API call working, the next step is to get it working on Google Cloud Platform so that it runs once a day.

Google provide an Always Free tier, which provides limited access to many common Google Cloud resources, free of charge. We will make use of that resource here. You can sign up for Google Cloud Platform [here](https://console.cloud.google.com)

Once signed in and To automate our data collection process, we need to make use of three Google Cloud Platform areas:

* **Cloud Storage**, to store out Met office forecasts
* **Cloud Functions** to run our function, make API call and save output
* **Cloud Scheduler** to run our function once a day

### Cloud Storage
The first thing we will want to do is create somewhere we can store our data. This is fairly easy to do from the Google Cloud Console.

One the left hand panel, there are numerous options for adding features. Under the Storage section, click on `Storage` which will take you to the Storage Browser. Here you can add a bucket. This is the name given to a particular storage container.

When creating a storage bucket, you will be asked for:
* a unique name ( I called mine met_office_forecast_bucket)
* Location type (I selected Region)
* where to store data (I selected `us-east-1` to stick within *Always Free tier*)
* default storage class (select based on how often you will be accessing data)
* how to control access (I left mine as fine-grained)

Click create and thats it, our storage bucket is ready to dump data in.

### Cloud Functions
There are numerous ways to run compute jobs in Google Cloud Platform:
* **App engine** is fully managed serverless application platform for web and API backends
* **Compute Engine** lets you use virtual machines that run on Google's infrastructure
* **Kubernates Engine** allows you to manage containers
* **Cloud Functions** allow you to deploy code as simple functions
* **Cloud Run** allows you to use containers such as Docker

Since our pipeline is fairly straightforward and can be written as a function, I have used **Cloud Functions**.

The first thing to do is click on Cloud Functions on the left hand panel, which will take you to the Functions browser. Here you can create a new function.

You will be asked for:
* a function name
* select the memory allocated
* what trigger to use (I have used Cloud Pub/Sub). This is the trigger sent by the scheduler to let the function know when to run
* Topic, sent by the scheduler
* Select where source code is (I chose to use the inline editor, other options include ZIP upload, or upload from cloud storage)
* Runtime, here I am using Python 3.7
* main.py (our code)
* requirements.txt (python libraries required by our code)
* Function to execute, whatever we called the function in main.py

My main.py (with client_id an client_scret blocked out) and requirements.txt can be found below.


#### main.py
Most of this is the same as when we called the API before, apart from the last 4 lines, where we use the `google.cloud` module to send data to our cloud storage bucket. Note, I have declared data and context as inputs even though I don't use them. For some reason the Cloud Function is expecting two arguements.


```python
import http
import json
from google.cloud import storage

def get_forecast(data, context):
    conn = http.client.HTTPSConnection("api-metoffice.apiconnect.ibmcloud.com")

    client_id='*********************'
    client_secret='********************'
    meta_data='False'
    location='True'
    L_long='0.115206'
    L_lat='50.778356'
    headers = {
        'x-ibm-client-id': client_id,
        'x-ibm-client-secret': client_secret,
        'accept': "application/json"
        }

    conn.request("GET", "/metoffice/production/v0/forecasts/point/hourly?excludeParameterMetadata="+meta_data+"&includeLocationName="+location+"&latitude="+L_lat+"&longitude="+L_long, headers=headers)

    res = conn.getresponse()
    data = res.read()

    data.decode("utf-8")

    j=json.loads(data.decode("utf-8"))

    client = storage.Client(project='metofficeforecast')
    bucket = client.get_bucket('met_office_forecast_bucket')
    blob = bucket.blob('./'+j['features'][0]['properties']['modelRunDate']+'.json')
    blob.upload_from_string(data=json.dumps(j),
                            content_type='application/json')
```

#### requirements.txt
As of Python 3.7, `http` and `json` are included with Python so do not need to be specified as requirements.
```python
# Function dependencies, for example:
# package>=version
google-cloud-storage
```

### Cloud Scheduler
The last stage is to create a job scheduler. Cloud Scheduler can be found on the left hand side panel,whhich takes you to the Cloud scheduler browser. Here you can create a new job. You will be asked for:
* a unique name
* a description
* Frequency (in unix-cron format) since I want to call it at 16:00 every day, I entered `0 16 * * *`
* Time zone
* Target (the type of trigger). I used Pub/Sub
    * Topic: The name of ther trigger. This needs to be thae name I used in the Cloud Function
    * Payload: arguements to send to the function



### And that is it
The Cloud Scheduler will now send a trigger at 16:00 every day, which the function will pick up and then run, calling the Met Office API, get the forecast for the next five days and save it to our Cloud storage bucket.
