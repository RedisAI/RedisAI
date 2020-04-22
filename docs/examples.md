# RedisAI Examples
The following sections consist of various sample projects and python notebook examples showing the uses for RedisAI.

To contribute your example (and get the credit for it), click the "Edit this page" button at the top to submit a Pull Request.

## Sample projects
This is a list of RedisAI sample projects that can be used as-is or as an inspiration source.

| Example | Description | Author | License | URL |
| --- | --- | --- | --- | --- |
| ChatBotDemo | An example of using RedisAI and a Web App for Conversational AI (i.e. chatbot) | [RedisLabs](https://redislabs.com/) | Apache-2.0 | [git](https://github.com/RedisAI/ChatBotDemo) |
| AnimalRecognitionDemo | An example of using Redis Streams, RedisGears and RedisAI for Realtime Video Analytics (i.e. filtering cats) | [RedisLabs](https://redislabs.com/) | BSD-3-Clause | [git](https://github.com/RedisGears/AnimalRecognitionDemo) |
| EdgeRealtimeVideoAnalytics | An example of using Redis Streams, RedisGears, RedisAI and RedisTimeSeries for Realtime Video Analytics (i.e. counting people) | [RedisLabs](https://redislabs.com/) | Apache-2.0 | [git](https://github.com/RedisGears/EdgeRealtimeVideoAnalytics) |

## Python notebook examples

### Visual Recognition Models


#### Resnet50

Here is an example of Python code that classifies ImageNet classes with ResNet50 using a trained tensorflow model.

**Author: [RedisLabs](https://redislabs.com/)**

**Python notebook example**

```python
{{ include('notebooks/tensorflow_resnet50.py') }}
```
**Sample output**

```bash
{{ include('notebooks/tensorflow_resnet50.output') }}
```

<img src="images/cat_classified.jpg" alt="cat classified image"/> 


#### YOLO - You only look once

Here is an example of Python code that classifies images with object detection by YOLO algorithm with bounding boxes using a trained tensorflow model.

**Author: [RedisLabs](https://redislabs.com/)**

**Python notebook example**

```python
TBD
```
**Sample output**

```bash
TBD
```

### Natural Language Classification


Here is an example of Python code that classifies text using a trained Natural Language Classifier model.

**Author: [RedisLabs](https://redislabs.com/)**

**Python notebook example**

```python
TBD
```
**Sample output**

```bash
TBD
```
