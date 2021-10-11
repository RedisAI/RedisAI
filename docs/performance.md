# Performance

To get an early sense of what RedisAI is capable of, you can test it with:
- [`redis-benchmark`](https://redis.io/topics/benchmarks): Redis includes the redis-benchmark utility that simulates running commands done by N clients at the same time sending M total queries (it is similar to the Apache's ab utility).

- [`memtier_benchmark`](https://github.com/RedisLabs/memtier_benchmark): from [Redis Labs](https://redislabs.com/) is a NoSQL Redis and Memcache traffic generation and benchmarking tool.

-  [`aibench`](https://github.com/RedisAI/aibench):  a collection of Go programs that are used to generate datasets and then benchmark the inference performance of various Model Servers.


This page is intended to provide clarity on how to obtain the benchmark numbers and links to the most recent results. We encourage developers, data scientists, and architects to run these benchmarks for themselves on their particular hardware, datasets, and Model Servers and pull request this documentation with links for the actual numbers.

## Blogs/White-papers that reference RedisAI performance

- [1] [Announcing RedisAI 1.0: AI Serving Engine for Real-Time Applications](https://redislabs.com/blog/redisai-ai-serving-engine-for-real-time-applications/), May 19, 2020


---------------------------------------

## Methodology and Infrastructure checklist
As stated previously we encourage the community to run the benchmarks on their own infrastructure and specific use case. As part of a reproducible and stable metodology we recommend that for each tested version/variation:

- Monitoring should be used to assert that the machines running the the benchmark client do not become the performance bottleneck.

- A minimum of 3 distinct full repetitions, and reported as a result the median (q50), q95, q99, overall achievable inference throughput, and if possible ( and recommended ) the referral to the full spectrum of latencies. Furthermore, benchmarks should be run for a sufficiently long time.

- A full platform description. Ideally, you should run the benchmark and Model Serving instances in separate machines, placed in an optimal networking scenario.

- Considering the weight of the network performance in the overall performance of the Model Serving solution we recommend that in addition to the aibench performance runs described above, to also run baseline netperf TCP_RR, in order to understand the underlying network characteristics.

## Using redis-benchmark

The redis-benchmark program is a quick and useful way to get some figures and evaluate the performance of a RedisAI setup on a given hardware.

By default, redis-benchmark is limited to 1 thread, which should be accounted for to ensure that the redis-benchmark client is not the first bottleneck.

As a rule of thumb you should monitor for errors and the resource usage of any benchmark client tool to ensure that the results being collect are meaningful.

The following example will:
- Use 50 simultaneous clients performing a total of 100000 requests.
- Run the test using the loopback interface.
- Enable multi-threaded client and use 2 benchmark client threads, meaning the benchmark client maximum CPU usage will be 200%.
- Be executed without pipelining (pipeline value of 1).
- If server replies with errors, show them on stdout.
- Benchmark [AI.MODELEXECUTE](https://oss.redislabs.com/redisai/commands/#aimodelexecute) command over a model stored as a key's value using its specified backend and device.

```
redis-benchmark --threads 2 -c 50 -P 1 -n 100000 AI.MODELEXECUTE <key> INPUTS <input_count> <input> [input ...] OUTPUTS <output_count> <output> [output ...]
```

## Using memtier-benchmark

The [`memtier_benchmark`](https://github.com/RedisLabs/memtier_benchmark) utility from [Redis Labs](https://redislabs.com/) is a NoSQL Redis and Memcache traffic generation and benchmarking tool. When compared to redis-benchmark it provides advanced features like multi-command benchmarks, pseudo-random Data, gaussian access pattern, range manipulation, and extended results reporting.

The following example will:
- Use 200 simultaneous clients performing a total of 2000000 requests ( 10000 per client ).
- Run the test using the loopback interface.
- Use 4 benchmark client threads.
- Store the benchmark result in `result.json`.
- Be executed without pipelining (pipeline value of 1).
- If server replies with errors, show them on stdout.
- Benchmark both [AI.MODELEXECUTE](https://oss.redislabs.com/redisai/commands/#aimodelexecute) command a model stored as a key's value using its specified backend and device, and [AI.SCRIPTEXECUTE](https://oss.redislabs.com/redisai/commands/#aiscriptexecute) stored as a key's value on its specified device.

```
memtier_benchmark --clients 50 --threads 4 --requests 10000 --pipeline 1 --json-out-file results.json --command "AI.MODELEXECUTE model_key INPUTS input_count input1 ... OUTPUTS output_count output1 ..." --command "AI.SCRIPTEXECUTE script_key entry_point INPUTS input_count input1 ... OUTPUTS output_count output1 ..."
```

## Using aibench

_AIBench_ is a collection of Go programs that are used to generate datasets and then benchmark the inference performance of various Model Servers. The intent is to make the AIBench extensible so that a variety of use cases and Model Servers can be included and benchmarked.


We recommend that you follow the detailed installation steps [here](https://github.com/RedisAI/aibench#installation) and refer to the per-use [documentation](https://github.com/RedisAI/aibench#current-use-cases).

###  Current DL solutions supported:

- [RedisAI](https://redisai.io): an AI serving engine for real-time applications built by Redis and Tensorwerk, seamlessly plugged into ​Redis.
- [Nvidia Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server): An open source inference serving software that lets teams deploy trained AI models from any framework (TensorFlow, TensorRT, PyTorch, ONNX Runtime, or a custom framework), from local storage or Google Cloud Platform or AWS S3 on any GPU- or CPU-based infrastructure.
- [TorchServe](https://pytorch.org/serve/): built and maintained by Amazon Web Services (AWS) in collaboration with Facebook, TorchServe is available as part of the PyTorch open-source project.
- [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving): a high-performance serving system, wrapping TensorFlow and maintained by Google.
- [Common REST API serving](https://redisai.io): a common DL production grade setup with Gunicorn (a Python WSGI HTTP server) communicating with Flask through a WSGI protocol, and using TensorFlow as the backend.

### Current use cases

Currently, aibench supports two use cases:
- **creditcard-fraud [[details here](https://github.com/RedisAI/aibench/blob/master/docs/creditcard-fraud-benchmark/description.md)]**: from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) with the extension of reference data. This use-case aims to detect a fraudulent transaction based on anonymized credit card transactions and reference data.


- **vision-image-classification[[details here](dhttps://github.com/RedisAI/aibench/blob/master/ocs/vision-image-classification-benchmark/description.md)]**: an image-focused use-case that uses one network “backbone”: MobileNet V1, which can be considered as one of the standards by the AI community. To assess inference performance we’re recurring to COCO 2017 validation dataset (a large-scale object detection, segmentation, and captioning dataset).

