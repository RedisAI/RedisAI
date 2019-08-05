ARG OS=debian:buster

#----------------------------------------------------------------------------------------------
FROM redis AS redis
FROM ${OS} AS builder

ARG PACK=0
ARG TEST=0

WORKDIR /redisai
COPY --from=redis /usr/local/ /usr/local/

COPY ./deps/readies/ deps/readies/
COPY ./system-setup.py .
COPY ./test/test_requirements.txt test/

RUN ./deps/readies/bin/getpy
RUN ./system-setup.py

COPY ./get_deps.sh .
RUN ./get_deps.sh cpu

ADD ./ /redisai
RUN make all
RUN [[ $PACK == 1 ]] && make pack
RUN [[ $TEST == 1 ]] && make test

#----------------------------------------------------------------------------------------------
FROM redis

RUN set -e; apt-get -qq update; apt-get install -y libgomp1

RUN mkdir -p /usr/lib/redis/modules/

COPY --from=builder /redisai/install/ /usr/lib/redis/modules/

WORKDIR /data
EXPOSE 6379
CMD ["--loadmodule", "/usr/lib/redis/modules/redisai.so", \
        "TF", "/usr/lib/redis/modules//backends/redisai_tensorflow/redisai_tensorflow.so", \
        "TORCH", "/usr/lib/redis/modules//backends/redisai_torch/redisai_torch.so", \
        "ONNX", "/usr/lib/redis/modules//backends/redisai_onnxruntime/redisai_onnxruntime.so"]
