ARG OS=debian:buster

#----------------------------------------------------------------------------------------------
FROM redis AS redis
FROM ${OS} AS builder

WORKDIR /redisai
COPY --from=redis /usr/local/ /usr/local/

COPY ./automation/ automation/
COPY ./test/test_requirements.txt test/

RUN ./automation/readies/bin/getpy
RUN ./automation/system-setup.py

COPY ./get_deps.sh .
RUN ./get_deps.sh cpu

ADD ./ /redisai
RUN make -C automation all

ARG PACK=0
ARG TEST=0

RUN if [ "$PACK" = "1" ]; then make -C automation pack; fi
RUN if [ "$TEST" = "1" ]; then make -C automation test; fi

#----------------------------------------------------------------------------------------------
FROM redis

RUN set -e; apt-get -qq update; apt-get install -y libgomp1

RUN mkdir -p /usr/lib/redis/modules/

COPY --from=builder /redisai/install-cpu/ /usr/lib/redis/modules/

WORKDIR /data
EXPOSE 6379
CMD ["--loadmodule", "/usr/lib/redis/modules/redisai.so"]
