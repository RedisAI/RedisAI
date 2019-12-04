# BUILD redisfab/redisai-cpu-${OSNICK}:M.m.b-${ARCH}

ARG REDIS_VER=5.0.7

# OSNICK=bionic|stretch|buster
ARG OSNICK=buster

# OS=debian:buster-slim|debian:stretch-slim|ubuntu:bionic
ARG OS=debian:buster-slim

# ARCH=x64|arm64v8|arm32v7
ARG ARCH=x64

#----------------------------------------------------------------------------------------------
FROM redisfab/redis:${REDIS_VER}-${ARCH}-${OSNICK} AS redis
FROM ${OS} AS builder

WORKDIR /build
COPY --from=redis /usr/local/ /usr/local/

COPY ./opt/ opt/
COPY ./test/test_requirements.txt test/

RUN ./opt/readies/bin/getpy
RUN ./opt/system-setup.py

ARG DEPS_ARGS=""
COPY ./get_deps.sh .
RUN if [ "$DEPS_ARGS" = "" ]; then ./get_deps.sh cpu; else env $DEPS_ARGS ./get_deps.sh cpu; fi

ARG BUILD_ARGS=""
ADD ./ /build
RUN make -C opt build $BUILD_ARGS SHOW=1

ARG PACK=0
ARG TEST=0

RUN if [ "$PACK" = "1" ]; then make -C opt pack; fi
RUN if [ "$TEST" = "1" ]; then make -C opt test $BUILD_ARGS NO_LFS=1; fi

#----------------------------------------------------------------------------------------------
# FROM redis:latest
FROM redisfab/redis:${REDIS_VER}-${ARCH}-${OSNICK}

RUN set -e; apt-get -qq update; apt-get -q install -y libgomp1

ENV REDIS_MODULES /usr/lib/redis/modules
ENV LD_LIBRARY_PATH $REDIS_MODULES

RUN mkdir -p $REDIS_MODULES/

COPY --from=builder /build/install-cpu/ $REDIS_MODULES/

WORKDIR /data
EXPOSE 6379
CMD ["--loadmodule", "/usr/lib/redis/modules/redisai.so"]
