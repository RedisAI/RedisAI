# BUILD redisfab/redisai:${VERSION}-cpu-${ARCH}-${OSNICK}

ARG REDIS_VER=6.0.9

# OSNICK=bionic|stretch|buster
ARG OSNICK=buster

# OS=debian:buster-slim|debian:stretch-slim|ubuntu:bionic
ARG OS=debian:buster-slim

# ARCH=x64|arm64v8|arm32v7
ARG ARCH=x64

ARG PACK=0
ARG TEST=0

#----------------------------------------------------------------------------------------------
FROM redisfab/redis:${REDIS_VER}-${ARCH}-${OSNICK} AS redis
FROM ${OS} AS builder

ARG OSNICK
ARG OS
ARG ARCH
ARG REDIS_VER

RUN echo "Building for ${OSNICK} (${OS}) for ${ARCH} [with Redis ${REDIS_VER}]"

WORKDIR /build
COPY --from=redis /usr/local/ /usr/local/

COPY ./opt/ opt/
COPY ./tests/flow/test_requirements.txt tests/flow/

RUN FORCE=1 ./opt/readies/bin/getpy3
RUN ./opt/system-setup.py

ARG DEPS_ARGS=""
COPY ./get_deps.sh .
RUN if [ "$DEPS_ARGS" = "" ]; then ./get_deps.sh cpu; else env $DEPS_ARGS ./get_deps.sh cpu; fi

ARG BUILD_ARGS=""
ADD ./ /build
RUN bash -c "set -e ;\
    . ./opt/readies/bin/sourced ./profile.d ;\
    make -C opt build $BUILD_ARGS SHOW=1"

ARG PACK
ARG TEST

RUN mkdir -p bin/artifacts
RUN set -e ;\
    if [ "$PACK" = "1" ]; then make -C opt pack; fi

RUN set -e ;\
    if [ "$TEST" = "1" ]; then \
        TEST= make -C opt test $BUILD_ARGS NO_LFS=1 ;\
        if [[ -d test/logs ]]; then \
            tar -C test/logs -czf bin/artifacts/test-logs-cpu.tgz . ;\
        fi ;\
    fi

#----------------------------------------------------------------------------------------------
FROM redisfab/redis:${REDIS_VER}-${ARCH}-${OSNICK}

ARG OSNICK
ARG OS
ARG ARCH
ARG REDIS_VER
ARG PACK

RUN if [ ! -z $(command -v apt-get) ]; then apt-get -qq update; apt-get -q install -y libgomp1; fi
RUN if [ ! -z $(command -v yum) ]; then yum install -y libgomp; fi 

ENV REDIS_MODULES /usr/lib/redis/modules
ENV LD_LIBRARY_PATH $REDIS_MODULES

RUN mkdir -p $REDIS_MODULES/

COPY --from=builder /build/install-cpu/ $REDIS_MODULES/

COPY --from=builder /build/bin/artifacts/ /var/opt/redislabs/artifacts

WORKDIR /data
EXPOSE 6379
CMD ["--loadmodule", "/usr/lib/redis/modules/redisai.so"]
