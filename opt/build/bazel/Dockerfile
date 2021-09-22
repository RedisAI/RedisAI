# BUILD redisfab/bazel-${ARCH}:$(BAZ_VER) 

ARG OSNICK=bullseye
ARG ARCH=arm64v8

FROM redisfab/${ARCH}-xbuild:${OSNICK}

RUN [ "cross-build-start" ]  

ARG BAZ_VER=0.24.1

WORKDIR /build

RUN set -e ;\
	apt-get -qq update ;\
	apt-get -q install -y pkg-config g++ zlib1g-dev zip unzip python wget ;\
	apt-get -q install -y openjdk-11-jdk

RUN set -e ;\
	wget -q https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-dist.zip ;\
	unzip bazel-0.24.1-dist.zip

RUN	set -e ;\
	env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh ;\
	copy bazel to /usr/local/bin

RUN [ "cross-build-end" ]

# FROM insready/bazel:latest as baz
# FROM ubuntu:latest
# COPY --from=baz /usr/bin/bazel* /usr/local/bin/

