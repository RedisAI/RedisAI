FROM redis:latest as builder

ENV DEPS "build-essential"

# Set up a build environment
RUN set -ex;\
    deps="$DEPS";\
    apt-get update;\
    apt-get install -y --no-install-recommends $deps;
    
# Build the source
ADD ./ /redisai

# Build RedisAI
WORKDIR /redisai
RUN set -ex;\
	cd RedisModulesSDK/rmutil;\
	make clean;\
	make;\
	cd -;
RUN set -ex;\
	cd src;\
    make clean; \
    make;

# Package the runner
FROM redis:latest
ENV LIBDIR /usr/lib/redis/modules
WORKDIR /data
RUN set -ex;\
    mkdir -p "$LIBDIR";

COPY --from=builder /redisai/src/redisai.so "$LIBDIR"

EXPOSE 6379
CMD ["redis-server", "--loadmodule", "/usr/lib/redis/modules/redisai.so"]
