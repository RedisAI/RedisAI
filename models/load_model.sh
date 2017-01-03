#../deps/redis/src/redis-cli -x SET foo < graph.pb
../deps/redis/src/redis-cli -x TF.GRAPH foo < graph.pb
../deps/redis/src/redis-cli TF.TENSOR a UINT8 1 2 VALUES 2 3
../deps/redis/src/redis-cli TF.TENSOR b UINT8 1 2 VALUES 2 3
../deps/redis/src/redis-cli TF.RUN foo 2 a a b b c c
#../deps/redis/src/redis-cli --raw TF.DATA c
../deps/redis/src/redis-cli TF.VALUES c

