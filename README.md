[![license](https://img.shields.io/github/license/RedisAI/RedisAI.svg)](https://github.com/RedisAI/RedisAI)
[![GitHub issues](https://img.shields.io/github/release/RedisAI/RedisAI.svg)](https://github.com/RedisAI/RedisAI/releases/latest)
[![CircleCI](https://circleci.com/gh/RedisAI/RedisAI/tree/master.svg?style=svg)](https://circleci.com/gh/RedisAI/RedisAI/tree/master)

# RedisAI

A Redis module for serving tensors and executing deep learning graphs.
Expect changes in the API and internals.

## Cloning
If you want to run examples, make sure you have [git-lfs](https://git-lfs.github.com) installed when you clone.

## Building
This will checkout and build Redis and download the libraries for the backends (TensorFlow and PyTorch) for your platform.
```
bash get_deps.sh
make install
```

## Running the server
On Linux
```
LD_LIBRARY_PATH=deps/install/lib ./deps/redis/src/redis-server -loadmodule src/redisai.so
```

On macos
```
DYLD_LIBRARY_PATH=deps/install/lib ./deps/redis/src/redis-server -loadmodule src/redisai.so
```

## Documentation
[Docs](http://redisai.io)

## Mailing List
[RedisAI Google group](https://groups.google.com/forum/#!forum/redisai)

## License

AGPL-3.0 https://opensource.org/licenses/AGPL-3.0

Copyright 2019, Orobix Srl & Redis Labs
