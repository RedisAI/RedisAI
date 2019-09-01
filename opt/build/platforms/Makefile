
ROOT=../..

all: centos7 xenial bionic stretch buster

ARGS=-f $(ROOT)/Dockerfile --build-arg TEST=1 --build-arg PACK=1 $(ROOT)

centos7:
	@docker build -t redisai:centos7 --build-arg OS=centos:7.6.1810 $(ARGS)

xenial:
	@docker build -t redisai:xenial --build-arg OS=ubuntu:xenial $(ARGS)

bionic:
	@docker build -t redisai:bionic --build-arg OS=ubuntu:xenial $(ARGS)

stretch:
	@docker build -t redisai:stretch --build-arg OS=debian:stretch $(ARGS)

buster:
	@docker build -t redisai:buster --build-arg OS=debian:buster $(ARGS)
