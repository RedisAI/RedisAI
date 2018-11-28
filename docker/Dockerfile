FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

#Install deps
RUN apt-get -y update && apt-get install -y \
    libgomp1 \
    curl \
    vim

#Install nodejs
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash && apt-get install -y nodejs

#Volume for code development
VOLUME /home

#Empty command in order to let the docker long run
CMD ["tail", "-f", "/dev/null"]
