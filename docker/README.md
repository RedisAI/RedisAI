# REDIS-LAB Docker development

In order to run the container you must be on a server with nvidia GPU installed along with its relative drivers (tested with NVIDIA driver v.410.73)

## Build container

```
docker build -t redis-lab .
```

This container is deployed within the orobix docker hub repository `orobix/redis-dl` (provided you have the credentials).

## Launch
In order to launch it 

`nvidia-docker run -d -v LOCAL_DIRECTORY:/home --name orobix/redis-lab-devel orobix/redis-lab-devel`

In order to access it

`docker exec -ti redis-lab-devel bash`

Happy coding!

