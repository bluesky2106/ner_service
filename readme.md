# Named Entity Recognition

## Prerequisite

Before running the app or pytest (unit test), please set the correct python path. Refer to below command as a reference:

```
export PYTHONPATH="${PYTHONPATH}:/Users/akagi/Projects/01-personal/02-sources/01-mine/ner_service"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Build

Nếu build trên MacOS M1 thì cần chạy lệnh sau :

```
export DOCKER_DEFAULT_PLATFORM=linux/amd64
```

```
docker build --tag akagi2106/named-entity-recognition:v3 .

docker buildx build --no-cache=true --platform linux/amd64 --tag akagi2106/named-entity-recognition:v3 .
```

```
docker login
```

```
docker push akagi2106/named-entity-recognition:v3
```

```
docker run --name ner -p 8080:8080 -e PYTHONPATH="$PYTHONPATH:$(pwd)" akagi2106/named-entity-recognition:v3
```
