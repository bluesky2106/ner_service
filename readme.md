# Named Entity Recognition

## Prerequisite

Before running the app or pytest (unit test), please set the correct python path. Refer to below command as a reference:

```
export PYTHONPATH="${PYTHONPATH}:/Users/akagi/Projects/01-personal/02-sources/ner_service"
```

## Build

```
docker build --tag akagi2106/named-entity-recognition:v2 .
```

```
docker login
```

```
docker push akagi2106/named-entity-recognition:v2
```

```
docker run --name ner -p 8080:8080 akagi2106/named-entity-recognition:v2
```

