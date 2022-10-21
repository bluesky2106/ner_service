# Named Entity Recognition

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
