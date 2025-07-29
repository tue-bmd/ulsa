docker run \
    -v ~/mounts:/mnt/z/ \
    -v ./:/ulsa \
    -w /ulsa \
    --gpus all \
    --rm \
    -it \
    -m 100g \
    --cpus 40 \
    --env-file .env \
    ulsa:latest \
    "$@"