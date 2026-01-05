# docker build -t ulsa:latest .

docker run \
    -v ~/mounts:/mnt/z/ \
    -v /data/USBMD_datasets:/mnt/USBMD_datasets \
    # -v ~/datasets:/mnt/datasets \
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