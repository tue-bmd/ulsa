# Inherit from zea base image
FROM zeahub/all:v0.0.9

# Install latex and fonts
# https://stackoverflow.com/questions/37920935/matplotlib-cant-find-font-installed-in-my-linux-machine
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing dvipng texlive-latex-extra texlive-fonts-recommended cm-super && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -fr ~/.cache/matplotlib

# Install tf2jax (and reinstall the right keras version)
RUN KERAS_VER=$(python3 -c "import keras; print(keras.__version__)") \
    && pip install --no-cache-dir tf2jax==0.3.6 \
    && pip install --no-cache-dir "keras==$KERAS_VER" pandas jaxwt SimpleITK

COPY . /ulsa
WORKDIR /ulsa

RUN pip install --no-cache-dir -e zea \
    && pip install --no-cache-dir -e .