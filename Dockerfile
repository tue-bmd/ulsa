# Inherit from usbmd base image
FROM zeahub/all:latest

# Install times new roman font (& latex but is commented out)
# If you already had matplotlib installed:
# https://stackoverflow.com/questions/37920935/matplotlib-cant-find-font-installed-in-my-linux-machine
RUN apt-get update && apt install -y ttf-mscorefonts-installer && fc-cache -fv && \
    # apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -fr ~/.cache/matplotlib

# Install tf2jax (and reinstall the right keras version)
RUN KERAS_VER=$(python3 -c "import keras; print(keras.__version__)") \
    && pip install --no-cache-dir tf2jax==0.3.6 \
    && pip install --no-cache-dir "keras==$KERAS_VER"