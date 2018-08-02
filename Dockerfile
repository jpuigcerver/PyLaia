FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

# Set encoding so that python doesn't fail to encode characters
# like \u22f1 (⋱) used in Tensor's string representation
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

WORKDIR /

# TODO: Remove credentials once PyLaia goes public
# Usage:
# $ docker build . -t pylaia --build-arg USR=<usr> --build-arg PW=<pw>
ARG USR
ARG PW
# Install PyLaia
RUN git clone https://"$USR":"$PW"@github.com/jpuigcerver/PyLaia && \
    cd PyLaia && \
    git submodule update --init && \
    python setup.py install

# Install third party libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends cmake &&
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#WORKDIR /PyLaia/third_party/nnutils/pytorch
#RUN python setup.py install

WORKDIR /PyLaia/third_party/warp-ctc/build
RUN cmake .. && \
    make && \
    cd ../pytorch_binding && \
    python setup.py install

WORKDIR /PyLaia
