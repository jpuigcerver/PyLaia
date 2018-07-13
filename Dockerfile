FROM floydhub/pytorch:0.4.0-gpu.cuda9cudnn7-py3.31

# Set encoding so that python doesn't fail to encode characters
# like \u22f1 (⋱) used in Tensor's string representation
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# TODO: Remove credentials once PyLaia goes public
# Usage:
# $ docker build . -t pylaia --build-arg USR=<usr> --build-arg PW=<pw>
ARG USR
ARG PW
# Install PyLaia
# TODO: Point to Puigcerver's repository when this fork is merged
RUN git clone https://"$USR":"$PW"@github.com/carmocca/PyLaia && \
    cd PyLaia && \
    git submodule update --init --recursive && \
    pip install .

# Install third party libraries
WORKDIR /PyLaia/third_party/imgdistort/build
RUN cmake -DPYTORCH_SETUP_PREFIX=/opt/conda .. && \
    make && make install

WORKDIR /PyLaia/third_party/nnutils/pytorch
RUN python setup.py install

WORKDIR /PyLaia/third_party/warp-ctc/build
RUN cmake .. && make && \
    cd ../pytorch_binding && \
    python setup.py install

WORKDIR /PyLaia
