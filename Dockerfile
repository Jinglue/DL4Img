FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-scipy \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        libgtk2.0-0 \
        git \
        tcl-dev \
        python-pydot \
        graphviz \
        libffi6   \
        libffi-dev \
        tk-dev && \
    rm -rf /var/lib/apt/lists/*

ADD https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh tmp/Miniconda3-4.2.12-Linux-x86_64.sh
RUN bash tmp/Miniconda3-4.2.12-Linux-x86_64.sh -b
ENV PATH $PATH:/root/miniconda3/bin/

COPY environment_py2.yml ./environment_py2.yml
COPY environment_py3.yml ./environment_py3.yml
RUN conda env create -f=environment_py2.yml --name py2 --debug -v -v
RUN conda env create -f=environment_py3.yml --name py3 --debug -v -v
WORKDIR /root
ADD http://dl4img-1251985129.cosbj.myqcloud.com/cudnn-8.0-linux-x64-v6.0.tgz .
RUN tar -zxvf cudnn-8.0-linux-x64-v6.0.tgz
RUN cp cuda/lib64/libcudnn.so.6.0.21 /usr/lib/libcudnn.so.6

RUN mkdir -p .jupyter
COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888
EXPOSE 6006
WORKDIR /srv/
COPY run.sh /run.sh
RUN chmod +x /run.sh
ENTRYPOINT ["/run.sh"]
