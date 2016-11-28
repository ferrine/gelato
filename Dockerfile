FROM yandex/rep:0.6.5

MAINTAINER Maxim Kochurov <maxim.v.kochurov@gmail.com>

# install apt-get dependencies
RUN apt-get -qq update && apt-get install -y \
            # for user friendly docker environment
            bc \
    	    bash-completion \
    	    git \
    	    htop \
    	    tmux \
    	    vim \

    	    # for other stuff
            libopenblas-dev \
            cmake \
            zlib1g-dev \
            libjpeg-dev \
            xvfb \
            libav-tools \
            xorg-dev \
            python-opengl \
            swig \

            # get smaller container size
            && \
            apt-get clean && \
            apt-get autoremove && \
            rm -rf /var/lib/apt/lists/*

# install requirements for python2
RUN /bin/bash --login -c "\
    source activate rep_py2 && \
    pip install --upgrade pip && \
    pip install -U \
    git+git://github.com/Theano/Theano.git \
    git+git://github.com/Lasagne/Lasagne.git \
    git+git://github.com/ferrine/pymc3.git@user_model \
    git+git://github.com/ferrine/gelato.git && \
    source deactivate \
    "

# install requirements for python3
RUN /bin/bash --login -c "\
    source activate jupyterhub_py3 && \
    pip install --upgrade pip && \
    pip install -U \
    git+git://github.com/Theano/Theano.git \
    git+git://github.com/Lasagne/Lasagne.git \
    git+git://github.com/ferrine/pymc3.git@user_model \
    git+git://github.com/ferrine/gelato.git && \
    source deactivate \
    "

# clear pip cache
RUN rm -rf /root/.pip/cache/*
