# cross python and R environments

# to use:
# docker build -t thistly-cross .
# docker run --rm -ti -v $(pwd):/home/felixity thistly-cross \
#                  python3 marginalizable_state_space_model.py
# to troubleshoot:
# docker run -it --entrypoint /bin/bash thistly-cross

FROM ubuntu:22.04

COPY renv.lock requirements-docker.txt ./

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y python3.10-dev \
    python3-pip \
    r-base-dev \
    g++ \
    fontconfig \
    make \
    cmake \
    git \
    libnlopt-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libxml2-dev \
    msttcorefonts \
    font-manager \
    cm-super \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements-docker.txt \
    && R -e "install.packages('remotes', repos = c(CRAN = 'https://cloud.r-project.org'))" \
    && R -e "remotes::install_github('rstudio/renv')" \
    && R -e "renv::restore()" \
    && useradd felixity

# switch to non-root user
USER felixity
WORKDIR /home/felixity
RUN echo ".libPaths('/usr/local/lib/R/site-library')" >> .Rprofile
ENV PYTHONPATH "${PYTHONPATH}:/home/felixity"
