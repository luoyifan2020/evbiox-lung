FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y git curl
COPY env/evbiox.yml /tmp/
RUN pip install --no-cache-dir mamba==0.25.1 && \
    mamba env create -f /tmp/evbiox.yml && \
    mamba clean --all -y
SHELL ["conda", "run", "-n", "evbiox", "/bin/bash", "-c"]
ENV PATH /opt/conda/envs/evbiox/bin:$PATH
WORKDIR /workspace
