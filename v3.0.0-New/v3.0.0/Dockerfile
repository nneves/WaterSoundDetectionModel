FROM ubuntu:latest

# Install xxd (part of vim-common)
RUN apt-get update && apt-get install -y vim-common

# Set working directory
WORKDIR /data

# Keep container running
ENTRYPOINT ["/bin/bash"]