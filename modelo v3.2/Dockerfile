# Use a base image with a package manager
FROM ubuntu:latest

# Install vim (which includes xxd)
RUN apt-get update && apt-get install -y vim

# Set the working directory
WORKDIR /data

# Command to keep the container running
CMD ["bash"]