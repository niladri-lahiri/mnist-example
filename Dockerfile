from ubuntu:20.04
COPY mnist-example /exp/mnist-example
COPY requirements.txt /exp/requirements.txt 
COPY models /exp/models
COPY api /exp/api

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt

WORKDIR /exp
CMD ["python3", "./api/app.py"]

RUN chmod u+r+x ./mnist-example/docker_example.sh
RUN ./mnist-example/docker_example.sh
