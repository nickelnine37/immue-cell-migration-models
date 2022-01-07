FROM python:3.7.9-buster

RUN useradd --create-home --shell /bin/bash diss
RUN apt update && apt install -y ffmpeg
USER diss
ENV CONTAINER_HOME=/home/diss

ADD ./requirements.txt $CONTAINER_HOME
RUN pip install -r $CONTAINER_HOME/requirements.txt
ENV PATH  "/home/diss/.local/bin:$PATH"

COPY --chown=diss:diss ./src $CONTAINER_HOME

WORKDIR $CONTAINER_HOME
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
