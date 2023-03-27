FROM python:3.9.13

ARG user
ARG uid

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libpng-dev \
    libonig-dev \
    libxml2-dev \
    zip \
    unzip\
    xz-utils
    
# Clear cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

RUN pip install -U  --upgrade pip

RUN useradd -G root -u $uid -d /home/$user $user
RUN mkdir -p /home/$user/.composer && \
    chown -R $user:$user /home/$user
RUN mkdir -p /home/data && \
    chown -R $user:$user /home/data
RUN mkdir -p /home/output && \
    chown -R $user:$user /home/output

COPY --chown=$user:$user ENIGMA_JVN_analysis.py /home

COPY --chown=$user:$user requirements.txt requirements.txt
RUN pip install -r requirements.txt
ENV PYTHONUNBUFFERED 1

COPY --chown=$user:$user networkx.tar.xz /usr/local/lib/python3.9/site-packages/
WORKDIR /usr/local/lib/python3.9/site-packages/
RUN tar xf networkx.tar.xz && rm networkx.tar.xz

USER $user

CMD ["python3",  "/home/ENIGMA_JVN_analysis.py"]
