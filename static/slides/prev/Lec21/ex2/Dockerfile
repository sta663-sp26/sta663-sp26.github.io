FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y r-base && \
    Rscript -e "install.packages('tibble')" && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

CMD ["R", "--vanilla"]
