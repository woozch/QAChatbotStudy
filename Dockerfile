FROM ubuntu:latest
# timezone
ENV TZ Asia/Seoul \
    DEBIAN_FRONTEND=noninteractive 
ARG DEBIAN_FRONTEND=noninteractive

# Install basic packages
RUN apt-get update && apt-get install -y \
    build-essential curl libfreetype6-dev libpng-dev libzmq3-dev pkg-config software-properties-common sudo wget

RUN apt-get update && sudo apt-get install -y \
    build-essential \
    net-tools \
    curl \
    git \
    libfreetype6-dev \
    libpng-dev \
    libzmq3-dev \
    pkg-config \
    software-properties-common \
    swig \
    zip \
    zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-tk \ 
    mesa-vulkan-drivers

RUN wget https://sdk.lunarg.com/sdk/download/1.3.275.0/linux/vulkansdk-linux-x86_64-1.3.275.0.tar.xz

# requirements
RUN pip3 install jupyter numpy scipy matplotlib pandas scikit-learn seaborn pytest python-dotenv unstructured markdown
# for LLM
RUN pip3 install langchain openai llamaapi llama-index langchain_openai langchain-community langchain-experimental transformers evaluate
# for data preprocessing
RUN pip3 install grobid-client faiss-cpu PyPDF2 tiktoken gradio praw stackapi sentence-transformers gpt4all wandb 
# chromadb streamlit

# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# set timezone
# RUN apk add --no-cache tzdata
