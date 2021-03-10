# Basic Standalone cudf application

This project contains a simple example demonstrating basic libcudf use case. It also show case ways
to build your own application based on libcudf in a minimalistic style.

The example source code includes operations to load a csv file that contains stock prices from 4 companies
spanning across 5 days, to compute the average of the closing price for each company and write the result
in csv format.

## How to compile and execute

### Step 1: build environment in docker (only perform once)
```bash
docker build . -t rapidsenv
```

### Step 2: start the container
```bash
docker run -t -d -v $PWD:/workspace --gpus all --name rapidsenvrt rapidsenv
```

### (Perform when active container is running) Configure project
```bash
docker exec rapidsenvrt sh -c "cmake -S . -B build/"
```

### Build project
```bash
docker exec rapidsenvrt sh -c "cmake --build build/ --parallel $PARALLEL_LEVEL"
```
The first time running this command will take a long time because it will build libcudf on the host machine. It may be sped up by configuring the proper `PARALLEL_LEVEL` number.

### Execute binary
```bash
docker exec rapidsenvrt sh -c "build/libcudf_example"
```