# Basic Standalone cudf application

Prerequisite:
- Your project should build with cmake 3.18.5

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
docker exec rapidsenvrt sh -c "build/process_csv"
```