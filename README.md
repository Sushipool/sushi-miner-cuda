## Quickstart (Ubuntu/Debian)

1. Install [Node.js](https://github.com/nodesource/distributions/blob/master/README.md#debinstall).
2. Install `git` and `build-essential`: `sudo apt-get install -y git build-essential`.
3. Install [CUDA](https://developer.nvidia.com/cuda-downloads)
4. Clone this repository: `git clone https://github.com/Sushipool/nimiq-cuda-miner`.
5. Build the project: `cd nimiq-cuda-miner && npm install`.
6. Copy miner.sample.conf to miner.conf: `cp miner.sample.conf miner.conf`.
7. Edit miner.conf, specify your wallet address.
8. Run the miner `UV_THREADPOOL_SIZE=12 nodejs index.js`. Ensure UV_THREADPOOL_SIZE is higher than a number of GPU in your system.

