# Nimiq CUDA GPU Mining Client for Nvidia Cards
[![Github All Releases](https://img.shields.io/github/downloads/Sushipool/nimiq-cuda-miner/total.svg)]()

Optimized Nimiq GPU mining client that provides high-performance results, open source codebase, nano protocol, and a **0%** Dev for Nvidia graphics cards.

## Quickstart (Ubuntu/Debian)

1. Install [Node.js](https://github.com/nodesource/distributions/blob/master/README.md#debinstall).
2. Install `git` and `build-essential`: `sudo apt-get install -y git build-essential`.
3. Install [CUDA](https://developer.nvidia.com/cuda-downloads)
4. Clone this repository: `git clone https://github.com/Sushipool/nimiq-cuda-miner`.
5. Build the project: `cd nimiq-cuda-miner && npm install`.
6. Copy miner.sample.conf to miner.conf: `cp miner.sample.conf miner.conf`.
7. Edit miner.conf, specify your wallet address.
8. Run the miner `UV_THREADPOOL_SIZE=12 nodejs index.js`. Ensure UV_THREADPOOL_SIZE is higher than a number of GPU in your system.

## HiveOS Mining FlightSheet
Use the following FlightSheet settings to start mining Nimiq with HiveOS.
![HiveOS](https://github.com/Sushipool/sushi-miner-cuda/blob/master/hiveos-flightsheet.png?raw=true)


## Developer Fee
This client offers a **0%** Dev Fee!

## Drivers Requirements
Please update to the latest Nvidia Cuda 10 drivers.

## Mining Parameters

```
Parameter       Description                                            Data Type

address         Nimiq wallet address                                    [string]
                Example: "address": "NQ...",

host            Pool server address
                Example: "host": "eu.sushipool.com"                     [string]
                
port            Pool server port
                Example: "port": "443"
                Default: 443                                            [number]

consensus       Consensus method used
                Possible values are "dumb" or "nano"
                Note that "dumb" mode (i.e. no consensus) only works with SushiPool.
                Example: "consensus": "nano"                            [string]
                
name            Device name to show in the dashboard                    [string]
                Example: "name": "My Miner"
                
hashrate        Expected hashrate in kH/s                               [number]
                Example: "hashrate": 100
                
devices         GPU devices to use
                Example: "devices": [0,1,2]
                Default: All available GPUs                              [array]
                
memory          Allocated memory in Mb for each device
                Example: "memory": [3072,3840,3840,3840]                 [array]
                
threads         Number of threads per GPU
                Example: "threads": [1,1,2,2]
                Default: 1                                               [array]
```

### Links
Website: https://sushipool.com

Discord: https://discord.gg/JCCExJu

Telegram: https://t.me/SushiPool

Releases: https://github.com/Sushipool/nimiq-cuda-miner/releases

