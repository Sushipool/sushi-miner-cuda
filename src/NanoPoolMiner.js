const Nimiq = require('@nimiq/core');
const NativeMiner = require('bindings')('nimiq_miner_cuda.node');

const HASHRATE_MOVING_AVERAGE = 5; // seconds
const HASHRATE_REPORT_INTERVAL = 5; // seconds

const SHARE_WATCHDOG_INTERVAL = 180; // seconds

class NanoPoolMiner extends Nimiq.NanoPoolMiner {

    constructor(blockchain, time, address, deviceId, deviceData, allowedDevices, memorySizes) {
        super(blockchain, time, address, deviceId, deviceData);

        allowedDevices = Array.isArray(allowedDevices) ? allowedDevices : [];
        memorySizes = Array.isArray(memorySizes) ? memorySizes : [];

        this._miner = new NativeMiner.Miner();
        this._devices = this._miner.getDevices();
        this._devices.forEach((device, idx) => {
            const enabled = (allowedDevices.length === 0) || allowedDevices.includes(idx);
            if (!enabled) {
                device.enabled = false;
                Nimiq.Log.i(NanoPoolMiner, `GPU #${idx}: ${device.name}. Disabled by user.`);
                return;
            }
            if (memorySizes.length > 0) {
                const memory = (memorySizes.length === 1) ? memorySizes[0] : memorySizes[(allowedDevices.length === 0) ? idx : allowedDevices.indexOf(idx)];
                if (Number.isInteger(memory)) {
                    device.memory = memory;
                }
            }
            Nimiq.Log.i(NanoPoolMiner, `GPU #${idx}: ${device.name}, ${device.multiProcessorCount} SM @ ${device.clockRate} MHz. Using ${device.memory} MB.`);
        });

        this._hashes = [];
        this._lastHashRates = [];
        this._sharesFound = 0;
    }

    _reportHashRates() {
        const averageHashRates = [];
        this._hashes.forEach((hashes, idx) => {
            const hashRate = hashes / HASHRATE_REPORT_INTERVAL;
            this._lastHashRates[idx] = this._lastHashRates[idx] || [];
            this._lastHashRates[idx].push(hashRate);
            if (this._lastHashRates[idx].length > HASHRATE_MOVING_AVERAGE) {
                this._lastHashRates[idx].shift();
            }
            averageHashRates[idx] = this._lastHashRates[idx].reduce((sum, val) => sum + val, 0) / this._lastHashRates[idx].length;
        });
        this._hashes = [];
        this.fire('hashrates-changed', averageHashRates);
    }

    _startMining() {
        if (!this._hashRateTimer) {
            this._hashRateTimer = setInterval(() => this._reportHashRates(), 1000 * HASHRATE_REPORT_INTERVAL);
        }
        if (!this._shareWatchDog) {
            this._shareWatchDog = setInterval(() => this._checkIfSharesFound(), 1000 * SHARE_WATCHDOG_INTERVAL);
        }
        const block = this.getNextBlock();
        if (!block) {
            return;
        }
        Nimiq.Log.i(NanoPoolMiner, `Starting work on block #${block.height}`);
        this._miner.startMiningOnBlock(block.header.serialize(), (error, obj) => {
            if (error) {
                throw error;
            }
            if (obj.done === true) {
                return;
            }
            if (obj.nonce > 0) {
                this._submitShare(block, obj.nonce);
            }
            this._hashes[obj.device] = (this._hashes[obj.device] || 0) + obj.noncesPerRun;
        });
    }

    _stopMining() {
        this._miner.stop();
        if (this._hashRateTimer) {
            this._hashes = [];
            this._lastHashRates = [];
            clearInterval(this._hashRateTimer);
            delete this._hashRateTimer;
        }
        if (this._shareWatchDog) {
            clearInterval(this._shareWatchDog);
            delete this._shareWatchDog;
        }
    }

    _onNewPoolSettings(address, extraData, shareCompact, nonce) {
        super._onNewPoolSettings(address, extraData, shareCompact, nonce);
        if (Nimiq.BlockUtils.isValidCompact(shareCompact)) {
            const difficulty = Nimiq.BlockUtils.compactToDifficulty(shareCompact);
            Nimiq.Log.i(NanoPoolMiner, `Set share difficulty: ${difficulty.toFixed(2)} (${shareCompact.toString(16)})`);
            this._miner.setShareCompact(shareCompact);
        } else {
            Nimiq.Log.w(NanoPoolMiner, `Pool sent invalid target: ${shareCompact}`);
        }
    }

    async _handleNewBlock(msg) {
        await super._handleNewBlock(msg);
        this._startMining();
    }

    async _submitShare(block, nonce) {
        const blockHeader = block.header.serialize();
        blockHeader.writePos -= 4;
        blockHeader.writeUint32(nonce);
        const hash = await (await Nimiq.CryptoWorker.getInstanceAsync()).computeArgon2d(blockHeader);
        this.onWorkerShare({
            block,
            nonce,
            hash: new Nimiq.Hash(hash)
        });
    }

    _onBlockMined(block) {
        super._onBlockMined(block);
        this._sharesFound++;
    }

    _checkIfSharesFound() {
        Nimiq.Log.d(NanoPoolMiner, `Shares found since the last check: ${this._sharesFound}`);
        if (this._sharesFound > 0) {
            this._sharesFound = 0;
            return;
        }
        Nimiq.Log.w(NanoPoolMiner, `No shares have been found for the last ${SHARE_WATCHDOG_INTERVAL} seconds. Reconnecting.`);
        this._timeoutReconnect();
    }

    _turnPoolOff() {
        super._turnPoolOff();
        this._stopMining();
    }

    get devices() {
        return this._devices;
    }
}

module.exports = NanoPoolMiner;
