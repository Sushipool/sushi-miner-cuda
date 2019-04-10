const Nimiq = require('@nimiq/core');
const NativeMiner = require('bindings')('nimiq_miner_cuda.node');

// TODO: configurable interval
const HASHRATE_MOVING_AVERAGE = 5; // seconds
const HASHRATE_REPORT_INTERVAL = 5; // seconds

class Miner extends Nimiq.Observable {

    constructor(allowedDevices, memorySizes, threads) {
        super();

        allowedDevices = Array.isArray(allowedDevices) ? allowedDevices : [];
        memorySizes = Array.isArray(memorySizes) ? memorySizes : [];
        threads = Array.isArray(threads) ? threads : [];

        this._miner = new NativeMiner.Miner();
        this._devices = this._miner.getDevices();
        this._devices.forEach((device, idx) => {
            const enabled = (allowedDevices.length === 0) || allowedDevices.includes(idx);
            if (!enabled) {
                device.enabled = false;
                Nimiq.Log.i(`GPU #${idx}: ${device.name}. Disabled by user.`);
                return;
            }
            if (memorySizes.length > 0) {
                const memory = (memorySizes.length === 1) ? memorySizes[0] : memorySizes[(allowedDevices.length === 0) ? idx : allowedDevices.indexOf(idx)];
                if (Number.isInteger(memory)) {
                    device.memory = memory;
                }
            }
            // TODO: Threads
            Nimiq.Log.i(`GPU #${idx}: ${device.name}, ${device.multiProcessorCount} SM @ ${device.clockRate} MHz. Using ${device.memory} MB.`);
        });

        this._hashes = [];
        this._lastHashRates = [];
    }

    _reportHashRate() {
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
        this.fire('hashrate-changed', averageHashRates);
    }

    setShareCompact(shareCompact) {
        this._miner.setShareCompact(shareCompact);
    }

    startMiningOnBlock(blockHeader) {
        this._miner.startMiningOnBlock(blockHeader, (error, obj) => {
            if (error) {
                throw error;
            }
            if (obj.done === true) {
                return;
            }
            if (obj.nonce > 0) {
                this.fire('share', obj.nonce);
            }
            this._hashes[obj.device] = (this._hashes[obj.device] || 0) + obj.noncesPerRun;
        });
        if (!this._hashRateTimer) {
            this._hashRateTimer = setInterval(() => this._reportHashRate(), 1000 * HASHRATE_REPORT_INTERVAL);
        }
    }

    stop() {
        this._miner.stop();
        if (this._hashRateTimer) {
            this._hashes = [];
            this._lastHashRates = [];
            clearInterval(this._hashRateTimer);
            delete this._hashRateTimer;
        }
    }
}

module.exports = Miner;
