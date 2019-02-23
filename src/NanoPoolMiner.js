const Nimiq = require('@nimiq/core');
const NativeMiner = require('bindings')('nimiq_cuda_miner.node');

const HASHRATE_MOVING_AVERAGE = 5; // seconds
const HASHRATE_REPORT_INTERVAL = 5; // seconds

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

        this._hashes = new Array(this._devices.length).fill(0);
        this._lastHashRates = this._hashes.map(_ => []);
    }

    _reportHashRates() {
        this._lastHashRates.forEach((hashRates, idx) => {
            const hashRate = this._hashes[idx] / HASHRATE_REPORT_INTERVAL;
            hashRates.push(hashRate);
            if (hashRates.length > HASHRATE_MOVING_AVERAGE) {
                hashRates.shift();
            }
        });
        this._hashes.fill(0);
        const averageHashRates = this._lastHashRates.map(hashRates => hashRates.reduce((sum, val) => sum + val, 0) / hashRates.length);
        this.fire('hashrates-changed', averageHashRates);
    }

    _startMining() {
        if (!this._hashRateTimer) {
            this._hashRateTimer = setInterval(() => this._reportHashRates(), 1000 * HASHRATE_REPORT_INTERVAL);
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
            this._hashes[obj.device] += obj.noncesPerRun;
        });
    }

    _stopMining() {
        this._miner.stop();
        if (this._hashRateTimer) {
            clearInterval(this._hashRateTimer);
            delete this._hashRateTimer;
        }
    }

    _onNewPoolSettings(address, extraData, shareCompact, nonce) {
        super._onNewPoolSettings(address, extraData, shareCompact, nonce);
        this._miner.setShareCompact(shareCompact);
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

    _turnPoolOff() {
        super._turnPoolOff();
        this._stopMining();
    }

    get devices() {
        return this._devices;
    }
}

module.exports = NanoPoolMiner;
