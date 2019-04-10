#include <cuda_runtime.h>
#include <nan.h>

#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "kernels.h"

typedef Nan::AsyncBareProgressQueueWorker<uint32_t>::ExecutionProgress MinerProgress;

class Device;

class Miner : public Nan::ObjectWrap
{
public:
  explicit Miner(int deviceCount);
  ~Miner();

  static NAN_MODULE_INIT(Init);
  static NAN_METHOD(New);
  static NAN_METHOD(GetDevices);
  static NAN_METHOD(SetShareCompact);
  static NAN_METHOD(StartMiningOnBlock);
  static NAN_METHOD(Stop);

  uint32_t GetShareCompact();
  bool IsMiningEnabled();
  uint32_t GetNextStartNonce(uint32_t noncesPerRun);
  uint32_t GetWorkId();

private:
  static Nan::Persistent<v8::Function> constructor;

  int deviceCount;
  std::vector<Device *> devices;
  std::atomic_uint_fast32_t shareCompact;
  std::atomic_bool miningEnabled;
  std::atomic_uint_fast32_t workId;
  std::atomic_uint_fast32_t startNonce;
};

class Device
{
public:
  Device(Miner *miner, int device);
  ~Device();

  static NAN_GETTER(HandleGetters);
  static NAN_SETTER(HandleSetters);

  bool IsEnabled();
  uint32_t GetNoncesPerRun();
  uint32_t GetDeviceIndex();
  void MineNonces(nimiq_block_header *blockHeader, const MinerProgress &progress);

private:
  void Initialize();
  void SetBlockHeader(struct nimiq_block_header *blockHeader);

  Miner *miner;
  int device;
  cudaDeviceProp prop;
  bool enabled = true;
  uint32_t memory;
  std::mutex mutex;
  bool initialized = false;
  worker_t worker;
};

class MinerWorker : public Nan::AsyncProgressQueueWorker<uint32_t>
{
public:
  MinerWorker(Nan::Callback *callback, Device *device, nimiq_block_header blockHeader);

  void Execute(const MinerProgress &progress);
  void HandleProgressCallback(const uint32_t *data, size_t count);
  void HandleOKCallback();

private:
  Device *device;
  nimiq_block_header blockHeader;
};

/*
* Miner
*/

Nan::Persistent<v8::Function> Miner::constructor;

Miner::Miner(int deviceCount) : deviceCount(deviceCount)
{
  devices.resize(deviceCount);
  for (int i = 0; i < deviceCount; i++)
  {
    devices[i] = new Device(this, i);
  }

  shareCompact = 0;
  miningEnabled = false;
  workId = 0;
  startNonce = 0;
}

Miner::~Miner()
{
  for (int i = 0; i < deviceCount; i++)
  {
    delete devices[i];
  }
}

uint32_t Miner::GetShareCompact()
{
  return shareCompact.load();
}

bool Miner::IsMiningEnabled()
{
  return miningEnabled.load();
}

uint32_t Miner::GetNextStartNonce(uint32_t noncesPerRun)
{
  return startNonce.fetch_add(noncesPerRun);
}

uint32_t Miner::GetWorkId()
{
  return workId.load();
}

NAN_MODULE_INIT(Miner::Init)
{
  v8::Local<v8::FunctionTemplate> tpl = Nan::New<v8::FunctionTemplate>(New);
  tpl->SetClassName(Nan::New("Miner").ToLocalChecked());
  tpl->InstanceTemplate()->SetInternalFieldCount(1);

  Nan::SetPrototypeMethod(tpl, "getDevices", GetDevices);
  Nan::SetPrototypeMethod(tpl, "setShareCompact", SetShareCompact);
  Nan::SetPrototypeMethod(tpl, "startMiningOnBlock", StartMiningOnBlock);
  Nan::SetPrototypeMethod(tpl, "stop", Stop);

  constructor.Reset(Nan::GetFunction(tpl).ToLocalChecked());
  Nan::Set(target, Nan::New("Miner").ToLocalChecked(), Nan::GetFunction(tpl).ToLocalChecked());
}

NAN_METHOD(Miner::New)
{
  if (!info.IsConstructCall())
  {
    return Nan::ThrowError(Nan::New("Miner() must be called with new keyword.").ToLocalChecked());
  }

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount < 1)
  {
    return Nan::ThrowError(Nan::New("Could not initialize miner. No CUDA devices found.").ToLocalChecked());
  }

  Miner *miner = new Miner(deviceCount);
  miner->Wrap(info.This());
  info.GetReturnValue().Set(info.This());
}

NAN_METHOD(Miner::GetDevices)
{
  Miner *miner = Nan::ObjectWrap::Unwrap<Miner>(info.This());
  v8::Local<v8::Array> devices = Nan::New<v8::Array>(miner->deviceCount);
  for (int i = 0; i < miner->deviceCount; i++)
  {
    v8::Local<v8::Object> device = Nan::New<v8::Object>();
    Nan::SetPrivate(device, Nan::New("device").ToLocalChecked(), v8::External::New(info.GetIsolate(), miner->devices[i]));
    Nan::SetAccessor(device, Nan::New("name").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("clockRate").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("memoryClockRate").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("memoryBusWidth").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("memoryBandwidth").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("multiProcessorCount").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("totalGlobalMem").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("sharedMemPerBlock").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("major").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("minor").ToLocalChecked(), Device::HandleGetters);
    Nan::SetAccessor(device, Nan::New("enabled").ToLocalChecked(), Device::HandleGetters, Device::HandleSetters);
    Nan::SetAccessor(device, Nan::New("memory").ToLocalChecked(), Device::HandleGetters, Device::HandleSetters);
    devices->Set(i, device);
  }
  info.GetReturnValue().Set(devices);
}

NAN_METHOD(Miner::SetShareCompact)
{
  if (!info[0]->IsUint32())
  {
    return Nan::ThrowError(Nan::New("Invalid share compact.").ToLocalChecked());
  }
  // TODO: Check if valid target

  uint32_t shareCompact = Nan::To<uint32_t>(info[0]).FromJust();
  Miner *miner = Nan::ObjectWrap::Unwrap<Miner>(info.This());
  miner->shareCompact.store(shareCompact);
}

NAN_METHOD(Miner::StartMiningOnBlock)
{
  if (!info[0]->IsUint8Array())
  {
    return Nan::ThrowError(Nan::New("Block header required.").ToLocalChecked());
  }
  v8::Local<v8::Uint8Array> blockHeader = info[0].As<v8::Uint8Array>();
  if (blockHeader->Length() != sizeof(nimiq_block_header))
  {
    return Nan::ThrowError(Nan::New("Invalid block header size.").ToLocalChecked());
  }
  nimiq_block_header *header = (nimiq_block_header *)blockHeader->Buffer()->GetContents().Data();

  if (!info[1]->IsFunction())
  {
    return Nan::ThrowError(Nan::New("Callback required.").ToLocalChecked());
  }
  v8::Local<v8::Function> cbFunc = info[1].As<v8::Function>();

  Miner *miner = Nan::ObjectWrap::Unwrap<Miner>(info.This());
  if (miner->shareCompact == 0)
  {
    return Nan::ThrowError(Nan::New("Share compact is not set.").ToLocalChecked());
  }

  miner->miningEnabled.store(true); // miningEnabled = true
  miner->workId.fetch_add(1);       // workId++
  miner->startNonce.store(0);       // startNonce = 0
  // TODO: Make startNonce consistent across threads. It can be incremented by the worker mining stale block.

  int enabledDevices = 0;
  for (int i = 0; i < miner->deviceCount; i++)
  {
    Device *device = miner->devices[i];
    if (device->IsEnabled())
    {
      Nan::AsyncQueueWorker(new MinerWorker(new Nan::Callback(cbFunc), device, *header));
      enabledDevices++;
    }
  }

  if (enabledDevices == 0)
  {
    return Nan::ThrowError(Nan::New("Can't start mining - all devices are disabled.").ToLocalChecked());
  }
}

NAN_METHOD(Miner::Stop)
{
  Miner *miner = Nan::ObjectWrap::Unwrap<Miner>(info.This());
  miner->miningEnabled.store(false); // miningEnabled = false
}

/*
* Device
*/

Device::Device(Miner *miner, int device) : miner(miner), device(device)
{
  cudaGetDeviceProperties(&prop, device);

  // whole number of GB minus one
  memory = (prop.totalGlobalMem / ONE_GB - 1) * (ONE_GB / ONE_MB);
}

Device::~Device()
{
  if (initialized)
  {
    cudaFree(worker.memory);
    cudaFree(worker.inseed);
    cudaFree(worker.nonce);
  }
}

NAN_GETTER(Device::HandleGetters)
{
  v8::Local<v8::Value> ext = Nan::GetPrivate(info.This(), Nan::New("device").ToLocalChecked()).ToLocalChecked();
  Device *device = (Device *)ext.As<v8::External>()->Value();

  std::string propertyName = std::string(*Nan::Utf8String(property));
  if (propertyName == "name")
  {
    info.GetReturnValue().Set(Nan::New(device->prop.name).ToLocalChecked());
  }
  else if (propertyName == "clockRate")
  {
    info.GetReturnValue().Set(device->prop.clockRate / 1e3); // MHz
  }
  else if (propertyName == "memoryClockRate")
  {
    info.GetReturnValue().Set(device->prop.memoryClockRate / 1e3); // MHz
  }
  else if (propertyName == "memoryBusWidth")
  {
    info.GetReturnValue().Set(device->prop.memoryBusWidth);
  }
  else if (propertyName == "memoryBandwidth")
  {
    info.GetReturnValue().Set((2.0 * device->prop.memoryClockRate * device->prop.memoryBusWidth / 8) / 1e6); // GB/s
  }
  else if (propertyName == "multiProcessorCount")
  {
    info.GetReturnValue().Set(device->prop.multiProcessorCount);
  }
  else if (propertyName == "totalGlobalMem")
  {
    info.GetReturnValue().Set((double)device->prop.totalGlobalMem);
  }
  else if (propertyName == "sharedMemPerBlock")
  {
    info.GetReturnValue().Set((double)device->prop.sharedMemPerBlock);
  }
  else if (propertyName == "major")
  {
    info.GetReturnValue().Set(device->prop.major);
  }
  else if (propertyName == "minor")
  {
    info.GetReturnValue().Set(device->prop.minor);
  }
  else if (propertyName == "enabled")
  {
    info.GetReturnValue().Set(device->enabled);
  }
  else if (propertyName == "memory")
  {
    info.GetReturnValue().Set(device->memory);
  }
}

NAN_SETTER(Device::HandleSetters)
{
  v8::Local<v8::Value> ext = Nan::GetPrivate(info.This(), Nan::New("device").ToLocalChecked()).ToLocalChecked();
  Device *device = (Device *)ext.As<v8::External>()->Value();

  std::string propertyName = std::string(*Nan::Utf8String(property));
  if (propertyName == "enabled")
  {
    if (!value->IsBoolean())
    {
      return Nan::ThrowError(Nan::New("Boolean value required.").ToLocalChecked());
    }
    device->enabled = Nan::To<bool>(value).FromJust();
  }
  else if (propertyName == "memory")
  {
    if (!value->IsUint32())
    {
      return Nan::ThrowError(Nan::New("Positive integer value required.").ToLocalChecked());
    }
    device->memory = Nan::To<uint32_t>(value).FromJust();
  }
}

bool Device::IsEnabled()
{
  return enabled;
}

uint32_t Device::GetNoncesPerRun()
{
  return worker.nonces_per_run;
}

uint32_t Device::GetDeviceIndex()
{
  return device;
}

void Device::Initialize()
{
  if (initialized)
  {
    return;
  }

  cudaSetDevice(device);
  cudaDeviceReset();
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync); // cudaDeviceScheduleAuto
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  uint32_t nonces_per_run = ((size_t) memory * ONE_MB) / (sizeof(block_g) * NIMIQ_ARGON2_COST);
  nonces_per_run = (nonces_per_run / BLAKE2B_THREADS_PER_BLOCK) * BLAKE2B_THREADS_PER_BLOCK;
  size_t mem_size = sizeof(block_g) * NIMIQ_ARGON2_COST * nonces_per_run;

  worker.nonces_per_run = nonces_per_run;

  cudaError_t result = cudaMalloc(&worker.memory, mem_size);
  if (result != cudaSuccess)
  {
    // TODO Exception
  }

  result = cudaMalloc(&worker.inseed, sizeof(initial_seed));
  if (result != cudaSuccess)
  {
    // TODO Exception
  }

  result = cudaMalloc(&worker.nonce, sizeof(uint32_t));
  if (result != cudaSuccess)
  {
    // TODO Exception
  }

  worker.init_memory_blocks = dim3(nonces_per_run / BLAKE2B_THREADS_PER_BLOCK);
  worker.init_memory_threads = dim3(BLAKE2B_THREADS_PER_BLOCK, 2);

  worker.argon2_blocks = dim3(1, nonces_per_run);
  worker.argon2_threads = dim3(THREADS_PER_LANE, 1);

  worker.get_nonce_blocks = dim3(nonces_per_run / BLAKE2B_THREADS_PER_BLOCK);
  worker.get_nonce_threads = dim3(BLAKE2B_THREADS_PER_BLOCK);

  initialized = true;
}

void Device::SetBlockHeader(struct nimiq_block_header *blockHeader)
{
  initial_seed inseed;
  inseed.lanes = 1;
  inseed.hash_len = ARGON2_HASH_LENGTH;
  inseed.memory_cost = NIMIQ_ARGON2_COST;
  inseed.iterations = 1;
  inseed.version = 0x13;
  inseed.type = 0;
  inseed.header_len = sizeof(nimiq_block_header);
  memcpy(&inseed.header, blockHeader, sizeof(nimiq_block_header));
  inseed.salt_len = NIMIQ_ARGON2_SALT_LEN;
  memcpy(&inseed.salt, NIMIQ_ARGON2_SALT, NIMIQ_ARGON2_SALT_LEN);
  inseed.secret_len = 0;
  inseed.extra_len = 0;
  memset(&inseed.padding, 0, sizeof(inseed.padding));

  cudaSetDevice(device);

  cudaError_t result = cudaMemcpy(worker.inseed, &inseed, sizeof(initial_seed), cudaMemcpyHostToDevice);
  if (result != cudaSuccess)
  {
    // TODO: Exception
  }

  cudaMemset(worker.nonce, 0, sizeof(uint32_t)); // zero nonce
}

void Device::MineNonces(nimiq_block_header *blockHeader, const MinerProgress &progress)
{
  std::lock_guard<std::mutex> lock(mutex);

  Initialize();

  SetBlockHeader(blockHeader);

  uint32_t workId = miner->GetWorkId();
  while (miner->IsMiningEnabled())
  {
    if (workId != miner->GetWorkId())
    {
      break;
    }
    uint32_t startNonce = miner->GetNextStartNonce(GetNoncesPerRun());
    // TODO: Handle startNonce overflow
    uint32_t nonce = mine_nonces(&worker, startNonce, miner->GetShareCompact());
    progress.Send(&nonce, 1);
  }

  // TODO: Catch: SetErrorMessage("Could not allocate memory.");
}

/*
* MinerWorker
*/

MinerWorker::MinerWorker(Nan::Callback *callback, Device *device, nimiq_block_header blockHeader)
    : AsyncProgressQueueWorker(callback), device(device), blockHeader(blockHeader)
{
}

void MinerWorker::Execute(const MinerProgress &progress)
{
  device->MineNonces(&blockHeader, progress);
}

void MinerWorker::HandleProgressCallback(const uint32_t *nonce, size_t count)
{
  Nan::HandleScope scope;

  v8::Local<v8::Object> obj = Nan::New<v8::Object>();
  Nan::Set(obj, Nan::New("done").ToLocalChecked(), Nan::New(false));
  Nan::Set(obj, Nan::New("device").ToLocalChecked(), Nan::New(device->GetDeviceIndex()));
  Nan::Set(obj, Nan::New("noncesPerRun").ToLocalChecked(), Nan::New(device->GetNoncesPerRun()));
  Nan::Set(obj, Nan::New("nonce").ToLocalChecked(), Nan::New(*nonce));

  v8::Local<v8::Value> argv[] = {Nan::Null(), obj};
  callback->Call(2, argv, async_resource);
}

void MinerWorker::HandleOKCallback()
{
  Nan::HandleScope scope;

  v8::Local<v8::Object> obj = Nan::New<v8::Object>();
  Nan::Set(obj, Nan::New("done").ToLocalChecked(), Nan::New(true));
  Nan::Set(obj, Nan::New("device").ToLocalChecked(), Nan::New(device->GetDeviceIndex()));
  Nan::Set(obj, Nan::New("noncesPerRun").ToLocalChecked(), Nan::New(device->GetNoncesPerRun()));
  Nan::Set(obj, Nan::New("nonce").ToLocalChecked(), Nan::Undefined());

  v8::Local<v8::Value> argv[] = {Nan::Null(), obj};
  callback->Call(2, argv, async_resource);
}

NODE_MODULE(nimiq_miner_cuda, Miner::Init);
