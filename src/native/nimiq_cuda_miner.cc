#include <cuda_runtime.h>
#include <nan.h>

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <atomic>
#include <string>
#include <vector>

#include "kernels.h"

class Device;
class MinerWorker;

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
  Device(int device);
  ~Device();

  static NAN_GETTER(HandleGetters);
  static NAN_SETTER(HandleSetters);

  bool IsEnabled();

private:
  friend class MinerWorker;
  int device;
  cudaDeviceProp prop;
  bool enabled = true;
  uint32_t memory;
  uv_mutex_t lock;
  bool workerInitialized = false;
  worker_t worker;
};

class MinerWorker : public Nan::AsyncProgressQueueWorker<uint32_t>
{
public:
  MinerWorker(Nan::Callback *callback, Miner *miner, Device *device, nimiq_block_header blockHeader);
  ~MinerWorker();

  void Execute(const ExecutionProgress &progress);
  void HandleProgressCallback(const uint32_t *data, size_t count);
  void HandleOKCallback();

private:
  Miner *miner;
  Device *device;
  nimiq_block_header blockHeader;
  uint32_t workId;
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
    devices[i] = new Device(i);
  }
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
  if (miner->shareCompact.load() == 0)
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
      Nan::AsyncQueueWorker(new MinerWorker(new Nan::Callback(cbFunc), miner, device, *header));
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

Device::Device(int device) : device(device)
{
  cudaGetDeviceProperties(&prop, device);
  uv_mutex_init(&lock);

  // whole number of GB minus one
  memory = (prop.totalGlobalMem / ONE_GB - 1) * (ONE_GB / ONE_MB);
}

Device::~Device()
{
  if (workerInitialized)
  {
    release_worker(&worker);
  }
  uv_mutex_destroy(&lock);
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

/*
* MinerWorker
*/

MinerWorker::MinerWorker(Nan::Callback *callback, Miner *miner, Device *device, nimiq_block_header blockHeader)
    : AsyncProgressQueueWorker(callback), miner(miner), device(device), blockHeader(blockHeader)
{
  workId = miner->GetWorkId();
}

MinerWorker::~MinerWorker()
{
}

void MinerWorker::Execute(const ExecutionProgress &progress)
{
  uv_mutex_lock(&device->lock);

  if (!device->workerInitialized)
  {
    device->workerInitialized = (0 == initialize_worker(&device->worker, device->device, (size_t)device->memory * ONE_MB));
  }

  if (device->workerInitialized)
  {
    if (0 == set_block_header(&device->worker, &blockHeader))
    {
      while (miner->IsMiningEnabled())
      {
        if (workId != miner->GetWorkId())
        {
          break;
        }
        uint32_t startNonce = miner->GetNextStartNonce(device->worker.nonces_per_run);
        // TODO: Handle startNonce overflow
        uint32_t nonce = mine_nonces(&device->worker, startNonce, miner->GetShareCompact());
        progress.Send(&nonce, 1);
      }
    }
  }
  else
  {
    SetErrorMessage("Could not allocate memory.");
  }

  uv_mutex_unlock(&device->lock);
}

void MinerWorker::HandleProgressCallback(const uint32_t *nonce, size_t count)
{
  Nan::HandleScope scope;

  v8::Local<v8::Object> obj = Nan::New<v8::Object>();
  Nan::Set(obj, Nan::New("done").ToLocalChecked(), Nan::New(false));
  Nan::Set(obj, Nan::New("device").ToLocalChecked(), Nan::New(device->device));
  Nan::Set(obj, Nan::New("noncesPerRun").ToLocalChecked(), Nan::New(device->worker.nonces_per_run));
  Nan::Set(obj, Nan::New("nonce").ToLocalChecked(), Nan::New(*nonce));

  v8::Local<v8::Value> argv[] = {Nan::Null(), obj};
  callback->Call(2, argv, async_resource);
}

void MinerWorker::HandleOKCallback()
{
  Nan::HandleScope scope;

  v8::Local<v8::Object> obj = Nan::New<v8::Object>();
  Nan::Set(obj, Nan::New("done").ToLocalChecked(), Nan::New(true));
  Nan::Set(obj, Nan::New("device").ToLocalChecked(), Nan::New(device->device));
  Nan::Set(obj, Nan::New("noncesPerRun").ToLocalChecked(), Nan::New(device->worker.nonces_per_run));
  Nan::Set(obj, Nan::New("nonce").ToLocalChecked(), Nan::Undefined());

  v8::Local<v8::Value> argv[] = {Nan::Null(), obj};
  callback->Call(2, argv, async_resource);
}

NODE_MODULE(nimiq_cuda_miner, Miner::Init);
