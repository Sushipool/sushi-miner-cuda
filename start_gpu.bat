@echo off
echo "------------------------START Miner----------------------"
SET UV_THREADPOOL_SIZE=32
sushi-miner-cuda.exe
echo "------------------------END Miner----------------------"
echo "Something went wrong or you exited"
pause