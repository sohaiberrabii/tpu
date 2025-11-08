## Tensor Processing Unit in Amaranth

### Cocotb Simulation
requires verilator

- simple tiled matmul tests
```shell
pytest -s test/cocotb_test/test_tpu.py
```

- train, quantize and run cocotb simulation of a simple mnist MLP
```shell
COCOTB=1 python examples/mnist.py
```

### FPGA
requires vivado, pynq-z2 board

1. generate the verilog (tpu.v, config.json) then perform implementation with vivado to generate (tpu.bit, tpu.hwh)
```shell
python fpga/helpers.py
```
outputs: fpga/build/{tpu.v,config.json,tpu.bit,tpu.hwh}

2. train simple mlp on mnist to generate quantized model (model.json, tensors.bin)
```shell
TRAIN=1 python examples/mnist.py
```
outputs: examples/assets/{model.json,tensors.bin}

3. ssh to the board then clone and install the repo
```shell
git clone https://github.com/sohaiberrabii/tpu.git
cd tpu
sudo pip install -e .
```
5. copy the generated artifacts (config.json, tpu.bit, tpu.hwh, model.json, tensors.bin) to the tpu repo on the board
6. run mnist inference on the board
```shell
sudo fpga/run.sh
```
