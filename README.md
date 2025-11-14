# MapInterp
MapInterp is a set of tools for interpolating ocean data.
Data is presumed to come from [OceanDB](https://github.com/Nazanne/OceanDB).


## Installation
Once OceanDB is installed, MapInterp should be cloned into a parallel directory,
i.e.
```
root-dir/
├─ OceanDB/
├─ MapInterp/

```

From here, docker can be used to start an interactive shell, via
```sh
make shell
```

## Running
An example can be found in [test.py](./test.py)
Run from the docker container via
```sh
python test.py
```
