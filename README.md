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
### Login information
You will need a .env file to connect to the OceanDB database.
You can copy .env.example to .env, and modify as necessary.
```sh
cp .env.example .env
```

### Python dependencies
If you wish to use a virtual environment, you can start one with
```sh
python -m venv .venv
source .venv/bin/activate
```

Install the python dependencies with
```sh
pip install .
```
(Note that if you are NOT using a virtual environment, you may have to use `python -m pip` instead of `pip`.)

## Running
An example can be found in [test.py](./test.py)
Run via
```sh
python test.py
```
Note that this test relies on the database being accessible with permissions in .env.
