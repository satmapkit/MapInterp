
PYTHON ?= ./venv/bin/python
TEST_ENV = PYTHONPATH=../OceanDB/src:src

test:
	$(TEST_ENV) $(PYTHON) -m unittest discover -s tests -v

build_image:
	docker build -f build/Dockerfile -t ocean_db_interp:latest .

shell:
	docker-compose run --rm -it map_interp bash
