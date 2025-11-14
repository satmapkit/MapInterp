
build_image:
	docker build -f build/Dockerfile -t ocean_db_interp:latest .

shell:
	docker-compose run --rm -it map_interp bash