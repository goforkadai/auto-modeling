exec-docker:
	docker exec -it autokeras-container bash
install:
	docker exec autokeras-container bash -c "cd app && pipenv install"
auto:
	docker exec autokeras-container bash -c "cd app/src && pipenv run python auto.py ../../data-directory/dataset/dataset.csv ../../data-directory/model-data/my-model/ ../../data-directory/graph/"
summary:
	docker exec autokeras-container bash -c "cd app/src && pipenv run python summary.py"


predict:
	docker exec autokeras-container bash -c "cd app/src && pipenv run python predict.py ../../data-directory/model-data/my-model"

make run:
	make install
	make auto
