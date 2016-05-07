.PHONY: clean analysis paper conditionfiles requirements

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

requirements:
	pip install -r requirements.txt
	
analysis:
	python generate_samples.py
	python random_forest_why.py
	python random_forest_when.py
	