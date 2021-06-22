all: docs tests linter dist

linter:
	flake8 julius && mypy -p julius


tests:
	coverage run -m unittest discover -s tests || exit 1
	coverage report --include 'julius/*'
	coverage html --include 'julius/*'

docs:
	pdoc3 --template-dir pdoc --html -o docs -f julius
	cp logo.png docs/

dist:
	python3 setup.py sdist

clean:
	rm -r docs dist build *.egg-info

live:
	pdoc3 --http : julius bench --template-dir pdoc

gen:
	python3 -m bench.gen > bench.md


.PHONY: tests docs dist
