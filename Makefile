.PHONY: data smoke check test test-all
data:        ## generate the full 90-day dataset (Parquet if pyarrow is installed)
	python -m sim.cli generate --days 90 --out data/world_v1
smoke:       ## 5-day end-to-end run + validation (~1s)
	python -m sim.cli check --days 5
check:       ## full 90-day run, print every check
	python -m sim.cli check
test:        ## fast tests
	python -m pytest
test-all:    ## include the full-scale marginals test
	python -m pytest -m "slow or not slow"
