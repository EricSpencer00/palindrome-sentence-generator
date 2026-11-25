# Build the paper with the ACL style files in ../ (acl.sty, acl_natbib.bst).
# `make numbers` regenerates numbers.tex and figdata/ from the metrics JSON.
main.pdf: main.tex numbers.tex references.bib $(wildcard figdata/*.tex)
	tectonic -Z search-path=.. main.tex

numbers:
	cd ../.. && python3 tools/paper_numbers.py

.PHONY: numbers
