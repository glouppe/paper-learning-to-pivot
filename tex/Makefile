paper.pdf: bibliography.bib paper.tex
	rm -f *.pdf
	pdflatex -shell-escape paper
	pdflatex -shell-escape paper
	rm -f *.log *.out *.aux *.bbl *.blg

full: bibliography.bib paper.tex
	pdflatex -shell-escape paper
	bibtex paper
	pdflatex -shell-escape paper
	pdflatex -shell-escape paper

clean:
	rm -f *.log *.out *.aux *.bbl *.blg *.data paper.pdf
