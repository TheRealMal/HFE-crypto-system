# Makefile для компиляции LaTeX презентации

TEXFILE = presentation.tex
PDFFILE = presentation.pdf
LATEX = pdflatex

.PHONY: all clean

all: $(PDFFILE) clean

$(PDFFILE): $(TEXFILE)
	$(LATEX) $(TEXFILE)

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.synctex.gz

clean-all: clean
	rm -f $(PDFFILE)

help:
	@echo "Доступные команды:"
	@echo "  make        - Компиляция презентации"
	@echo "  make clean  - Удаление временных файлов"
	@echo "  make clean-all - Удаление всех файлов, включая PDF"
	@echo "  make help   - Показать эту справку"

