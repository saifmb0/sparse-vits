FILE_TO_WATCH="paper.tex"

while true; do
	echo "$(date) Starting iteration"

	inotifywait -q -e close_write "$FILE_TO_WATCH"
	echo "Change detcted, reloading"
	pdflatex -interaction=nonstopmode -halt-on-error --output-dir=/tmp "$FILE_TO_WATCH" >/dev/null && \
	echo 'Converted to pdf' && \
	cp /tmp/paper.pdf . && \
	echo "Reloaded" || \
	echo "Failed to convert to pdf"
	done
