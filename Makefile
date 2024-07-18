all:
	make -C src
	mv src/CircleDet .

clean:
	rm -f CircleDet
	make -C src clean