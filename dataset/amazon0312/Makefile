# Description: Amazon product co-purchasing network from March 12 2003

URL := https://www.cise.ufl.edu/research/sparse/RB/SNAP/amazon0312.tar.gz

GRAHP := amazon0312

all : setup

setup :
	wget -N $(URL)
	tar xvf $(GRAHP).tar.gz
	mv $(GRAHP)/* .
	rm $(GRAHP).rb
	rm -rf $(GRAHP)
	rm $(GRAHP).tar.gz

clean :
	rm $(GRAHP).txt