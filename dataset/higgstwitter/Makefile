# The Higgs dataset has been built after monitoring the spreading processes on Twitter before, 
# during and after the announcement of the discovery of a new particle with the features of the elusive Higgs boson on 4th July 2012.

URL := http://snap.stanford.edu/data/higgs-social_network.edgelist.gz
DATA := higgs-social_network.edgelist
GRAPH := higgstwitter

all : setup

setup :
	wget -N $(URL)
	gzip -d $(DATA).gz
	mv $(DATA) $(GRAPH).txt

clean :
	rm $(GRAPH).txt