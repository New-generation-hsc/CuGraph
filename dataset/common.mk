# a general makefile for all graph dataset

all : setup

setup :
	wget -N $(URL) -O $(GRAPH).gz
	gzip -d $(GRAPH).gz

clean :
	rm $(GRAPH)