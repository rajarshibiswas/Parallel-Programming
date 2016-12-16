#
# CSE 5441 : Lab 2
# Filename : makefile
# Author   : Rajarshi Biswas (biswas.91@osu.edu)
#	     The Ohio State University.

all: biswas_rajarshi_disposable biswas_rajarshi_persistent 

biswas_rajarshi_disposable: biswas_rajarshi_disposable.cc
	g++ -std=gnu++0x -O3 -fopenmp -o disposable biswas_rajarshi_disposable.cc -I.
biswas_rajarshi_persistent: biswas_rajarshi_persistent.cc 
	g++ -std=gnu++0x -O3 -fopenmp -o persistent biswas_rajarshi_persistent.cc -I.
