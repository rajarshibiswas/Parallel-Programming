#
# CSE 5441 : Lab 4
# Filename : makefile
# Author   : Rajarshi Biswas (biswas.91@osu.edu)
# 	     The Ohio State University.


all: biswas_rajarshi_lab4p1  biswas_rajarshi_lab4p2
biswas_rajarshi_lab4p1: biswas_rajarshi_lab4p1.cu
	 nvcc -O -o lab4p1 biswas_rajarshi_lab4p1.cu
biswas_rajarshi_lab4p2:
	nvcc -O -o lab4p2 biswas_rajarshi_lab4p2.cu ./lab4_lib/bmp_reader.o
