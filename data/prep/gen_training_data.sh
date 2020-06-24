#!/bin/sh
F90=ifort
$F90 -o spinup SFMT.f90 common.f90 lorenz96.f90 spinup.f90
./spinup
mv fort.90 fort.10

$F90 -o nature SFMT.f90 common.F90 lorenz96.f90 nature.f90
./nature
mv fort.90 nature.dat

rm -f *.mod
rm -f *.o
rm -f nature spinup
rm -f fort.10
