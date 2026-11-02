#!/bin/sh
#SBATCH -p i8cpu
#SBATCH -N 2
#SBATCH -n 32
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH --exclusive

# Option to use dual rail
#export MPI_IB_RAILS=2

#MKL_DEBUG_CPU_TYPE=5
ulimit -s unlimited
module purge
module load oneapi_compiler/2023.0.0 oneapi_mkl/2023.0.0 openmpi/4.1.5-oneapi-2023.0.0-classic

export KMP_STACKSIZE=512m
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/local/ap/hdf5-openmpi/1.10.10/lib
export UCX_TLS='self,sm,ud'

#BINDIR=/work/k0585/k058500/vasp.5.4.4.pl2/bin
BINDIR=/work/k0585/k058500/vasp.6.5.1/bin
EXE=vasp_std
#EXE=vasp_gam
#EXE=vasp_ncl

srun $BINDIR/$EXE > progress 2>&1
#zsh /work/k0585/k058500/common/repeatGeomOpt.sh srun $BINDIR/$EXE


