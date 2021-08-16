#ifndef _VISUALIZATION_H_
#define _VISUALIZATION_H_ 

#include <petsc.h>
#include "Discretization.h"

PetscErrorCode WriteVTK(DM, Vec, const char *);

static int WriteVTKHeader(MPI_Comm, MPI_File, const char *, unsigned int *);

static int WriteVTKPoints(MPI_Comm, MPI_File, const int, const double *, unsigned int *);

static int WriteVTKPolygones(MPI_Comm, MPI_File, const int, const int *, const int, unsigned int *);
static inline int WriteVTKPolygonesLine(MPI_File, const int *, const int, unsigned int *);
#endif
