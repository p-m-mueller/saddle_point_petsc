#ifndef _MESHFUNCTIONS_H_
#define _MESHFUNCTIONS_H_

#include <petsc.h>


#define GAUSS_POINTS 4
#define DIM 2
#define NODES_PER_ELEMENT 4

typedef struct
{
	PetscInt i, j;
}Index2d;

typedef struct 
{
	PetscScalar 	gp_coords[DIM*GAUSS_POINTS];
	PetscScalar	gp_weights[GAUSS_POINTS];
	PetscScalar	f[DIM*GAUSS_POINTS];
}ElementProperties;

PetscErrorCode SetupDMs(PetscInt, PetscInt, DM *, DM *);

PetscErrorCode GetElementOwnershipRanges2d(DM, PetscInt**, PetscInt**); 

PetscErrorCode GetElementCoords(DMDACoor2d **, PetscInt, PetscInt, PetscScalar []);

PetscErrorCode ConstructGaussQuadrature(PetscInt *, PetscScalar [][DIM], PetscScalar []);

PetscErrorCode ConstructQ12D_Ni(PetscScalar *, PetscScalar *); 

PetscErrorCode SetElementForcingTerm(PetscScalar *, PetscScalar *);

#endif
