#ifndef _MESHFUNCTIONS_H_
#define _MESHFUNCTIONS_H_

#include <petsc.h>

#define GAUSS_POINTS 4

typedef struct
{
	PetscInt i, j;
}Index2d;

typedef struct 
{
	PetscScalar 	gp_coords[2*GAUSS_POINTS];
	PetscScalar 	eta[GAUSS_POINTS];
	PetscScalar	fx[GAUSS_POINTS];
	PetscScalar	fy[GAUSS_POINTS];
}GaussPointCoefficients;

PetscErrorCode SetupDMs(PetscInt, PetscInt, DM *, DM *);

PetscErrorCode GetElementOwnershipRanges2d(DM, PetscInt**, PetscInt**); 

#endif
