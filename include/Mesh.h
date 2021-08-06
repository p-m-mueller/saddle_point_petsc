#ifndef _MESHFUNCTIONS_H_
#define _MESHFUNCTIONS_H_

#include <petsc.h>

#define GAUSS_POINTS 4

typedef struct
{
	DM 	da_u;
	DM	da_prop;
	const PetscInt 	dof_u = 2;
};

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

PetscErrorCode setupMesh(PetscInt, PetscInt, DM *);

PetscErrorcode freeMesh(

PetscErrorCode GetElementOwnershipRanges2d(DM, PetscInt**, PetscInt**); 

#endif
