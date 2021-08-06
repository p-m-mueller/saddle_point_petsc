#ifndef _MESHFUNCTIONS_H_
#define _MESHFUNCTIONS_H_

#include <petsc.h>

typedef struct
{
	PetscInt i, j;
}Index2d;

PetscErrorCode GetElementOwnershipRanges2d(DM, PetscInt**, PetscInt**); 

#endif
