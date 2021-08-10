#ifndef _SADDLEPOINTPROBLEM_H_
#define _SADDLEPOINTPROBLEM_H_

#include <petsc.h>
#include "Discretization.h"

PetscErrorCode SolveSaddlePointProblem(PetscInt, PetscInt); 

PetscErrorCode SolveConstraintLaplaceProblem(DM, Vec, DM, Vec *);

#endif
