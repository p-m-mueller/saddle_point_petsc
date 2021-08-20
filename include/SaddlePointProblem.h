#ifndef _SADDLEPOINTPROBLEM_H_
#define _SADDLEPOINTPROBLEM_H_

#include <petsc.h>
#include "Discretization.h"

PetscErrorCode SolveSaddlePointProblem(const char *); 

PetscErrorCode SolveConstraintLaplaceProblem(Mesh *, Vec *);

#endif
