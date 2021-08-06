#ifndef _SADDLEPOINTPROBLEM_H_
#define _SADDLEPOINTPROBLEM_H_

#include <petsc.h>
#include "Mesh.h"


PetscErrorCode solveConstraintLaplaceProblem(DM, Vec *);

#endif
