#ifndef _UTILS_H_
#define _UTILS_H_

#include <petsc.h>

PetscReal Determinant2x2(const PetscScalar *);

PetscErrorCode InvertMatrix2x2(const PetscScalar *A, PetscScalar *invA);

PetscErrorCode Transform(const PetscInt, const PetscReal *, const PetscReal *, PetscReal *);

#endif
