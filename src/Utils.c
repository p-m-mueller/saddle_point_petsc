#include "Utils.h"

PetscReal Determinant2x2(const PetscScalar *A)
{
	return A[0]*A[3] - A[1]*A[2];
}

PetscErrorCode InvertMatrix2x2(const PetscScalar *A, PetscScalar *invA)
{
	PetscScalar	detJ = Determinant2x2(A);

	invA[0] =  A[3] / detJ;
	invA[1] = -A[1] / detJ;
	invA[2] = -A[2] / detJ;
	invA[3] =  A[0] / detJ;

	return 0;
}

PetscErrorCode Transform(const PetscInt dim, const PetscReal *J, const PetscReal *v, PetscReal *Jv)
{
	PetscInt i, j;
	
	for (i = 0; i < dim; ++i)
	{
		Jv[i] = 0.0;
		for (j = 0; j < dim; ++j)
			Jv[i] += J[i*dim+j] * v[j];
	}		

	return 0;
}
