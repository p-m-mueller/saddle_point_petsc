#include "SaddlePointProblem.h"

PetscErrorCode solveSaddlePointProblem(PetscInt nx, PetscInt ny)
{
	DM		da;
	Vec		u;
	PetscErrorCode 	ierr;

	ierr = setupMesh(nx, ny, &da); CHKERRQ(ierr);
	ierr = solveConstraintLaplaceProblem(da, &u); CHKERRQ(ierr);

	return ierr;
}



PetscErrorCode solveConstraintLaplaceProblem(DM da, Vec *u)
{
	KSP		ksp;
	PetscErrorCode 	ierr;
	
	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);

	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

	ierr = 0;
	return ierr;
}
