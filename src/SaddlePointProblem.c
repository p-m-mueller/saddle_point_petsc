#include "SaddlePointProblem.h"


PetscErrorCode SolveSaddlePointProblem(PetscInt nx, PetscInt ny)
{
	DM		da_u, da_prop;
	Vec		u;
	PetscErrorCode 	ierr;

	ierr = SetupDMs(nx, ny, &da_u, &da_prop); CHKERRQ(ierr);

	ierr = SolveConstraintLaplaceProblem(da_u, da_prop, &u); CHKERRQ(ierr);

	return ierr;
}


PetscErrorCode SolveConstraintLaplaceProblem(DM da_u, DM da_prop, Vec *u)
{
	KSP		ksp;
	PetscErrorCode 	ierr;
	
	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);

	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

	ierr = 0;
	return ierr;
}
