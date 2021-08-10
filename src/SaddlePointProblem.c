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
	Mat		A, B;
	Vec		f, g;
	PetscInt	nCols;
	PetscErrorCode 	ierr;

	ierr = DMCreateMatrix(da_u, &A); CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(da_u, &f); CHKERRQ(ierr); 

	ierr = MatGetSize(A, NULL, &nCols); CHKERRQ(ierr);

	ierr = MatCreate(PETSC_COMM_WORLD, &B); CHKERRQ(ierr);
	ierr = MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, 4, nCols); CHKERRQ(ierr);

	ierr = VecCreate(PETSC_COMM_WORLD, &g); CHKERRQ(ierr);
	ierr = VecSetSizes(g, PETSC_DECIDE, 4); CHKERRQ(ierr);

	ierr = AssembleOperator_Laplace(da_u, da_prop, &A); CHKERRQ(ierr);
	ierr = AssembleRHS_Laplace(da_u, da_prop, &f); CHKERRQ(ierr);
	ierr = AssembleOperator_Constraints(da_u, da_prop, &B); CHKERRQ(ierr);
	ierr = AssembleRHS_Constraints(da_u, da_prop, &g); CHKERRQ(ierr);

	
	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);

	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

	ierr = 0;
	return ierr;
}
