#include "SaddlePointProblem.h"
#include "Visualization.h"

PetscErrorCode SolveSaddlePointProblem(const char *filename)
{
	DM		da;
	Vec		u;
	PetscErrorCode 	ierr;

	ierr = CreateMesh(PETSC_COMM_WORLD, filename, &da); CHKERRQ(ierr);

	ierr = DMCreateGlobalVector(da, &u); CHKERRQ(ierr);

	ierr = SolveConstraintLaplaceProblem(da, &u); CHKERRQ(ierr);

	ierr = VecViewFromOptions(u, NULL, "-solution_view"); CHKERRQ(ierr);

	ierr = WriteVTK(da, u, "test.vtk"); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode SolveConstraintLaplaceProblem(DM da, Vec *u)
{
	KSP		ksp;
	Mat		A, B;
	Vec		f, g;
	PetscInt	nCols;
	PetscErrorCode 	ierr;

	ierr = DMCreateMatrix(da, &A); CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(da, &f); CHKERRQ(ierr); 

	/*
	ierr = MatGetSize(A, NULL, &nCols); CHKERRQ(ierr);

	ierr = MatCreate(PETSC_COMM_WORLD, &B); CHKERRQ(ierr);
	ierr = MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, 4, nCols); CHKERRQ(ierr);

	ierr = VecCreate(PETSC_COMM_WORLD, &g); CHKERRQ(ierr);
	ierr = VecSetSizes(g, PETSC_DECIDE, 4); CHKERRQ(ierr);
	*/
	ierr = AssembleOperator_Laplace(da, &A); CHKERRQ(ierr);
	ierr = AssembleRHS_Laplace(da, &f); CHKERRQ(ierr);
	ierr = ApplyBC_Laplace(da, &A, &f); CHKERRQ(ierr); 
	/*
	ierr = AssembleOperator_Constraints(da, da_prop, &B); CHKERRQ(ierr);
	ierr = AssembleRHS_Constraints(da, da_prop, &g); CHKERRQ(ierr);
	*/

	ierr = MatViewFromOptions(A, NULL, "-A_mat_view"); CHKERRQ(ierr);
	ierr = VecViewFromOptions(f, NULL, "-f_vec_view"); CHKERRQ(ierr);

	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
	ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
	ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
	ierr = KSPSetUp(ksp);

	ierr = KSPSolve(ksp, f, *u); CHKERRQ(ierr);

	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

	ierr = 0;
	return ierr;
}
