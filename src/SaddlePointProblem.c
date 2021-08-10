#include "SaddlePointProblem.h"


/*
 * Input:	nx	PetscInt, the number of elements in x direction
 * 		ny 	PetscInt, the number of elements in y direction 
 */
PetscErrorCode SolveSaddlePointProblem(PetscInt nx, PetscInt ny)
{
	DM		da_u, da_prop;
	Vec		properties, u;
	PetscErrorCode 	ierr;

	ierr = SetupDMs(nx, ny, &da_u, &da_prop, &properties); CHKERRQ(ierr);

	ierr = DMCreateGlobalVector(da_u, &u); CHKERRQ(ierr);

	ierr = SolveConstraintLaplaceProblem(da_prop, properties, da_u, &u); CHKERRQ(ierr);

	ierr = VecViewFromOptions(u, NULL, "-solution_view"); CHKERRQ(ierr);


	return ierr;
}

/*
 * Input: 	da_prop		DM which manages the properties data
 * 		properties	Vec that contains the element properties data (Gauss points, weights, ...)
 * 		da_u		DM which manages the solution data 
 * Output:
 * 		u		Vec that contains the solution data
 */
PetscErrorCode SolveConstraintLaplaceProblem(DM da_prop, Vec properties, DM da_u, Vec *u)
{
	KSP		ksp;
	Mat		A, B;
	Vec		f, g;
	PetscInt	nCols;
	PetscErrorCode 	ierr;

	ierr = DMCreateMatrix(da_u, &A); CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(da_u, &f); CHKERRQ(ierr); 

	/*
	ierr = MatGetSize(A, NULL, &nCols); CHKERRQ(ierr);

	ierr = MatCreate(PETSC_COMM_WORLD, &B); CHKERRQ(ierr);
	ierr = MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, 4, nCols); CHKERRQ(ierr);

	ierr = VecCreate(PETSC_COMM_WORLD, &g); CHKERRQ(ierr);
	ierr = VecSetSizes(g, PETSC_DECIDE, 4); CHKERRQ(ierr);
	*/
	ierr = AssembleOperator_Laplace(da_u, da_prop, properties, &A); CHKERRQ(ierr);
	ierr = AssembleRHS_Laplace(da_u, da_prop, properties, &f); CHKERRQ(ierr);
	/*
	ierr = AssembleOperator_Constraints(da_u, da_prop, &B); CHKERRQ(ierr);
	ierr = AssembleRHS_Constraints(da_u, da_prop, &g); CHKERRQ(ierr);
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
