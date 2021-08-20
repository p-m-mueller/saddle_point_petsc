#include "SaddlePointProblem.h"
#include "Visualization.h"

PetscErrorCode SolveSaddlePointProblem(const char *filename)
{
	Mesh		mesh;
	Vec		u;
	PetscErrorCode 	ierr;

	ierr = MeshCreate(PETSC_COMM_WORLD, filename, &mesh); CHKERRQ(ierr);
	
	ierr = DMGetGlobalVector(mesh.dm, &u); CHKERRQ(ierr);
	
	ierr = SolveConstraintLaplaceProblem(mesh, &u); CHKERRQ(ierr);

	ierr = VecViewFromOptions(u, NULL, "-solution_view"); CHKERRQ(ierr);
	
	ierr = DMRestoreGlobalVector(mesh.dm, &u); CHKERRQ(ierr);

	//ierr = WriteVTK(dm, u, "test.vtk"); CHKERRQ(ierr);
	//
	ierr = MeshDestroy(&mesh); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode SolveConstraintLaplaceProblem(Mesh mesh, Vec *u)
{
	KSP		ksp;
	Mat		A, B;
	Vec		f, g;
	PetscInt	nCols;
	PetscErrorCode 	ierr;

	ierr = DMCreateMatrix(mesh.dm, &A); CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(mesh.dm, &f); CHKERRQ(ierr); 

	/*
	ierr = MatGetSize(A, NULL, &nCols); CHKERRQ(ierr);

	ierr = MatCreate(PETSC_COMM_WORLD, &B); CHKERRQ(ierr);
	ierr = MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, 4, nCols); CHKERRQ(ierr);

	ierr = VecCreate(PETSC_COMM_WORLD, &g); CHKERRQ(ierr);
	ierr = VecSetSizes(g, PETSC_DECIDE, 4); CHKERRQ(ierr);
	*/
	ierr = AssembleOperator_Laplace(mesh, &A); CHKERRQ(ierr);
	//ierr = AssembleRHS_Laplace(da, &f); CHKERRQ(ierr);
	//ierr = ApplyBC_Laplace(da, &A, &f); CHKERRQ(ierr); 
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

	return ierr;
}
