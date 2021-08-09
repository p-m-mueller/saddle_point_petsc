static const char help[] = "Solves a example saddlepoint problem with the 2D Poisson problem with barycentre and volume constraints \non a Nx times Ny structured mesh using Q1Q1 finite elemnts.\nSee also the PETSc tutorial ksp/ex42.c.\n\n";

#include <petsc.h>
#include "SaddlePointProblem.h"

extern PetscErrorCode SolveSaddlePointProblem(PetscInt, PetscInt);

int main(int argc, char **argv)
{
	PetscInt 	Nx, Ny;
	PetscErrorCode 	ierr;

	ierr = PetscInitialize(&argc, &argv, (char*)0, help); if (ierr) return ierr;	

	Nx = 5; Ny = 3;
	ierr = SolveSaddlePointProblem(Nx, Ny); CHKERRQ(ierr);

	ierr = PetscFinalize(); CHKERRQ(ierr);
	return ierr;
}

