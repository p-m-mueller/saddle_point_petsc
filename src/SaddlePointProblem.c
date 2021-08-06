#include "SaddlePointProblem.h"

PetscErrorCode solveSaddlePointProblem(PetscInt nx, PetscInt ny)
{
	DM		da_u;
	KSP		ksp;
	AppCtx		ctx;
	const PetscInt 	dof_u = 2;
	PetscInt	nProc_x, nProc_y;
	PetscInt	*lx, *ly;
	PetscInt	i, j, k;
	PetscErrorCode	ierr;

	ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, nx+1, ny+1, PETSC_DECIDE, PETSC_DECIDE, dof_u, 1, NULL, NULL, &da_u); CHKERRQ(ierr);
	ierr = DMSetMatType(da_u, MATAIJ); CHKERRQ(ierr);
	ierr = DMSetFromOptions(da_u); CHKERRQ(ierr);
	ierr = DMSetUp(da_u); CHKERRQ(ierr);

	ierr = DMDASetFieldName(da_u, 0, "Ux"); CHKERRQ(ierr);
	ierr = DMDASetFieldName(da_u, 1, "Uy"); CHKERRQ(ierr);
	ierr = DMDASetUniformCoordinates(da_u, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0); CHKERRQ(ierr);

	ierr = DMDAGetInfo(da_u, 0, 0, 0, 0, &nProc_x, &nProc_y, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);
	ierr = GetElementOwnershipRanges2d(da_u, &lx, &ly); CHKERRQ(ierr);
		

	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);

	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
	ierr =  DMDestroy(&da_u); CHKERRQ(ierr);
	return ierr;
}
