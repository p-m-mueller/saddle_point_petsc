#include "Discretization.h"

PetscErrorCode SetupDMs(PetscInt nx, PetscInt ny, DM *da_u, DM *da_prop)
{
	const PetscInt 	dof_u = 2;
	PetscInt	nProc_x, nProc_y;
	PetscInt	*lx, *ly;
	PetscInt	prop_dof, prop_stencil_width;
	PetscInt	M, N;
	PetscReal	dx, dy;
	PetscInt	i, j, k;
	PetscErrorCode	ierr;

	/*
	 * Create the da for u
	 */
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, nx+1, ny+1, PETSC_DECIDE, PETSC_DECIDE, dof_u, 1, NULL, NULL, da_u); CHKERRQ(ierr);
	ierr = DMSetMatType(*da_u, MATAIJ); CHKERRQ(ierr);
	ierr = DMSetFromOptions(*da_u); CHKERRQ(ierr);
	ierr = DMSetUp(*da_u); CHKERRQ(ierr);

	ierr = DMDASetFieldName(*da_u, 0, "Ux"); CHKERRQ(ierr);
	ierr = DMDASetFieldName(*da_u, 1, "Uy"); CHKERRQ(ierr);
	ierr = DMDASetUniformCoordinates(*da_u, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0); CHKERRQ(ierr);

	/*
	 * Generate element properties
	 */
	ierr = DMDAGetInfo(*da_u, 0, 0, 0, 0, &nProc_x, &nProc_y, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);
	ierr = GetElementOwnershipRanges2d(*da_u, &lx, &ly); CHKERRQ(ierr);

	prop_dof = (PetscInt)(sizeof(GaussPointCoefficients) / sizeof(PetscScalar));
	prop_stencil_width = 0;
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, nx, ny, nProc_x, nProc_y, prop_dof, prop_stencil_width, lx, ly, da_prop); CHKERRQ(ierr);
	ierr = DMSetFromOptions(*da_prop); CHKERRQ(ierr); 
	ierr = DMSetUp(*da_prop); CHKERRQ(ierr); 
	ierr = PetscFree(lx); CHKERRQ(ierr);
	ierr = PetscFree(ly); CHKERRQ(ierr);

	ierr = DMDAGetInfo(*da_prop, 0, &M, &N, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);
	dx = 1.0 / (PetscReal)M;	
	dy = 1.0 / (PetscReal)N;
	ierr = DMDASetUniformCoordinates(*da_prop, 0.0 + 0.5 * dx, 1.0 - 0.5 * dx, 0.0 + 0.5 * dy, 1.0 - 0.5 * dy, 0.0, 0.0); CHKERRQ(ierr); 

	
	return ierr;
}



PetscErrorCode GetElementOwnershipRanges2d(DM da, PetscInt **lx, PetscInt **ly)
{
	PetscMPIInt	rank;
	PetscInt	nProc_x, nProc_y;
	Index2d		procIdx;
	PetscInt	*Lx, *Ly;
	PetscInt	local_mx, local_my;
	Vec 		vlx, vly, V_SEQ;
	VecScatter	ctx;
	PetscScalar	*_a;
	PetscInt 	i;
	PetscErrorCode 	ierr;

	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

	ierr = DMDAGetInfo(da, 0, 0, 0, 0, &nProc_x, &nProc_y, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);

	procIdx.j = rank / nProc_x;
	procIdx.i = rank - nProc_x * procIdx.j;

	ierr = PetscMalloc1(nProc_x, &Lx); CHKERRQ(ierr);
	ierr = PetscMalloc1(nProc_y, &Ly); CHKERRQ(ierr);

	ierr = DMDAGetElementsSizes(da, &local_mx, &local_my, 0); CHKERRQ(ierr);	

	ierr = VecCreate(PETSC_COMM_WORLD, &vlx); CHKERRQ(ierr);
	ierr = VecSetSizes(vlx, PETSC_DECIDE, nProc_x); CHKERRQ(ierr);
	ierr = VecSetFromOptions(vlx); CHKERRQ(ierr);

	ierr = VecCreate(PETSC_COMM_WORLD, &vly); CHKERRQ(ierr);
	ierr = VecSetSizes(vly, PETSC_DECIDE, nProc_y); CHKERRQ(ierr);
	ierr = VecSetFromOptions(vly); CHKERRQ(ierr);

	ierr = VecSetValue(vlx, procIdx.i, (PetscScalar)(local_mx + 1 * 1.0e-9), INSERT_VALUES); CHKERRQ(ierr);
	ierr = VecSetValue(vly, procIdx.j, (PetscScalar)(local_my + 1 * 1.0e-9), INSERT_VALUES); CHKERRQ(ierr);

	ierr = VecAssemblyBegin(vlx); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(vlx); CHKERRQ(ierr);
	ierr = VecAssemblyBegin(vly); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(vly); CHKERRQ(ierr);

	ierr = VecScatterCreateToAll(vlx, &ctx, &V_SEQ); CHKERRQ(ierr);
	ierr = VecScatterBegin(ctx, vlx, V_SEQ, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	ierr = VecScatterEnd(ctx, vlx, V_SEQ, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	ierr = VecGetArray(V_SEQ, &_a); CHKERRQ(ierr);
	for (i = 0; i < nProc_x; ++i)
		Lx[i] = (PetscInt)PetscRealPart(_a[i]);
	ierr = VecRestoreArray(V_SEQ, &_a); CHKERRQ(ierr);
	ierr = VecScatterDestroy(&ctx); CHKERRQ(ierr);
	ierr = VecDestroy(&V_SEQ); CHKERRQ(ierr);

	ierr = VecScatterCreateToAll(vly, &ctx, &V_SEQ); CHKERRQ(ierr);
	ierr = VecScatterBegin(ctx, vly, V_SEQ, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	ierr = VecScatterEnd(ctx, vly, V_SEQ, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	ierr = VecGetArray(V_SEQ, &_a); CHKERRQ(ierr);
	for (i = 0; i < nProc_x; ++i)
		Ly[i] = (PetscInt)PetscRealPart(_a[i]);
	ierr = VecRestoreArray(V_SEQ, &_a); CHKERRQ(ierr);
	ierr = VecScatterDestroy(&ctx); CHKERRQ(ierr);
	ierr = VecDestroy(&V_SEQ); CHKERRQ(ierr);

	*lx = Lx;
	*ly = Ly;

	ierr = VecDestroy(&vlx); CHKERRQ(ierr);
	ierr = VecDestroy(&vly); CHKERRQ(ierr);

	return ierr;
}
