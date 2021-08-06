#include "Meshfunctions.h"

PetscErrorCode GetElementOwnershipRanges2d(DM da, PetscInt **lx, PetscInt **ly)
{
	PetscMPIInt	rank;
	PetscInt	nProc_x, nProc_y;
	Index2d		procIdx;
	PetscErrorCode 	ierr;

	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

	ierr = DMDAGetInfo(da, 0, 0, 0, 0, &nProc_x, &nProc_y, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);

	procIdx.i = rank / nProc_x;
	procIdx.j = rank / nProc_y;

	return ierr;
}
