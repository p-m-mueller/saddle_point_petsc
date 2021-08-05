static const char help[] = "Saddle point example.\n\n";

#include <petsc.h>

typedef struct
{
	PetscScalar uu, tt;
}UserContext;

extern PetscErrorCode ComputeJacobian(KSP, Mat, Mat, void*);
extern PetscErrorCode ComputeRHS(KSP, Vec, void*);

int main(int argc, char **argv)
{

	KSP		ksp;
	DM		da;
	Vec		x, b;
	UserContext 	user;
	PetscErrorCode 	ierr;

	ierr = PetscInitialize(&argc, &argv, (char*)0, help); if (ierr) return ierr;	
	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 11, 11, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da); CHKERRQ(ierr);
	ierr = DMSetFromOptions(da); CHKERRQ(ierr);	
	ierr = DMSetUp(da); CHKERRQ(ierr);
	ierr = KSPSetDM(ksp, (DM)da); CHKERRQ(ierr);
	ierr = DMSetApplicationContext(da, &user); CHKERRQ(ierr);
	
	user.uu = 1.0;
	user.tt = 1.0;

	ierr = KSPSetComputeRHS(ksp, ComputeRHS, &user); CHKERRQ(ierr);
	ierr = KSPSetComputeOperators(ksp, ComputeJacobian, &user); CHKERRQ(ierr);
	ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);


	ierr = KSPSolve(ksp, NULL, NULL); CHKERRQ(ierr);

	{
		PetscViewer 		viewer = NULL;
		PetscViewerFormat 	format;
		PetscBool		flg;

		ierr = KSPGetSolution(ksp, &x); CHKERRQ(ierr);
		ierr = PetscOptionsGetViewer(PETSC_COMM_WORLD, NULL, NULL, "-sol_view", &viewer, &format, &flg); CHKERRQ(ierr);
		if (flg)
		{
			ierr = PetscViewerPushFormat(viewer, format); CHKERRQ(ierr);
			ierr = VecView(x, viewer); CHKERRQ(ierr);
			ierr = PetscViewerPopFormat(viewer); CHKERRQ(ierr);
		}
		ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
	}

	{
		PetscViewer 		viewer = NULL;
		PetscViewerFormat 	format;
		PetscBool		flg;

		ierr = KSPGetRhs(ksp, &b); CHKERRQ(ierr);
		ierr = PetscOptionsGetViewer(PETSC_COMM_WORLD, NULL, NULL, "-rhs_view", &viewer, &format, &flg); CHKERRQ(ierr);
		if (flg)
		{
			ierr = PetscViewerPushFormat(viewer, format); CHKERRQ(ierr);
			ierr = VecView(b, viewer); CHKERRQ(ierr);
			ierr = PetscViewerPopFormat(viewer); CHKERRQ(ierr);
		}
		ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
	}

	ierr = DMDestroy(&da); CHKERRQ(ierr);
	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

	ierr = PetscFinalize(); CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode ComputeJacobian(KSP ksp, Mat J, Mat jac, void *ctx)
{
	PetscInt 	i, j, M, N, xm, ym, xs, ys, num, numi, numj;
	PetscScalar	v[5], Hx, Hy, HydHx, HxdHy;
	MatStencil	row, col[5];
	DM		da;
	MatNullSpace	nullspace;
	PetscErrorCode  ierr;


	ierr = KSPGetDM(ksp, &da); CHKERRQ(ierr);
	ierr = DMDAGetInfo(da, 0, &M, &N,0,0,0,0,0,0,0,0,0,0); CHKERRQ(ierr);
	Hx = 1.0 / (PetscReal)M;
	Hy = 1.0 / (PetscReal)N;
	HxdHy = Hx/Hy;
	HydHx = Hy/Hx;
	
	ierr = DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0);
	for (j = ys; j < ys+ym; ++j)
		for (i = xs; i < xs+xm; ++i)
		{
			row.i = i; row.j = j;

			if (i == 0  || j == 0 || i == M-1 || j == N-1)
			{
				num = 0; numi = 0; numj = 0;
				if (j != 0) 
				{
					v[num] = -HxdHy;		col[num].i = i; 	col[num].j = j-1;
					num++; numj++;
				}
				if (i != 0)
				{
					v[num] = -HydHx;		col[num].i = i-1;	col[num].j = j;
					num++; numi++;
				}
				if (i != M-1)
				{
					v[num] = -HydHx;		col[num].i = i+1; 	col[num].j = j;
					num++; numi++;
				}
				if (j != N-1)
				{
					v[num] = -HxdHy;		col[num].i = i;		col[num].j = j+1;
					num++; numj++;
				}
				v[num] = (PetscReal)numj * HxdHy + (PetscReal)numi * HydHx; 	col[num].i = i;		col[num].j = j;
				num++;
			       	ierr = MatSetValuesStencil(jac, 1, &row, num, col, v, INSERT_VALUES); CHKERRQ(ierr);	
			} else {
				v[0] = -HxdHy;			col[0].i = i; 	col[0].j = j-1;
				v[1] = -HydHx;			col[1].i = i-1;	col[1].j = j;
				v[2] = 2.0 * (HxdHy + HydHx);	col[2].i = i; 	col[2].j = j;
				v[3] = -HydHx;			col[3].i = i+1;	col[3].j = j;
				v[4] = -HxdHy;			col[4].i = i;	col[4].j = j+1;
				ierr = MatSetValuesStencil(jac, 1, &row, 5, col, v, INSERT_VALUES); CHKERRQ(ierr);
			}
		}
	ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace); CHKERRQ(ierr);
	ierr = MatSetNullSpace(J, nullspace); CHKERRQ(ierr);
	ierr = MatNullSpaceDestroy(&nullspace); CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx)
{
	UserContext 	*user = (UserContext*)ctx;
	PetscInt	i, j, M, N, xm, ym, xs, ys;
	PetscScalar	Hx, Hy, pi, uu, tt;
	PetscScalar	**array;
	DM		da;
	MatNullSpace	nullspace;
	PetscInt	ierr; 

	ierr = KSPGetDM(ksp, &da); CHKERRQ(ierr);
	ierr = DMDAGetInfo(da, 0, &M, &N, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);
	uu = user->uu; tt = user->tt;
	pi = 4.0*atan(1.0);
	Hx = 1.0 / (PetscReal)M;
	Hy = 1.0 / (PetscReal)N;

	ierr = DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, b, &array); CHKERRQ(ierr);

	for (j = ys; j < ys+ym; ++j)
		for (i = xs; i < xs+xm; ++i)
			array[j][i] = - PetscCosScalar(uu * pi * ((PetscReal)i+0.5) * Hx) * PetscCosScalar(tt * pi * ((PetscReal)j+0.5) * Hy) * Hx * Hy;
	
	ierr = DMDAVecRestoreArray(da, b, &array); CHKERRQ(ierr);
	ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

	ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace); CHKERRQ(ierr);
	ierr = MatNullSpaceRemove(nullspace, b); CHKERRQ(ierr);
	ierr = MatNullSpaceDestroy(&nullspace); CHKERRQ(ierr);
	return ierr;
}
