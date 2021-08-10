#include "Discretization.h"

PetscErrorCode SetupDMs(PetscInt nx, PetscInt ny, DM *da_u, DM *da_prop)
{
	const PetscInt 	dof_u = 2;
	PetscInt	nProc_x, nProc_y;
	PetscInt	*lx, *ly;
	PetscInt	prop_dof, prop_stencil_width;
	PetscInt	M, N;
	PetscReal	dx, dy;
	ElementProperties	**elementProperties;
	Vec		properties, l_properties;
	DM		prop_cda, u_cda;
	Vec		prop_coords, u_coords;
	DMDACoor2d	**_prop_coords, **_u_coords;
	PetscInt	si, sj, ni, nj;
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

	prop_dof = (PetscInt)(sizeof(ElementProperties) / sizeof(PetscScalar));
	prop_stencil_width = 0;
	ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, nx, ny, nProc_x, nProc_y, prop_dof, prop_stencil_width, lx, ly, da_prop); CHKERRQ(ierr);
	ierr = DMSetFromOptions(*da_prop); CHKERRQ(ierr); 
	ierr = DMSetUp(*da_prop); CHKERRQ(ierr); 
	ierr = PetscFree(lx); CHKERRQ(ierr);
	ierr = PetscFree(ly); CHKERRQ(ierr);

	// Setup centroid coordiantes
	ierr = DMDAGetInfo(*da_prop, 0, &M, &N, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);
	dx = 1.0 / (PetscReal)M;	
	dy = 1.0 / (PetscReal)N;
	ierr = DMDASetUniformCoordinates(*da_prop, 0.0 + 0.5 * dx, 1.0 - 0.5 * dx, 0.0 + 0.5 * dy, 1.0 - 0.5 * dy, 0.0, 0.0); CHKERRQ(ierr); 

	// Setup element properties 
	ierr = DMCreateGlobalVector(*da_prop, &properties); CHKERRQ(ierr);
	ierr = DMCreateLocalVector(*da_prop, &l_properties); CHKERRQ(ierr); 
	ierr = DMDAVecGetArray(*da_prop, l_properties, &elementProperties); CHKERRQ(ierr);

	ierr = DMGetCoordinateDM(*da_prop, &prop_cda); CHKERRQ(ierr); 
	ierr = DMGetCoordinatesLocal(*da_prop, &prop_coords); CHKERRQ(ierr); 
	ierr = DMDAVecGetArray(prop_cda, prop_coords, &_prop_coords); CHKERRQ(ierr);

	ierr = DMGetCoordinateDM(*da_u, &u_cda); CHKERRQ(ierr);
	ierr = DMGetCoordinatesLocal(*da_u, &u_coords); CHKERRQ(ierr); 
	ierr = DMDAVecGetArray(u_cda, u_coords, &_u_coords); CHKERRQ(ierr);

	ierr = DMDAGetGhostCorners(prop_cda, &si, &sj, 0, &ni, &nj, 0); CHKERRQ(ierr);
	for (j = sj; j < sj+nj; ++j)
		for (i = si; i < si+ni; ++i)
		{
			PetscInt 	ngp;
			PetscScalar	gp_xi[GAUSS_POINTS][DIM], gp_weights[GAUSS_POINTS];
			PetscScalar	el_coords[DIM*NODES_PER_ELEMENT];
			PetscInt	p;

			ierr = GetElementCoords(_u_coords, i, j, el_coords);  CHKERRQ(ierr);
			ierr = ConstructGaussQuadrature(&ngp, gp_xi, gp_weights); CHKERRQ(ierr); 

			for (p = 0; p < GAUSS_POINTS; ++p)
			{
				PetscScalar	gp[DIM], xi_p[DIM], Ni_p[4];
				PetscInt	d, n;
				
				// Gauss point coordinates
				for (d = 0; d < DIM; ++d)
				{
					xi_p[d] = gp_xi[p][d];
					gp[d] = 0.0;
				}

				ierr = ConstructQ12D_Ni(xi_p, Ni_p); CHKERRQ(ierr);

				for (n = 0; n < NODES_PER_ELEMENT; ++n)
					for (d = 0; d < DIM; ++d)
						gp[d] += Ni_p[n] * el_coords[DIM*n+d];

				for (d = 0; d < DIM; ++d)
				{
					elementProperties[j][i].gp_coords[DIM*p+d] = gp[d];
					elementProperties[j][i].gp_weights[DIM*p+d] = gp_weights[d];
				}

				// Forcing term 
				{
					PetscScalar	f[DIM], x[DIM];
					for (d = 0; d < DIM; ++d)
						x[d] = elementProperties[j][i].gp_coords[DIM*p+d];

					ierr = SetElementForcingTerm(x, f); CHKERRQ(ierr);

					for (d = 0; d < DIM; ++d)
						elementProperties[j][i].f[DIM*p+d] = f[d];
				}
					
			}
		}
	ierr = DMDAVecRestoreArray(prop_cda, prop_coords, &_prop_coords); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(u_cda, u_coords, &_u_coords); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(*da_prop, l_properties, &elementProperties); CHKERRQ(ierr);
	ierr = DMLocalToGlobalBegin(*da_prop, l_properties, ADD_VALUES, properties); CHKERRQ(ierr); 
	ierr = DMLocalToGlobalEnd(*da_prop, l_properties, ADD_VALUES, properties); CHKERRQ(ierr); 

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

PetscErrorCode GetElementCoords(DMDACoor2d **_coords, PetscInt ei, PetscInt ej, PetscScalar el_coords[])
{
	PetscErrorCode 	ierr;

	if (DIM*3+1 < 8) 
	{
		ierr = 0;
	} else {
		return 1;
	}

	el_coords[DIM*0+0] = _coords[ej][ei].x;			el_coords[DIM*0+1] = _coords[ej][ei].y;
	el_coords[DIM*1+0] = _coords[ej+1][ei].x;		el_coords[DIM*1+1] = _coords[ej+1][ei].y;
	el_coords[DIM*2+0] = _coords[ej+1][ei+1].x;		el_coords[DIM*2+1] = _coords[ej+1][ei+1].y;
	el_coords[DIM*3+0] = _coords[ej][ei+1].x;		el_coords[DIM*3+1] = _coords[ej][ei+1].y;

	return ierr;
}

PetscErrorCode ConstructGaussQuadrature(PetscInt *ngp, PetscScalar gp_xi[][2], PetscScalar gp_weights[])
{
	* ngp = 4;
	gp_xi[0][0] = -0.57735026919;	gp_xi[0][1] = -0.57735026919;
	gp_xi[1][0] = -0.57735026919;	gp_xi[1][1] =  0.57735026919;
	gp_xi[2][0] =  0.57735026919;	gp_xi[2][1] =  0.57735026919;
	gp_xi[3][0] =  0.57735026919;	gp_xi[3][1] = -0.57735026919;

	gp_weights[0] = 1.0;
	gp_weights[1] = 1.0;
	gp_weights[2] = 1.0;
	gp_weights[3] = 1.0;

	return 0;
}

PetscErrorCode ConstructQ12D_Ni(PetscScalar _xi[2], PetscScalar Ni[4])
{
	PetscScalar xi = _xi[0];
	PetscScalar eta = _xi[1];

	Ni[0] = 0.25 * (1.0 - xi) * (1.0 - eta);	
	Ni[1] = 0.25 * (1.0 - xi) * (1.0 + eta);	
	Ni[2] = 0.25 * (1.0 + xi) * (1.0 + eta);	
	Ni[3] = 0.25 * (1.0 + xi) * (1.0 - eta);	

	return 0;
}

PetscErrorCode SetElementForcingTerm(PetscScalar x[DIM], PetscScalar f[DIM])
{
	f[0] = sin(x[0])*cos(x[1]);
	f[1] = f[0]*sin(x[1]);

	return 0;
}

PetscErrorCode AssembleOperator_Laplace(DM da_u, DM da_prop, Mat *A)
{
	
	PetscErrorCode 	ierr;

	return ierr;	
}

PetscErrorCode AssembleRHS_Laplace(DM da_u, DM da_prop, Vec *f)
{
	PetscErrorCode 	ierr;

	return ierr;	
}


PetscErrorCode AssembleOperator_Constraints(DM da_u, DM da_prop, Mat *B)
{
	
	PetscErrorCode 	ierr;

	return ierr;	
}

PetscErrorCode AssembleRHS_Constraints(DM da_u, DM da_prop, Vec *g)
{
	PetscErrorCode 	ierr;

	return ierr;	
}


