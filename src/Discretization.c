#include "Discretization.h"

PetscErrorCode SetupDMs(PetscInt nx, PetscInt ny, DM *da_u, DM *da_prop, Vec *properties)
{
	const PetscInt 	dof_u = 2;
	PetscInt	nProc_x, nProc_y;
	PetscInt	*lx, *ly;
	PetscInt	prop_dof, prop_stencil_width;
	PetscInt	M, N;
	PetscReal	dx, dy;
	ElementProperties	**elementProperties;
	Vec		l_properties;
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
	ierr = DMCreateGlobalVector(*da_prop, properties); CHKERRQ(ierr);
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
			}
		}
	ierr = DMDAVecRestoreArray(prop_cda, prop_coords, &_prop_coords); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(u_cda, u_coords, &_u_coords); CHKERRQ(ierr);
	
	ierr = DMDAVecRestoreArray(*da_prop, l_properties, &elementProperties); CHKERRQ(ierr);
	ierr = DMLocalToGlobalBegin(*da_prop, l_properties, ADD_VALUES, *properties); CHKERRQ(ierr); 
	ierr = DMLocalToGlobalEnd(*da_prop, l_properties, ADD_VALUES, *properties); CHKERRQ(ierr); 
		
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

PetscErrorCode GetElementCoords(DMDACoor2d **_coords, PetscInt ei, PetscInt ej, PetscScalar *el_coords)
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

PetscErrorCode ConstructQ12D_Ni(PetscScalar _xi[DIM], PetscScalar Ni[NODES_PER_ELEMENT])
{
	PetscScalar xi = _xi[0];
	PetscScalar eta = _xi[1];

	Ni[0] = 0.25 * (1.0 - xi) * (1.0 - eta);	
	Ni[1] = 0.25 * (1.0 - xi) * (1.0 + eta);	
	Ni[2] = 0.25 * (1.0 + xi) * (1.0 + eta);	
	Ni[3] = 0.25 * (1.0 + xi) * (1.0 - eta);	

	return 0;
}

PetscErrorCode ConstructQ12D_GNi(PetscScalar _xi[DIM], PetscScalar GNi[][NODES_PER_ELEMENT])
{
	PetscScalar xi = _xi[0];
	PetscScalar eta = _xi[1];

	GNi[0][0] = -0.25 * (1.0 - eta);
	GNi[0][1] = -0.25 * (1.0 + eta); 
	GNi[0][2] =  0.25 * (1.0 + eta);
	GNi[0][3] =  0.25 * (1.0 - eta);

	GNi[1][0] = -0.25 * (1.0 - xi);
	GNi[1][1] =  0.25 * (1.0 - xi); 
	GNi[1][2] =  0.25 * (1.0 + xi);
	GNi[1][3] = -0.25 * (1.0 + xi);

	return 0;
}

PetscErrorCode ConstructQ12D_GNx(PetscScalar GNi[][NODES_PER_ELEMENT], PetscScalar *el_coords, PetscScalar GNx[][NODES_PER_ELEMENT], PetscScalar *detJ)
{
	PetscScalar 	Jac[DIM][DIM], J;
	PetscScalar 	invJ[DIM][DIM];
	PetscInt	i, c, d;
	
	for (c = 0; c < DIM; ++c)
		for (d = 0; d < DIM; ++d)
			Jac[c][d] = 0.0;
	
	for (c = 0; c < DIM; ++c)
		for (d = 0; d < DIM; ++d)
			for (i = 0; i < NODES_PER_ELEMENT; ++i)
				Jac[c][d] += GNi[c][i] * el_coords[i*DIM+d];

	J = Jac[0][0] * Jac[1][1] - Jac[0][1] * Jac[1][0];

	invJ[0][0] =   Jac[1][1] / J;
	invJ[0][1] = - Jac[0][1] / J;
	invJ[1][0] = - Jac[1][0] / J;
	invJ[1][1] =   Jac[0][0] / J;


	for (i = 0; i < NODES_PER_ELEMENT; ++i)
	{
		GNx[0][i] = invJ[0][0] * GNi[0][i] + invJ[0][1] * GNi[1][i];
		GNx[1][i] = invJ[1][0] * GNi[0][i] + invJ[1][1] * GNi[1][i];
	}
	
	if (detJ) *detJ = J;
	
	return 0;
}

PetscErrorCode AssembleOperator_Laplace(DM da_u, DM da_prop, Vec properties, Mat *A)
{
	DM		cda;
	Vec		coords;
	DMDACoor2d	**_coords;
	Vec		l_properties;
	ElementProperties	**elementProperties;
	PetscInt	si, sj, ni, nj;
	PetscInt	i, j, u_eqn;
	PetscScalar	Ae[NODES_PER_ELEMENT*U_DOF*NODES_PER_ELEMENT*U_DOF];
	PetscErrorCode 	ierr;

	ierr = DMGetCoordinateDM(da_u, &cda); CHKERRQ(ierr);
	ierr = DMGetCoordinatesLocal(da_u, &coords); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(cda, coords, &_coords); CHKERRQ(ierr);

	ierr = DMCreateLocalVector(da_prop, &l_properties); CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(da_prop, properties, INSERT_VALUES, l_properties); CHKERRQ(ierr); 
	ierr = DMGlobalToLocalEnd(da_prop, properties, INSERT_VALUES, l_properties); CHKERRQ(ierr); 
	ierr = DMDAVecGetArray(da_prop, l_properties, &elementProperties); CHKERRQ(ierr);

	ierr = DMDAGetElementsCorners(da_u, &si, &sj, NULL); CHKERRQ(ierr);
	ierr = DMDAGetElementsSizes(da_u, &ni, &nj, NULL); CHKERRQ(ierr);
	for (j = sj; j < sj+nj; ++j)
		for (i = si; i < si+ni; ++i)
		{
			PetscScalar 	el_coords[DIM*GAUSS_POINTS];
			MatStencil	u_eqn[U_DOF*GAUSS_POINTS];
			PetscScalar	coeff[NODES_PER_ELEMENT];
			PetscInt	p;

			ierr = GetElementCoords(_coords, i, j, el_coords); CHKERRQ(ierr);

			for (p = 0; p < NODES_PER_ELEMENT; ++p)
				coeff[p] = 1.0;

			PetscMemzero(Ae, sizeof(Ae));

			ierr = FormStressOperatorQ12D(el_coords, coeff, Ae); 

			ierr = DMDAGetElementEqnums(i, j, u_eqn); CHKERRQ(ierr);

			ierr = MatSetValuesStencil(*A, NODES_PER_ELEMENT*U_DOF, u_eqn, NODES_PER_ELEMENT*U_DOF, u_eqn, Ae, ADD_VALUES); CHKERRQ(ierr);
		}

	ierr = MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(cda, coords, &_coords); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da_prop, properties, &elementProperties); CHKERRQ(ierr);

	return ierr;	
}

PetscErrorCode AssembleRHS_Laplace(DM da_u, DM da_prop, Vec properties, Vec *f)
{
	DM		cda;
	Vec		coords;
	DMDACoor2d	**_coords;
	Vec		l_properties;
	ElementProperties	**elementProperties;
	Vec		l_f;
	PetscScalar	**_f;
	PetscInt	si, sj, ni, nj;
	PetscInt	i, j;
	PetscScalar	Fe[U_DOF*NODES_PER_ELEMENT];
	PetscErrorCode 	ierr;
	
	ierr = DMGetCoordinateDM(da_u, &cda); CHKERRQ(ierr);
	ierr = DMGetCoordinatesLocal(da_u, &coords); CHKERRQ(ierr);	
	ierr = DMDAVecGetArray(cda, coords, &_coords); CHKERRQ(ierr); 

	ierr = DMGetLocalVector(da_prop, &l_properties); CHKERRQ(ierr);	
	ierr = DMGlobalToLocalBegin(da_prop, properties, INSERT_VALUES, l_properties); CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(da_prop, properties, INSERT_VALUES, l_properties); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_prop, l_properties, &elementProperties); CHKERRQ(ierr);

	ierr = DMGetLocalVector(da_u, &l_f); CHKERRQ(ierr);
	ierr = VecZeroEntries(l_f); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_u, l_f, &_f); CHKERRQ(ierr);

	ierr = DMDAGetElementsCorners(da_u, &si, &sj, NULL); CHKERRQ(ierr); 
	ierr = DMDAGetElementsSizes(da_u, &ni, &nj, NULL); CHKERRQ(ierr);
	for (j = sj; j < sj+nj; ++j)
		for (i = si; i < si+ni; ++i)
		{
			PetscScalar 	el_coords[DIM*GAUSS_POINTS];
			MatStencil 	u_eqn[NODES_PER_ELEMENT*U_DOF];
			PetscInt	n, c;

			ierr = GetElementCoords(_coords, i, j, el_coords); CHKERRQ(ierr);	
			
			ierr = PetscMemzero(Fe, sizeof(Fe));

			ierr = FormLaplaceRHSQ12D(el_coords, FormRHS, Fe); CHKERRQ(ierr);
		
			ierr = DMDAGetElementEqnums(i, j, u_eqn); CHKERRQ(ierr);			
			for (n = 0; n < NODES_PER_ELEMENT; ++n)
				for (c = 0; c < U_DOF; ++c)
					_f[u_eqn[U_DOF*n+c].j][u_eqn[DIM*n+c].i] += Fe[U_DOF*n+c];
		}
	
	ierr = DMDAVecRestoreArray(da_u, l_f, &_f); CHKERRQ(ierr);
	ierr = DMLocalToGlobalBegin(da_u, l_f, ADD_VALUES, *f); CHKERRQ(ierr);
	ierr = DMLocalToGlobalEnd(da_u, l_f, ADD_VALUES, *f); CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(da_u, &l_f); CHKERRQ(ierr);
	
	ierr = DMDAVecRestoreArray(cda, coords, &_coords); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(da_prop, l_properties, &elementProperties); CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(da_prop, &l_properties); CHKERRQ(ierr);

	return ierr;	
}


PetscErrorCode AssembleOperator_Constraints(DM da_u, DM da_prop, Mat *B)
{
	
	PetscErrorCode 	ierr;
	ierr = 0;
	return ierr;	
}

PetscErrorCode AssembleRHS_Constraints(DM da_u, DM da_prop, Vec *g)
{
	PetscErrorCode 	ierr;
	ierr = 0;
	return ierr;	
}


PetscErrorCode FormStressOperatorQ12D(PetscScalar *el_coords, PetscScalar *coeff, PetscScalar *Ke)
{
	PetscInt	ngp;
	PetscScalar	gp_xi[GAUSS_POINTS][2];
	PetscScalar	gp_w[GAUSS_POINTS];
	PetscInt	i, j, k, p;
	PetscScalar	J_p, tildeD[3], B[3][NODES_PER_ELEMENT*U_DOF];
	PetscErrorCode 	ierr;
	
	ierr = ConstructGaussQuadrature(&ngp, gp_xi, gp_w); CHKERRQ(ierr);	

	for (p = 0; p < ngp; ++p)
	{
		PetscScalar	GNi_p[DIM][NODES_PER_ELEMENT], GNx_p[DIM][NODES_PER_ELEMENT], detJ_p;
		PetscInt	d;


		ierr = ConstructQ12D_GNi(gp_xi[p], GNi_p); CHKERRQ(ierr);
		ierr = ConstructQ12D_GNx(GNi_p, el_coords, GNx_p, &detJ_p); CHKERRQ(ierr);

		for (i = 0; i < NODES_PER_ELEMENT; ++i)
		{
			B[0][DIM*i+0] = GNx_p[0][i];	B[0][DIM*i+1] = 0.0;
			B[1][DIM*i+0] = 0.0;		B[1][DIM*i+1] = GNx_p[1][i];
			B[2][DIM*i+0] = GNx_p[1][i];	B[2][DIM*i+1] = GNx_p[0][i];
		}

		tildeD[0] = 2.0 * gp_w[p] * detJ_p * coeff[p];
		tildeD[1] = 2.0 * gp_w[p] * detJ_p * coeff[p];
		tildeD[2] =       gp_w[p] * detJ_p * coeff[p];

		for (i = 0; i < U_DOF*NODES_PER_ELEMENT; ++i)
			for (j = 0; j < U_DOF*NODES_PER_ELEMENT; ++j)
				for (k = 0; k < 3; ++k)
					Ke[i+U_DOF*NODES_PER_ELEMENT*j] += B[k][i] * tildeD[k] * B[k][j];

	}

	return ierr;
}

PetscErrorCode FormLaplaceRHSQ12D(PetscScalar *el_coords, PetscErrorCode (*f)(PetscScalar *x, PetscScalar *f), PetscScalar *Fe)
{
	PetscInt	ngp;
	PetscScalar	gp_xi[GAUSS_POINTS][2];
	PetscScalar	gp_w[GAUSS_POINTS];
	PetscInt	n, c, p;
	PetscErrorCode	ierr;

	ierr = ConstructGaussQuadrature(&ngp, gp_xi, gp_w); CHKERRQ(ierr);

	for (p = 0; p < ngp; ++p)
	{
		PetscScalar	Ni_p[NODES_PER_ELEMENT];
		PetscScalar	GNi_p[DIM][NODES_PER_ELEMENT], GNx_p[DIM][NODES_PER_ELEMENT];
		PetscScalar	detJ_p, fac;
		PetscScalar	f_p[U_DOF*NODES_PER_ELEMENT];
		PetscInt	c, i;

		ierr = ConstructQ12D_Ni(gp_xi[p], Ni_p); CHKERRQ(ierr);
		ierr = ConstructQ12D_GNi(gp_xi[p], GNi_p); CHKERRQ(ierr); 
		ierr = ConstructQ12D_GNx(GNi_p, el_coords, GNx_p, &detJ_p); CHKERRQ(ierr);

		for (i = 0; i < NODES_PER_ELEMENT; ++i)
		{
			PetscScalar	x[DIM], fe[U_DOF];
			PetscInt 	d;
			for (d = 0;  d < DIM; ++d)
				x[d] = el_coords[i*DIM+d];

			ierr = f(x, fe); CHKERRQ(ierr);
		
			for (c = 0; c < U_DOF; ++c)	
				f_p[i*U_DOF+c] = fe[c];	
		}

		fac = gp_w[p] * detJ_p;

		for (i = 0; i < NODES_PER_ELEMENT; ++i)
			for (c = 0; c < U_DOF; ++c)
				Fe[U_DOF*i+c] += fac * Ni_p[i] * f_p[i*U_DOF+c];
	}
	
	return ierr;
}


static PetscErrorCode DMDAGetElementEqnums(PetscInt i, PetscInt j, MatStencil u_eqn[NODES_PER_ELEMENT*U_DOF])
{
	// Node 0
	u_eqn[0].i = i;		u_eqn[0].j = j; 	u_eqn[0].c = 0;	
	u_eqn[1].i = i;		u_eqn[1].j = j; 	u_eqn[1].c = 1;	

	// Node 1
	u_eqn[2].i = i;		u_eqn[2].j = j+1; 	u_eqn[2].c = 0;	
	u_eqn[3].i = i;		u_eqn[3].j = j+1; 	u_eqn[3].c = 1;	

	// Node 2	
	u_eqn[4].i = i+1; 	u_eqn[4].j = j+1; 	u_eqn[4].c = 0;	
	u_eqn[5].i = i+1;	u_eqn[5].j = j+1; 	u_eqn[5].c = 1;	

	// Node 3
	u_eqn[6].i = i+1; 	u_eqn[6].j = j; 	u_eqn[6].c = 0;	
	u_eqn[7].i = i+1;	u_eqn[7].j = j; 	u_eqn[7].c = 1;	
	return 0;	
}

PetscErrorCode FormRHS(PetscScalar *x, PetscScalar *f_p)
{
	f_p[0] = sin(x[0])*cos(x[1]);
	f_p[1] = 2.0;	
	return 0;
}
