#include "Discretization.h"

PetscErrorCode CreateMesh(MPI_Comm comm, const char *filename, DM *da)
{
	const PetscInt 	dof_u = 2;
	size_t 		len;
	PetscErrorCode	ierr;

	ierr = PetscStrlen(filename, &len); CHKERRQ(ierr);
	
	if (len)
	{
		ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, da); CHKERRQ(ierr);
	} else {
		ierr = DMPlexCreateBoxMesh(comm, 2, PETSC_TRUE, NULL, NULL, NULL, NULL, PETSC_TRUE, da); CHKERRQ(ierr);
	}
	ierr = DMSetFromOptions(*da); CHKERRQ(ierr);
	
	return ierr;
}

PetscErrorCode SetupFields(DM *da)
{
	PetscSection 	sect;
	PetscInt 	pStart, pEnd;
	PetscInt	v, vStart, vEnd;
	PetscErrorCode	ierr;

	ierr = DMPlexGetChart(*da, &pStart, &pEnd); CHKERRQ(ierr);
	ierr = DMPlexGetHeightStratum(*da, 2, &vStart, &vEnd); CHKERRQ(ierr);
	ierr = PetscSectionSetChart(sect, pStart, pEnd); CHKERRQ(ierr);
	for (v = vStart; v < vEnd; ++v)
		PetscSectionSetDof(sect, v, 1);

	return ierr;
}


PetscErrorCode GetElementCoords(DMDACoor2d **_coords, PetscInt ei, PetscInt ej, PetscScalar *el_coords)
{
	PetscInt	k;
	for (k = 0; k < NODES_PER_ELEMENT; ++k)
	{
			el_coords[k*DIM+0] = _coords[ej][ei].x;
			el_coords[k*DIM+1] = _coords[ej][ei].y;
	}
	/*
	el_coords[DIM*0+0] = _coords[ej][ei].x;			el_coords[DIM*0+1] = _coords[ej][ei].y;
	el_coords[DIM*1+0] = _coords[ej+1][ei].x;		el_coords[DIM*1+1] = _coords[ej+1][ei].y;
	el_coords[DIM*2+0] = _coords[ej+1][ei+1].x;		el_coords[DIM*2+1] = _coords[ej+1][ei+1].y;
	el_coords[DIM*3+0] = _coords[ej][ei+1].x;		el_coords[DIM*3+1] = _coords[ej][ei+1].y;
	*/
	return 0;
}


PetscErrorCode ConstructGaussQuadratureQ12D(PetscInt *ngp, PetscScalar gp_xi[][DIM], PetscScalar gp_weights[])
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

PetscErrorCode AssembleOperator_Laplace(DM da_u, Mat *A)
{
	DM		cda;
	Vec		coords;
	DMDACoor2d	**_coords;
	PetscInt	si, sj, ni, nj;
	PetscInt	i, j, u_eqn;
	PetscScalar	Ae[NODES_PER_ELEMENT*U_DOF*NODES_PER_ELEMENT*U_DOF];
	PetscErrorCode 	ierr;

	ierr = DMGetCoordinateDM(da_u, &cda); CHKERRQ(ierr);
	ierr = DMGetCoordinatesLocal(da_u, &coords); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(cda, coords, &_coords); CHKERRQ(ierr);

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

	return ierr;	
}

PetscErrorCode AssembleRHS_Laplace(DM da_u, Vec *f)
{
	DM		cda;
	Vec		coords;
	DMDACoor2d	**_coords;
	Vec		l_f;
	Field		**_f;
	PetscInt	si, sj, ni, nj;
	PetscInt	i, j;
	PetscScalar	Fe[U_DOF*NODES_PER_ELEMENT];
	PetscErrorCode 	ierr;
	
	ierr = DMGetCoordinateDM(da_u, &cda); CHKERRQ(ierr);
	ierr = DMGetCoordinatesLocal(da_u, &coords); CHKERRQ(ierr);	
	ierr = DMDAVecGetArray(cda, coords, &_coords); CHKERRQ(ierr); 

	ierr = DMGetLocalVector(da_u, &l_f); CHKERRQ(ierr);
	ierr = VecZeroEntries(l_f); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da_u, l_f, &_f); CHKERRQ(ierr);

	ierr = DMDAGetElementsCorners(da_u, &si, &sj, NULL); CHKERRQ(ierr); 
	ierr = DMDAGetElementsSizes(da_u, &ni, &nj, NULL); CHKERRQ(ierr);
	for (j = sj; j < sj+nj; ++j)
		for (i = si; i < si+ni; ++i)
		{
			PetscScalar 	el_coords[DIM*GAUSS_POINTS];
			MatStencil 	idxm[NODES_PER_ELEMENT*U_DOF];
			PetscInt	n;

			ierr = GetElementCoords(_coords, i, j, el_coords); CHKERRQ(ierr);	
			
			ierr = PetscMemzero(Fe, sizeof(Fe));

			ierr = FormLaplaceRHSQ12D(el_coords, FormRHS, Fe); CHKERRQ(ierr);
		
			ierr = DMDAGetElementEqnums(i, j, idxm); CHKERRQ(ierr);			
			for (n = 0; n < NODES_PER_ELEMENT; ++n)
			{
				_f[idxm[U_DOF*n+0].j][idxm[DIM*n+0].i].Ux += Fe[U_DOF*n+0];
				_f[idxm[U_DOF*n+1].j][idxm[DIM*n+1].i].Uy += Fe[U_DOF*n+1];
			}
		}
	
	ierr = DMDAVecRestoreArray(da_u, l_f, &_f); CHKERRQ(ierr);
	ierr = DMLocalToGlobalBegin(da_u, l_f, ADD_VALUES, *f); CHKERRQ(ierr);
	ierr = DMLocalToGlobalEnd(da_u, l_f, ADD_VALUES, *f); CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(da_u, &l_f); CHKERRQ(ierr);
	

	
	ierr = DMDAVecRestoreArray(cda, coords, &_coords); CHKERRQ(ierr);

	return ierr;	
}

PetscErrorCode ApplyBC_Laplace(DM da_u, Mat *A, Vec *f)
{
	DMDALocalInfo		info;
	PetscInt		nbnd;
	PetscInt		i, j, d;
	ISLocalToGlobalMapping	ltogm;
	const PetscInt		*gIdx;
	PetscInt		ngIdx;
	PetscInt		nbcIds, ibcId;
	PetscInt		*bcIds_g;
	PetscScalar		*u_bc;
	PetscErrorCode 		ierr;

	ierr = DMDAGetLocalInfo(da_u, &info); CHKERRQ(ierr);
	nbcIds = 0;
	for (j = info.ys; j < info.ys+info.ym; ++j)
		for (i = info.xs; i < info.xs+info.xm; ++i)
			if (i == 0 || i == info.mx-1 || j == 0 || j == info.my-1)
				nbcIds += info.dof;

	ierr = PetscMalloc1(nbcIds, &bcIds_g); CHKERRQ(ierr);
	ierr = PetscMalloc1(nbcIds, &u_bc); CHKERRQ(ierr);

	ibcId = 0;
	for (j = info.ys; j < info.ys+info.ym; ++j)
		for (i = info.xs; i < info.xs+info.xm; ++i)
			if (i == 0 || i == info.mx-1 || j == 0 || j == info.my-1)
				for (d = 0; d < info.dof; ++d)
				{
					bcIds_g[ibcId] = (j*info.xm+i)*info.dof+d;
					u_bc[ibcId] = 0.0;
					ibcId++;
				}


	ierr = VecSetValues(*f, nbcIds, bcIds_g, u_bc, INSERT_VALUES); CHKERRQ(ierr);
	ierr = VecAssemblyBegin(*f); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(*f); CHKERRQ(ierr);

	ierr = MatZeroRowsColumns(*A, nbcIds, bcIds_g, 1.0, NULL, NULL); CHKERRQ(ierr);
		
	ierr = PetscFree(bcIds_g); CHKERRQ(ierr);
	ierr = PetscFree(u_bc); CHKERRQ(ierr);

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
	
	ierr = ConstructGaussQuadratureQ12D(&ngp, gp_xi, gp_w); CHKERRQ(ierr);	

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

	ierr = ConstructGaussQuadratureQ12D(&ngp, gp_xi, gp_w); CHKERRQ(ierr);

	for (p = 0; p < ngp; ++p)
	{
		PetscScalar	Ni_p[NODES_PER_ELEMENT];
		PetscScalar	GNi_p[DIM][NODES_PER_ELEMENT], GNx_p[DIM][NODES_PER_ELEMENT];
		PetscScalar	detJ_p, fac;
		PetscInt	i;

		ierr = ConstructQ12D_Ni(gp_xi[p], Ni_p); CHKERRQ(ierr);
		ierr = ConstructQ12D_GNi(gp_xi[p], GNi_p); CHKERRQ(ierr); 
		ierr = ConstructQ12D_GNx(GNi_p, el_coords, GNx_p, &detJ_p); CHKERRQ(ierr);
		
		fac = gp_w[p] * detJ_p;

		for (i = 0; i < NODES_PER_ELEMENT; ++i)
		{
			PetscScalar	x_p[DIM], f_p[U_DOF];
			PetscInt 	c, d;

			for (d = 0;  d < DIM; ++d)
				x_p[d] = gp_xi[p][d];

			ierr = f(x_p, f_p); CHKERRQ(ierr);
			

			for (c = 0; c < U_DOF; ++c)	
				Fe[i*U_DOF+c] += fac * Ni_p[i] * f_p[c];
		}
	}
	
	return ierr;
}


PetscErrorCode DMDAGetElementEqnums(PetscInt i, PetscInt j, MatStencil u_eqn[NODES_PER_ELEMENT*U_DOF])
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
	f_p[0] = 1.0; //sin(x[0])*cos(x[1]);
	f_p[1] = 2.0;	
	return 0;
}
