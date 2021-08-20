#include "Discretization.h"

PetscErrorCode MeshCreate(MPI_Comm comm, const char *filename, Mesh *mesh)
{
	const PetscInt 	dof_u = 2;
	size_t 		len;
	PetscErrorCode	ierr;

	ierr = PetscStrlen(filename, &len); CHKERRQ(ierr);
	if (len)
	{
		ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, &mesh->dm); CHKERRQ(ierr);
	} else {
		ierr = DMPlexCreateBoxMesh(comm, 2, PETSC_TRUE, NULL, NULL, NULL, NULL, PETSC_TRUE, &mesh->dm); CHKERRQ(ierr);
	}
	ierr = PetscObjectSetName((PetscObject)(mesh->dm), "Mesh"); CHKERRQ(ierr);

	// Distrebute mesh over processes
	{
		DM 			dmDist = NULL;
		PetscPartitioner	part;

		ierr = DMPlexGetPartitioner(mesh->dm, &part);  CHKERRQ(ierr);
		ierr = PetscPartitionerSetFromOptions(part); CHKERRQ(ierr);
		ierr = DMPlexDistribute(mesh->dm, 0, NULL, &dmDist); CHKERRQ(ierr);
		if (dmDist) 
		{
			ierr = DMDestroy(&mesh->dm); CHKERRQ(ierr);
			mesh->dm = dmDist;
		}
	}

	ierr = DMSetFromOptions(mesh->dm); CHKERRQ(ierr);
	ierr = DMViewFromOptions(mesh->dm, NULL, "-dm_view"); CHKERRQ(ierr);
	
	ierr = MeshSetupSection(mesh); CHKERRQ(ierr); 

	ierr = MeshSetupGeometry(mesh); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode MeshSetupSection(Mesh *mesh)
{
	PetscSection 	sect;
	PetscInt	v, vStart, vEnd;
	PetscInt	dim;
	PetscErrorCode	ierr;

	ierr = DMPlexGetDepthStratum(mesh->dm, 0, &vStart, &vEnd); CHKERRQ(ierr); // vertices

	ierr = PetscSectionCreate(PetscObjectComm((PetscObject)mesh->dm), &sect); CHKERRQ(ierr);
	
	ierr = PetscSectionSetNumFields(sect, 1); CHKERRQ(ierr);
	ierr = DMGetDimension(mesh->dm, &dim); CHKERRQ(ierr); 
	ierr = PetscSectionSetFieldComponents(sect, 0, dim); CHKERRQ(ierr);
	ierr = PetscSectionSetChart(sect, vStart, vEnd); CHKERRQ(ierr);

	// Set number of  DOFs
	for (v = vStart; v < vEnd; ++v)
	{
		ierr = PetscSectionSetDof(sect, v, 2); CHKERRQ(ierr);
		ierr = PetscSectionSetFieldDof(sect, v, 0,  1); CHKERRQ(ierr);

	}

	ierr = PetscSectionSetUp(sect); CHKERRQ(ierr);
	ierr = DMSetLocalSection(mesh->dm, sect); CHKERRQ(ierr);
	ierr = PetscSectionDestroy(&sect); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode MeshSetupGeometry(Mesh *mesh)
{
	PetscScalar	*triag;
	PetscSection	coordSect, cellSect;
	Vec		coordinates;
	PetscInt	c, cStart, cEnd, cgdof;
	PetscErrorCode 	ierr;

	ierr = DMClone(mesh->dm, &mesh->dmCell); CHKERRQ(ierr);

	ierr = DMGetCoordinateSection(mesh->dm, &coordSect); CHKERRQ(ierr);
	ierr = DMSetCoordinateSection(mesh->dmCell, PETSC_DETERMINE, coordSect); CHKERRQ(ierr);
	
	ierr = DMGetCoordinatesLocal(mesh->dm, &coordinates); CHKERRQ(ierr);
	ierr = DMSetCoordinatesLocal(mesh->dmCell, coordinates); CHKERRQ(ierr);

	ierr = PetscSectionCreate(PetscObjectComm((PetscObject)(mesh->dm)), &cellSect); CHKERRQ(ierr);

	ierr = DMPlexGetHeightStratum(mesh->dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
	ierr = PetscSectionSetChart(cellSect, cStart, cEnd); CHKERRQ(ierr);

	cgdof = (PetscInt) PetscCeilReal( (PetscReal) sizeof(Triangle) / sizeof(PetscScalar) );
	for (c = cStart; c < cEnd; ++c)
		ierr = PetscSectionSetDof(cellSect, c, cgdof); CHKERRQ(ierr);

	ierr = PetscSectionSetUp(cellSect); CHKERRQ(ierr);
	ierr = DMSetLocalSection(mesh->dmCell, cellSect); CHKERRQ(ierr);
	ierr = PetscSectionDestroy(&cellSect); CHKERRQ(ierr);



	ierr = DMCreateLocalVector(mesh->dmCell, &mesh->triangles); CHKERRQ(ierr);
	ierr = VecGetArray(mesh->triangles, &triag); CHKERRQ(ierr);


	for (c = cStart; c < cEnd; ++c)
	{
		Triangle	*tg;
		PetscInt	numCoords = 0;
		PetscScalar	*coords = NULL;
		PetscInt	i, j;

		ierr = DMPlexPointLocalRef(mesh->dmCell, c, triag, &tg); CHKERRQ(ierr);
		ierr = DMPlexVecGetClosure(mesh->dm, coordSect, coordinates, c, &numCoords, &coords); CHKERRQ(ierr);
		printf("%d %d\n", c, numCoords);
		if (numCoords == 6)
		{
			const PetscInt 	dim = 2;
			
			for (i = 0; i < dim; ++i) 
				tg->v[i] = PetscRealPart(coords[i]);

			for (j = 0; j < dim; ++j)
				for (i = 0; i < dim; ++i)
					tg->J[j*dim+i] = 0.5 * (PetscRealPart(coords[(i+1)*dim+j]) - PetscRealPart(coords[0*dim+j]));

			tg->detJ = Determinant2x2(tg->J);

			ierr = InvertMatrix2x2(tg->J, tg->invJ); CHKERRQ(ierr);

		} else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of coordinates for a triangle 6");
	}

	ierr = VecRestoreArray(mesh->triangles, &triag); CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode MeshDestroy(Mesh *mesh)
{
	PetscErrorCode 	ierr;

	ierr = VecDestroy(&mesh->triangles); CHKERRQ(ierr);
	ierr = DMDestroy(&mesh->dm); CHKERRQ(ierr);
	ierr = DMDestroy(&mesh->dmCell); CHKERRQ(ierr);
	 
	return ierr;
}

PetscErrorCode AssembleOperator_Laplace(Mesh mesh, Mat *A)
{
	PetscInt		c, cStart, cEnd;
	PetscInt		dim;
	DM			cdm;
	Vec			coordinates;
	const PetscScalar	*coords;
	PetscErrorCode 		ierr;

	ierr = DMGetCoordinateDim(mesh.dm, &dim); CHKERRQ(ierr);
	ierr = DMPlexGetHeightStratum(mesh.dm, 0, &cStart, &cEnd); CHKERRQ(ierr);

	PetscScalar	*triag;
	ierr = VecGetArray(mesh.triangles, &triag);

	for (c = cStart; c < cEnd; ++c)
	{	
		Triangle *tg;
		ierr = DMPlexPointLocalRef(mesh.dmCell, c, triag, &tg); CHKERRQ(ierr);
		int dim = 2;
	}

	return ierr;	
}

PetscErrorCode AssembleRHS_Laplace(DM da, Vec *f)
{
	PetscErrorCode 	ierr;
	
	return ierr;	
}

PetscErrorCode ApplyBC_Laplace(DM da, Mat *A, Vec *f)
{
	PetscErrorCode 		ierr;

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


PetscErrorCode FormStressOperatorQ2D(PetscScalar *el_coords, PetscScalar *coeff, PetscScalar *Ke)
{
	PetscErrorCode 	ierr;

	return ierr;
}

PetscErrorCode FormLaplaceRHSQ12D(PetscScalar *el_coords, PetscErrorCode (*f)(PetscScalar *x, PetscScalar *f), PetscScalar *Fe)
{
	PetscErrorCode	ierr;

	
	return ierr;
}

PetscErrorCode Phi(const PetscInt i, const PetscScalar *x, PetscScalar *phi)
{
	if (i == 0)
	{
		*phi = 1.0 - x[0] - x[1];
	} else if (i == 1) {
		*phi = x[0];
	} else if (i == 2) {
		*phi = x[1];
	}
	return 0;
}

PetscErrorCode GradPhi(const PetscInt i, const PetscScalar *x, PetscScalar *gradPhi)
{
	if (i == 0) 
	{
		gradPhi[0] = -1.0;
		gradPhi[1] = -1.0;
	} else if (i == 1) {
		gradPhi[0] =  1.0;
		gradPhi[1] =  0.0;
	} else if (i == 2) {
		gradPhi[0] =  0.0;
		gradPhi[1] =  1.0;
	
	}	
	return 0;
}
