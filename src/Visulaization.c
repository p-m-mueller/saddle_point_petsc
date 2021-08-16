#include "Visualization.h"

PetscErrorCode WriteVTK(DM da_u, Vec u, const char *filename)
{
	MPI_Comm	comm;
	MPI_File	fh;
	DM		cda;
	Vec		coords;
	const DMDACoor2d 	**_coords;
	Vec		lu;
	const 	Field	**_u;
	DMDALocalInfo	info;
	PetscInt 	i, j, k, d;
	int		rank, size;
	int 		nPoints, nElements, nodesPerElement;
	double		*points;
	const PetscInt	*elements;
	unsigned int 	offset;
	PetscErrorCode 	ierr;

	ierr = PetscObjectGetComm((PetscObject)da_u, &comm); CHKERRQ(ierr);

	ierr = DMGetCoordinateDM(da_u, &cda); CHKERRQ(ierr);
	ierr = DMGetCoordinatesLocal(da_u, &coords); CHKERRQ(ierr);
	ierr = DMDAVecGetArrayRead(cda, coords, &_coords); CHKERRQ(ierr); 

	ierr = DMGetLocalVector(da_u, &lu); CHKERRQ(ierr);
	ierr = DMDAVecGetArrayRead(da_u, lu, &_u); CHKERRQ(ierr);

	ierr = DMDAGetLocalInfo(da_u, &info); CHKERRQ(ierr);

	ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
	ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);


	nPoints = info.xm * info.ym;
	points = (double*)malloc( nPoints * 3 * sizeof(double) );

	for (j = info.ys; j < info.ys+info.ym; ++j)
		for (i = info.xs; i < info.xs+info.xm; ++i)
		{
			int idx = (j-info.ys) * info.xm + (i-info.xs);
			points[idx * 3 + 0] = _coords[j][i].x;
			points[idx * 3 + 1] = _coords[j][i].y;
			points[idx * 3 + 2] = 0.0;
		}
	
	ierr = DMDAGetElements(da_u, &nElements, &nodesPerElement, &elements); CHKERRQ(ierr);
	
	
	ierr = MPI_File_open(comm, filename, MPI_MODE_CREATE + MPI_MODE_WRONLY, MPI_INFO_NULL, &fh); CHKERRQ(ierr);

	WriteVTKHeader(comm, fh, "Very cool data...", &offset);
	WriteVTKPoints(comm, fh, nPoints, points, &offset); 
	WriteVTKPolygones(comm, fh, nElements, elements, nodesPerElement, &offset);

	ierr = MPI_File_close(&fh);

	ierr = DMDARestoreElements(da_u, &nElements, &nodesPerElement, &elements); CHKERRQ(ierr);

	free(points);

	ierr = DMDAVecRestoreArray(da_u, lu, &_u); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(cda, coords, &_coords); CHKERRQ(ierr);

	return ierr;
}

static int WriteVTKHeader(MPI_Comm comm, MPI_File fh, const char *comment, unsigned int *offset)
{
	char 		line[256];
	unsigned int 	len;
	int 		rank;
	MPI_Request 	request;
	
	*offset = 0;

	MPI_Comm_rank(comm, &rank);

	if (rank == 0)
	{
		len = snprintf(line, 256, "%s\n", "# vtk DataFile Version 2.0");
		MPI_File_write_at(fh, *offset, line, len, MPI_CHAR, MPI_STATUS_IGNORE);
		*offset += len;
		
		len = snprintf(line, 256, "%s\n", comment);
		MPI_File_write_at(fh, *offset, line, len, MPI_CHAR, MPI_STATUS_IGNORE);
		*offset += len;
		
		len = snprintf(line, 256, "%s\n", "ASCII");
		MPI_File_write_at(fh, *offset, line, len, MPI_CHAR, MPI_STATUS_IGNORE);
		*offset += len;
	
		len = snprintf(line, 256, "%s\n", "DATASET POLYDATA");
		MPI_File_write_at(fh, *offset, line, len, MPI_CHAR, MPI_STATUS_IGNORE);
		*offset += len;
	}

	MPI_Ibcast(offset, 1, MPI_INT, 0, comm, &request);

	return 0;
}

static int WriteVTKPoints(MPI_Comm comm, MPI_File fh, const int n, const double *points, unsigned int *offset)
{
	char line[256];
	unsigned int len;
	int rank, size;
	MPI_Request request;
	unsigned int nglobal;
	
	MPI_Comm_size(comm, &size);
	
	if (size == 1)
	{
		nglobal = n;
		len = snprintf(line, 256, "%s %d %s\n", "POINTS", nglobal, "double");
		MPI_File_write_at(fh, *offset, line, len, MPI_CHAR, MPI_STATUS_IGNORE);
		*offset += len;
		
		for (int i = 0; i < n; ++i)
		{
			len = snprintf(line, 256, "%lf %lf %lf\n", points[i*3], points[i*3+1], points[i*3+2]);
			MPI_File_write_at(fh, *offset, line, len, MPI_CHAR, MPI_STATUS_IGNORE);
			*offset += len;
		}
		
	} else {

		MPI_Comm_rank(comm, &rank);
		MPI_Allreduce(&n, &nglobal, 1, MPI_INT, MPI_SUM, comm);
	
		if (rank == 0)
		{
			len = snprintf(line, 256, "%s %d %s\n", "POINTS", nglobal, "double");
			MPI_File_write_at(fh, *offset, line, len, MPI_CHAR, MPI_STATUS_IGNORE);
			*offset += len;
			
			for (int i = 0; i < n; ++i)
			{
				len = snprintf(line, 256, "%lf %lf %lf\n", points[i*3], points[i*3+1], points[i*3+2]);
				MPI_File_write_at(fh, *offset, line, len, MPI_CHAR, MPI_STATUS_IGNORE);
				*offset += len;
			}
	
			MPI_Isend(offset, 1, MPI_INT, rank+1, 0, comm, &request);
			MPI_Recv(offset, 1, MPI_INT, size-1, 0, comm, MPI_STATUS_IGNORE);
	
		} else if (rank == size-1) {
	
			MPI_Recv(offset, 1, MPI_INT, rank-1, 0, comm, MPI_STATUS_IGNORE);
	
			for (int i = 0; i < n; ++i)
			{
				len = snprintf(line, 256, "%lf %lf %lf\n", points[i*3], points[i*3+1], points[i*3+2]);
				MPI_File_write_at(fh, *offset, line, len, MPI_CHAR, MPI_STATUS_IGNORE);
				*offset += len;
			}
			
			MPI_Send(offset, 1, MPI_INT, 0, 0, comm);
	
		} else {
	
			MPI_Recv(offset, 1, MPI_INT, rank-1, 0, comm, MPI_STATUS_IGNORE);
	
			for (int i = 0; i < n; ++i)
			{
				len = snprintf(line, 256, "%lf %lf %lf\n", points[i*3], points[i*3+1], points[i*3+2]);
				MPI_File_write_at(fh, *offset, line, len, MPI_CHAR, MPI_STATUS_IGNORE);
				*offset += len;
			}
	
			MPI_Send(offset, 1, MPI_INT, rank+1, 0, comm);
		}

	}
}

static int WriteVTKPolygones(MPI_Comm comm, MPI_File fh, const int n,  const int *elements, const int nodesPerElement, unsigned int *offset)
{
	char line[256];
	unsigned int len;
	int rank, size;
	MPI_Request request;
	int nglobal;

	MPI_Comm_size(comm, &size);

	if (size == 1)
	{
		nglobal = n;
		len = snprintf(line, 256, "\nPOLYGONS %ld %ld\n", nglobal, nglobal*nodesPerElement);
		MPI_File_write_at(fh, *offset, line, len, MPI_CHAR, MPI_STATUS_IGNORE);
		*offset += len;

		for (int i = 0; i < n; ++i)
		{
			const PetscInt	*element = &elements[i];
			WriteVTKPolygonesLine(fh, element, nodesPerElement, offset);
		}
		
	} else {
		MPI_Comm_rank(comm, &rank);
		MPI_Allreduce(&n, &nglobal, 1, MPI_INT, MPI_SUM, comm);

		if (rank == 0)
		{
			len = snprintf(line, 256, "\nPOLYGONS %ld %ld\n", nglobal, nglobal*nodesPerElement);
			MPI_File_write_at(fh, *offset, line, len, MPI_CHAR, MPI_STATUS_IGNORE);
			*offset += len;
		
			for (int i = 0; i < n; ++i)
			{
				const PetscInt	*element = &elements[i];
				WriteVTKPolygonesLine(fh, element, nodesPerElement, offset);
			}
			
			MPI_Isend(offset, 1, MPI_INT, rank+1, 0, comm, &request);
			MPI_Recv(offset, 1, MPI_INT, size-1, 0, comm, MPI_STATUS_IGNORE);

		} else if ( rank == size-1) {
			
			MPI_Recv(offset, 1, MPI_INT, rank-1, 0, comm, MPI_STATUS_IGNORE);
		
			for (int i = 0; i < n; ++i)
			{
				const PetscInt	*element = &elements[i];
				WriteVTKPolygonesLine(fh, element, nodesPerElement, offset);
			}
			
			MPI_Send(offset, 1, MPI_INT, 0, 0, comm);

		} else {
			
			MPI_Recv(offset, 1, MPI_INT, rank-1, 0, comm, MPI_STATUS_IGNORE);

			for (int i = 0; i < n; ++i)
			{
				const PetscInt *element = &elements[i];
				WriteVTKPolygonesLine(fh, element, nodesPerElement, offset);
			}
			
			MPI_Send(offset, 1, MPI_INT, rank+1, 0, comm);
		}
	}
}

static inline int WriteVTKPolygonesLine(MPI_File fh, const int *element, const int nodesPerElement, unsigned int *offset)
{
	char line[256];
	int len;

	len = snprintf(line, 256, "%ld ", nodesPerElement);
	MPI_File_write_at(fh, *offset, line, len, MPI_CHAR, MPI_STATUS_IGNORE);
	*offset += len;
		
	for (int j = 0; j < nodesPerElement; ++j)
	{
		len = snprintf(line, 256, "%ld ", element[j]);
		MPI_File_write_at(fh, *offset, line, len, MPI_CHAR, MPI_STATUS_IGNORE);
		*offset += len;
	}
	len = snprintf(line, 256, "\n");
	MPI_File_write_at(fh, *offset, line, len, MPI_CHAR, MPI_STATUS_IGNORE);
	*offset += len;
}
