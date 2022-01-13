#include<stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define N 700
double X[N];
void serial_gaussianElimination(double mat[N][N+1]){
	for (int pivot=0; pivot<N; pivot++){
        for (int row=pivot+1; row<N; row++){
            double f = mat[row][pivot]/mat[pivot][pivot];
            for (int col=pivot+1; col<=N; col++){
                mat[row][col] -= mat[pivot][col]*f;
            }
			mat[row][pivot] = 0;
        }
    } 
    for (int row = N-1; row >= 0; row--){
        X[row] = mat[row][N];
        for (int col=row+1; col<N; col++){
            X[row] -= mat[row][col]*X[col];
        }

        X[row] = X[row]/mat[row][row];
    }

}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    int map[N];
    double A[N][N],b[N],c[N],x[N],sum=0.0;
    int rank, nprocs;
	FILE *serail_output = fopen("serial.txt", "w");
	FILE *parallel_output = fopen("parallel.txt", "w");
    clock_t begin1, end1;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank==0){
        for (int row=0; row<N; row++){
            for (int col=0; col<N; col++){
                A[row][col]=rand()/50000.0;
            }
            b[row]=rand()/50000.0;
        }
        double mat[N][N+1];
        for(int row=0;row<N;row++){
            for(int col=0;col<N;col++){
                mat[row][col] = A[row][col];
            }
            mat[row][N] = b[row];
        }
        begin1 = clock();
        serial_gaussianElimination(mat);
        end1 = clock();
        printf("\n\n %f  ", (double)(end1 - begin1) / CLOCKS_PER_SEC);
		for(int i=0;i<N;i++){
			fprintf(serail_output, "%lf\n", X[i]);
		}
    }
    begin1 = clock();

    MPI_Bcast (&A[0][0],N*N,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast (b,N,MPI_DOUBLE,0,MPI_COMM_WORLD);    

    for(int row=0; row<N; row++){
        map[row]= row % nprocs;
    } 

    for(int pivot=0;pivot<N;pivot++){
        MPI_Bcast (&A[pivot][pivot],N-pivot,MPI_DOUBLE,map[pivot],MPI_COMM_WORLD);
        MPI_Bcast (&b[pivot],1,MPI_DOUBLE,map[pivot],MPI_COMM_WORLD);
        for(int row= pivot+1; row<N; row++){
            if(map[row] == rank){
                c[row]=A[row][pivot]/A[pivot][pivot];
            }
        }               
        for(int row= pivot+1; row<N; row++)  {       
            if(map[row] == rank){
                for(int col=0;col<N;col++){
                    A[row][col]=A[row][col]-( c[row]*A[pivot][col] );
                }
                b[row]=b[row]-( c[row]*b[pivot] );
            }
        }
    }

    if (rank==0){ 
        x[N-1]=b[N-1]/A[N-1][N-1];
        for(int row=N-2;row>=0;row--){
            sum=0;
            for(int col=row+1;col<N;col++){
                sum=sum+A[row][col]*x[col];
            }
            x[row]=(b[row]-sum)/A[row][row];
        }
        end1 = clock();
    }


    if (rank==0){ 
        printf("%f\n", (double)(end1 - begin1) / CLOCKS_PER_SEC);
		for(int row=0;row<N;row++){
			fprintf(parallel_output, "%lf\n", x[row]);
		}
    }
    MPI_Finalize();
	
    return(0);
}