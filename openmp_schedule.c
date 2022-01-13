#include <stdio.h>
#include <sys/time.h>
#include <omp.h>
#include <stdlib.h>
double mat[705][705];
double matcopy[705][705];
double X[705];
double x[705];

void initializeMat(int N){
	int row, col;
	for (row=0; row<N; row++){
		for (col=0; col<N; col++){
			mat[row][col]=rand()/50000.0;
			matcopy[row][col]=mat[row][col];
		}
	mat[row][N] = rand()/50000.0;
	matcopy[row][N]=mat[row][N];

	}
}

void print(int N){
    for (int row=0; row<N; row++, printf("\n")){
        for (int col=0; col<=N; col++){
            printf("%lf ", mat[row][col]);
        }
    printf("\n");
    }
}

void parallel_gaussianElimination_static(int N){
	for (int pivot=0; pivot<N; pivot++){
		#pragma omp parallel for schedule(guided,5)
			for (int row=pivot+1; row<N; row++){
				double f = (double)mat[row][pivot]/mat[pivot][pivot];
				#pragma omp parallel for schedule(guided,5)
					for (int col=pivot+1; col<=N; col++){
						mat[row][col] -= mat[pivot][col]*f;
                    }
				mat[row][pivot] = 0;
			}
    }
    for (int row = N-1; row>= 0; row--){
        x[row] = mat[row][N];
		for (int col=row+1; col<N; col++){
			x[row] -= mat[row][col]*x[col];
		}
        x[row] = x[row]/mat[row][row];
    }
}

void parallel_gaussianElimination_dynamic(int N){
	for (int pivot=0; pivot<N; pivot++){
		#pragma omp parallel for schedule(guided,5)
			for (int row=pivot+1; row<N; row++){
				double f = (double)mat[row][pivot]/mat[pivot][pivot];
				#pragma omp parallel for schedule(guided,5)
					for (int col=pivot+1; col<=N; col++){
						mat[row][col] -= mat[pivot][col]*f;
                    }
				mat[row][pivot] = 0;
			}
    }
    for (int row = N-1; row>= 0; row--){
        x[row] = mat[row][N];
		for (int col=row+1; col<N; col++){
			x[row] -= mat[row][col]*x[col];
		}
        x[row] = x[row]/mat[row][row];
    }
}

void parallel_gaussianElimination_guided(int N){
	for (int pivot=0; pivot<N; pivot++){
		#pragma omp parallel for schedule(guided,5)
			for (int row=pivot+1; row<N; row++){
				double f = (double)mat[row][pivot]/mat[pivot][pivot];
				#pragma omp parallel for schedule(guided,5)
					for (int col=pivot+1; col<=N; col++){
						mat[row][col] -= mat[pivot][col]*f;
                    }
				mat[row][pivot] = 0;
			}
    }
    for (int row = N-1; row>= 0; row--){
        x[row] = mat[row][N];
		for (int col=row+1; col<N; col++){
			x[row] -= mat[row][col]*x[col];
		}
        x[row] = x[row]/mat[row][row];
    }
}
 
 void serial_gaussianElimination(int N){
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
int main(){
	int N=700;
	struct timeval TimeValue_Start;
	struct timezone TimeZone_Start;
	struct timeval TimeValue_Final;
	struct timezone TimeZone_Final;
	long time_start, time_end;
	double time_overhead;
	FILE *serail_output = fopen("serial.txt", "w");
	FILE *parallel_output = fopen("parallel.txt", "w");
	initializeMat(N);
	gettimeofday(&TimeValue_Start, &TimeZone_Start);
	serial_gaussianElimination(N);
	gettimeofday(&TimeValue_Final, &TimeZone_Final);
	time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
	time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
	time_overhead = (time_end - time_start)/1000000.0;
	printf("\n %lf",time_overhead);
	for(int i=0;i<N;i++){
		fprintf(serail_output, "%lf\n", X[i]);
	}
    // FILE* ptr=fopen("input.txt","w");
    // fprintf(ptr,"%d\n",N);
    // for (int i=0;i<N;i++)
    // {
    //     for(int j=0;j<N;j++)
    //     {
    //         fprintf(ptr,"%.2lf",mat[i][j]);
    //         if(j!=N-1) fprintf(ptr," ");
    //     }
    //     fprintf(ptr,"\n");
    // }
    // for(int i=0;i<N;i++)
    // {
    //     fprintf(ptr,"%.2lf",mat[i][N]);
    //     if(i!=N-1) fprintf(ptr," ");
    // }
    for (int row=0; row<N; row++){
		for (int col=0; col<N; col++){
            mat[row][col] = matcopy[row][col];
        }
    }
	gettimeofday(&TimeValue_Start, &TimeZone_Start);
	parallel_gaussianElimination_static(N);
	gettimeofday(&TimeValue_Final, &TimeZone_Final);
	time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
	time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
	time_overhead = (time_end - time_start)/1000000.0;
	printf("\n %lf \n",time_overhead);
    for (int row=0; row<N; row++){
		for (int col=0; col<N; col++){
            mat[row][col] = matcopy[row][col];
        }
    }
	gettimeofday(&TimeValue_Start, &TimeZone_Start);
	parallel_gaussianElimination_dynamic(N);
	gettimeofday(&TimeValue_Final, &TimeZone_Final);
	time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
	time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
	time_overhead = (time_end - time_start)/1000000.0;
	printf("\n %lf \n",time_overhead);
    for (int row=0; row<N; row++){
		for (int col=0; col<N; col++){
            mat[row][col] = matcopy[row][col];
        }
    }
	gettimeofday(&TimeValue_Start, &TimeZone_Start);
	parallel_gaussianElimination_guided(N);
	gettimeofday(&TimeValue_Final, &TimeZone_Final);
	time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
	time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
	time_overhead = (time_end - time_start)/1000000.0;
	printf("\n %lf \n",time_overhead); 
	for(int i=0;i<N;i++){
		fprintf(parallel_output, "%lf\n", x[i]);
	}
    return 0;
}