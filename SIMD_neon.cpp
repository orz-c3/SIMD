# include <arm_neon.h> // use Neon
# include <sys/time.h>
#include <cstring>  // 包含对memset的定义

# include <iostream>
using namespace std;

const int n = 1000;
float** A;
float** B;

float32x4_t va = vdupq_n_f32(0);
float32x4_t vx = vdupq_n_f32(0);
float32x4_t vaij = vdupq_n_f32(0);
float32x4_t vaik = vdupq_n_f32(0);
float32x4_t vakj = vdupq_n_f32(0);


void init() {
    A = new float*[n];
    B = new float*[n];
    for (int i = 0; i < n; i++) {
        A[i] = new float[n];
        B[i] = new float[n];
        memset(A[i], 0, n * sizeof(float));
        A[i][i] = 1.0;
        for (int j = i + 1; j < n; j++) {
            A[i][j] = rand() % 1000;
        }
    }

    // Additional matrix operations as before
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] += A[k][j];
                A[i][j] = (int)A[i][j] % 1000;
            }
        }
    }
}

void init_align() {
    A = new float*[n];
    B = new float*[n];

    for (int i = 0; i < n; i++) {
        if (posix_memalign(reinterpret_cast<void**>(&A[i]), 16, n * sizeof(float)) != 0 ||
            posix_memalign(reinterpret_cast<void**>(&B[i]), 16, n * sizeof(float)) != 0) {
            cerr << "Memory allocation failed for row " << i << endl;
            exit(EXIT_FAILURE);
        }
        memset(A[i], 0, n * sizeof(float));
        A[i][i] = 1.0;
        for (int j = i + 1; j < n; j++) {
            A[i][j] = rand() % 1000;
        }
    }

    // Additional matrix operations as before
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] += A[k][j];
                A[i][j] = (int)A[i][j] % 1000;
            }
        }
    }
}

void cleanup() {
    for (int i = 0; i < n; i++) {
        free(A[i]);
        free(B[i]);
    }
    delete[] A;
    delete[] B;
}



void f_ordinary()
{
    for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j++)
		{
			A[k][j] = A[k][j] * 1.0 / A[k][k];
		}
		A[k][k] = 1.0;

		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
}


void f_ordinary_cache()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            B[j][i] = A[i][j];
            A[i][j] = 0; 
        }
    }

    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            A[k][j] = A[k][j] * 1.0 / A[k][k];
        }
        A[k][k] = 1.0;



        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[i][j] = A[i][j] - B[k][i] * A[k][j];
            }
            
        }
    }
}




void f_neon()
{
    for (int k = 0; k < n; k++)
	{
	    float32x4_t vt=vdupq_n_f32(A[k][k]);
	    int j;
		for (j = k + 1; j+4 <= n; j+=4)
		{
		    va=vld1q_f32(&(A[k][j]) );
			va= vdivq_f32(va,vt);
			vst1q_f32(&(A[k][j]), va);
		}

		for(; j<n; j++)
        {
            A[k][j]=A[k][j]*1.0 / A[k][k];

        }
		A[k][k] = 1.0;

		for (int i = k + 1; i < n; i++)
		{
		    vaik=vdupq_n_f32(A[i][k]);

			for (j = k + 1; j+4 <= n; j+=4)
			{
				vakj=vld1q_f32(&(A[k][j]));
				vaij=vld1q_f32(&(A[i][j]));
				vx=vmulq_f32(vakj,vaik);
				vaij=vsubq_f32(vaij,vx);

				vst1q_f32(&A[i][j], vaij);
			}

			for(; j<n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }

			A[i][k] = 0;
		}
	}
}


void f_neon_cache()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            B[j][i] = A[i][j];
            A[i][j] = 0; // 相当于原来的 A[i][k] = 0;
        }
    }


    for (int k = 0; k < n; k++)
	{
	    float32x4_t vt=vdupq_n_f32(A[k][k]);
	    int j;
		for (j = k + 1; j+4 <= n; j+=4)
		{
		    va=vld1q_f32(&(A[k][j]) );
			va= vdivq_f32(va,vt);
			vst1q_f32(&(A[k][j]), va);
		}

		for(; j<n; j++)
        {
            A[k][j]=A[k][j]*1.0 / A[k][k];

        }
		A[k][k] = 1.0;

		for (int i = k + 1; i < n; i++)
		{
		    vaik=vdupq_n_f32(B[k][i]);

			for (j = k + 1; j+4 <= n; j+=4)
			{
				vakj=vld1q_f32(&(A[k][j]));
				vaij=vld1q_f32(&(A[i][j]));
				vx=vmulq_f32(vakj,vaik);
				vaij=vsubq_f32(vaij,vx);

				vst1q_f32(&A[i][j], vaij);
			}

			for(; j<n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
		}
	}
}




void f_neon_align()
{
    for(int k = 0; k < n; k++)
    {
        float32x4_t vt = vdupq_n_f32(A[k][k]);
        int j = k + 1;
        
        // 对齐检查到下一个16字节边界
        while(((uintptr_t)(&A[k][j]) % 16) != 0 && j < n)
        {
            A[k][j] = A[k][j] / A[k][k];
            j++;
        }
        
        // 使用对齐的指令进行处理
        for(; j + 4 <= n; j += 4)
        {
            float32x4_t va = vld1q_f32(&A[k][j]);  // 对齐加载
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k][j], va);  // 对齐存储
        }

        for(; j < n; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;

        for(int i = k + 1; i < n; i++)
        {
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
            j = k + 1;
            
            // 对齐检查到下一个16字节边界
            while(((uintptr_t)(&A[i][j]) % 16) != 0 && j < n)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
                j++;
            }

            for(; j + 4 <= n; j += 4)
            {
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }

            for(; j < n; j++)
            {
                A[i][j] = A[i][j] - A[k][j] * A[i][k];
            }
            A[i][k] = 0.0;
        }
    }
}

void getResult()
{
    for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << A[i][j] << " ";
		}
		cout << endl;
	}
}



int main()
{

    struct timeval head,tail;



    init();
    gettimeofday(&head, NULL);//开始计时
    f_ordinary();
    gettimeofday(&tail, NULL);//结束计时
    double seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_ordinary: "<<seconds<<" ms"<<endl;


    init();
    gettimeofday(&head, NULL);//开始计时
    f_ordinary_cache();
    gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_ordinary_cache: "<<seconds<<" ms"<<endl;

    init();
    gettimeofday(&head, NULL);//开始计时
    f_neon();
    gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_neon: "<<seconds<<" ms"<<endl;

    init();
    gettimeofday(&head, NULL);//开始计时
    f_neon_cache();
    gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_neon_cache: "<<seconds<<" ms"<<endl;

    init_align();
    gettimeofday(&head, NULL);//开始计时
    f_neon_align();
    gettimeofday(&tail, NULL);//结束计时
    seconds = ((tail.tv_sec - head.tv_sec)*1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;//单位 ms
    cout<<"f_neon_alignment: "<<seconds<<" ms"<<endl;

    cleanup();

    

}






