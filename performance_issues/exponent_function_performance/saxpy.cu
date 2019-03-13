__global__ void test_exp(int a)
 {
   float b = 0;

   a = threadIdx.x * a;
   for (int n = 1; n < 1000; ++n)
   {
      b = exp( a * n/100 );
   }
}

int main(void)
{
   float a = 3.14159265359;

   test_exp<<<1, 1>>>(a);

   return 0;
}
