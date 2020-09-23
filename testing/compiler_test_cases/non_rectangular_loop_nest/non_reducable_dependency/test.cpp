int main()
{

   // 10 'zones', 4 'nodes' per zone, 2 values per node = 80 values
	int a[10][2][4];
   int num_v[10];

   int start_n[10];
   int end_n[10];

   int foo = 0;

   for (int z = 0 ; z < 10; ++z)
	{
	#pragma omp target teams distribute parallel for collapse(2) 
	{
		for (int v = 0; v < num_v[z]; ++v)
		{
	   	for (int n = start_n[z]; n < end_n[z]; ++n)
			{
				a[z][v][n] = 5;
			}
		}
	}
	}
	return 0;
}
