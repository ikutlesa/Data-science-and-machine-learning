//primjer razlikovanaja strukture u C-u i klase u C++, program mnozi 
//vektor sa skalarom

#include <iostream>
#include <iomanip>
#include <stdio.h>
using namespace std;

//preko strukture
struct Vektor1
	{
	  float ax,ay;   //prokekcije na os x i os y
	};
void MnoziVektorSkalarom(struct Vektor1 v, float skalar)
	{
	  v.ax*=skalar;
	  v.ay*=skalar;
	}

//ista stvar preko klase
class Vektor2
	{
	public:
		float ax,ay;
		void MnoziSkalarom(float skalar);	
	};
void Vektor2::MnoziSkalarom(float skalar)
	{
	 ax*=skalar;
	 ay*=skalar;
	}
int main()
	{
	 struct Vektor1 v1;
	 Vektor2 v2;
	 printf("Unesi projekciju na x os:");
	 cin>>v1.ax;
	 printf("Unesi rojekciju na y os:");
	 cin>>v1.ay;
	 cout<< "Uneseni vektor je: ("<<v1.ax<<","<<v1.ay<<")"<<endl;
	 MnoziVektorSkalarom(v1,5);
	 cout<<"Sada je vektor: ("<<v1.ax<<","<<v1.ay<<")"<<endl;
	 v2.MnoziSkalarom(5.0);
	 return 0;
	}

	 
 