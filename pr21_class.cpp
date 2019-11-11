//Prosirujemo pr20, tako da prosirujemo razred vektor funkcijom 
//ZbrojiSa() koja zbraja vektor s nekim drugim vektorom. 
//Promjenit cemo povratni tip funkcije na na Vektor ,cime ce 
//funkcijski clan vracati referencu na objekt za koji je pozvan

#include <iostream>
#include <iomanip>
using namespace std;

class Vektor
	{
	 public:      //ovim podatkovnim clanovima i funkcijskim clanovima 
	 float ax,ay; //dozvpoljava se javni pristup,pa ih mozemo inicijali.
	 Vektor &MnoziSkalarom(float skalar); //i pozvati u gl. programu
	 Vektor &ZbrojiSa(float zx,float zy);
	};

Vektor &Vektor::MnoziSkalarom(float skalar)
	{
		ax*=skalar;
		ay*=skalar;
		return *this;
	}

Vektor &Vektor::ZbrojiSa(float zx,float zy)
	{
	  ax+=zx;
	  ay+=zy;
	  return *this;
	}
int main()
	{
	 Vektor v;
	 cout<<"Unesi projekciju vektora na os x:";
	 cin>>v.ax;
	 cout<<"Unesi projekciju vektora na os y:";
	 cin>>v.ay;
	 cout<<"Uneseni vektor je: ("<<v.ax<<","<<v.ay<<")"<<endl;
	 v.MnoziSkalarom(4).ZbrojiSa(5,-7);
	 cout<<"Sada je vektor: ("<<v.ax<<","<<v.ay<<")"<<endl;
	 return 0;
	}
