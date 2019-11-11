//Program je nadogradnja na pr21, uz koristenje konstruktora=funkcijski
//clan koji se automatski poziva prilikom stvaranja objekta u svrhu 
//njegove inicijalizacije, imena je kao i razred te nema povratni tip

#include <iostream>
#include <iomanip>
using namespace std;

class Vektor
{
//davanje pristupa privatnim clanovima razreda fumkciji ZbrojiVektore
//to postizemo sljedecom sintaksom:
	friend Vektor ZbrojiVektore(Vektor a,Vektor b){
	//takodjer je moguce tu funkciju odmah i definirati
      Vektor c;
	  c.ax=a.ax+b.ax;
	  c.ay=a.ay+b.ay;
	  return c;
	}
private:
	float ax,ay;
public:
	//deklarirana su dva konstruktora
	Vektor();  //inicijalizira novostvoreni vektor na nul-vektor
	Vektor(float x,float y)
		{
		  ax=x;
		  ay=y;
		}
    ~Vektor();  //destruktor,koji nam u ovom sl. ne treba jer ne 
				//alociramo memoriju koji oslobadja sve resurse 
				//dodjeljene objektu
	
	void PostaviXY(float x,float y)
		{
		 ax=x;
		 ay=y;
		}
	float DajX()
		{
		 return ax;
		}
	float DajY()
		{
		  return ay;
		 }
	void MnoziSkalarom(float skalar);
};

Vektor::Vektor()
	{
	  ax=0;
	  ay=0;
	}

Vektor::~Vektor() //definicija destruktora koji u ovom sl. ne radi nista
{			      //jer se u programu ne alocira memorija
		}  

int main()
	{
	  Vektor mojInicijaliziraniVektor(3.0,-7.0);
	  cout<<"Projekcija inicijaliziranog vektora na x os je:"
		  <<mojInicijaliziraniVektor.DajX()<<endl; 
	  Vektor bezParametara;
	  cout<<"Projekcija nul vektora na y os je:"
		  <<bezParametara.DajY()<<endl;
	  Vektor a(12.0,3.2),b(-5,2);
	  cout<<"Rezultat je:"<<"("<<ZbrojiVektore(a,b).DajX()<<","<<ZbrojiVektore(a,b).DajY()<<")"<<endl;
	  return 0;
}
