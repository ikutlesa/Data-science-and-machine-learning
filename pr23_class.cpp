//ovo je primjer programa sa vjezbi koji stvara klasu Date u svrhu 
//kojomse inicijalizira i ispisuje dan,mjesec i godina nekog datuma

#include <iostream.h>

//slijedi definicija klase 
class Date
{
private://'private' se ne treba pisati jer on po defaultu tu postoji
	int day; //day, month i year nisu varijable
	int month;
	int year;
public: //kvalifikator vidljivosti
	Date(int d,int m,int y); //f-ja konstruktor
	void Print();
	Date(int y); //konstruktor
	~Date(); //f-ja destruktor,moze postojati samo jedan 
}; //obavezno staviti ; na kraju definicije klase ili strukture

//:: scope operator;eng. scope = hrv. podrucje djelovanja
Date::Date(int d,int m,int y)
	{
	  day=d;
	  month=m;
	  year=y;
	}

Date::Date(int y)
	{
	  day=1;
	  month=1;
	  year=y;
	}

Date::~Date()
	{
		}

void Date::Print()
	{
	  cout<<day<<endl;
	  cout<<month<<endl;
	  cout<<year<<endl;
	}

int main()
	{
	  Date d(19,1,2001); //implicitni poziv konstruktora!! prima i 4. 
	  //skriveni parametar isto i d.print() prim 1 skriveni parametar	
	  //taj skriveni parametar je this i njegov tip je Date*
	  //nikad nemogu i NESMIJEM napisati nesto oblika:Date d; komajler
	  //ce u tom slucaju javiti gresku!!
	  d.Print(); //=Date::print(&d)
	  Date da(2000);
	  cout<<endl;
	  da.Print();
	  //cout<<sizeof(d)<<endl;
	  //zabranjeno je od kompajlera: d.day=20; jer se day nalazi  u
	  //private dijelu klase, a da je u public bilo bi dozvoljeno
	  return 0;
}
