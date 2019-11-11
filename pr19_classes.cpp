//primjer jednostvnog rada s razredima sa podatkovnim clanovima koji su
//analogni strukturama u C-u;
#include <iostream>
#include <iomanip>
using namespace std;

class Racunalo //oznacava da ce clanovi razreda biti javno dostuoni
	{
	public:
		int kBMemorije;
		int brojDiskova;
		int megahertza;
	};

int main()
	{
	Racunalo mojKucniCray;
	Racunalo &refRacunalo=mojKucniCray;

    int a=mojKucniCray.kBMemorije=64;  //. operator za pristup clanovima
    refRacunalo.brojDiskova=5;   //odnosi se na isti objekt

    Racunalo *pokRacunalo=&mojKucniCray; //pokzivac na objekt razr. Racunalo
   //Clanu pristupamo u ovom slucaju preko operatora ->
    int b=pokRacunalo->megahertza=16;
    //cout<<mojKucniCray<<endl; ne sljaka
	cout<<a<<endl<<b<<endl; //ovo sljaka
	return 0;
	}