//*****************************************************************************
//                                                         
//			            Growth simulation							 
//														 
//					v. 0.2.2 - 20140704
//
//				(C) Nicola Ferralis - 2006-2014
//
//		This program (source code and binaries) is free software; 
//		you can redistribute it and/or modify it under the terms of the
//		GNU General Public License as published by the Free Software 
//		Foundation; either version 2 of the License, or any later version.
//
//		This program is distributed in the hope that it will be useful,
//		but WITHOUT ANY WARRANTY; without even the implied warranty of
//		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//		GNU General Public License for more details.
//
//		You can find a complete copy of the GNU General Public License at:
//		http://www.gnu.org/licenses/gpl.txt
//												             
//**********************************************************************************


#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <iomanip>

#define OPENMP

using namespace std;

#define Maxim(a,b) (((a)>(b))?(a):(b))
#define Minim(a,b) (((a)<(b))?(a):(b))



int type, vplot, j, i, rel, flag, multfacttime, tdep, NN;


char nameIN[20], nameout[20], nameout2[20], nameout3[20]; 
char namenoext[20], mxtc[20];
char datextension[]=".dat";
//char grextension[]="_gr";
char extension[]=".dat";
char extension2[]=".dat";
char ini[]="gr.cfg";
char version[]="0.2.1 - 20140704";

double Xp, Hp;

double Per(int n, double s);
double prec, constant;

double prec0 = 0.0005;
double constant0 =1;
int vplot0=1;
int multfacttime0=1000;

double PI=3.14159;

int ReadKey();
float ReadKeyF();
void IniPlot(char* name, string xlabel, string ylabel, int pt);
void Plot(char* name);
void SavePlot(char *name);



double Per(int n, double s);
int random(int MaxValue);
double AvHeight(int steps, double * Hi);
double IntWidth(int steps, double av, double * Hi);
void deposition(int i, double * Xi, double * Hi);
void relaxation(int t, int imax, double * Xi, double * Hi);

FILE *command, *gpin;

int main()
{
	ifstream infile2;
	
	infile2.open(ini);    		  //gcc

	if(!infile2)
	{	
		ofstream outfile2(ini);
		outfile2<<prec0<<"\n"<<constant0<<"\n"<<vplot0<<"\n"<<multfacttime0<<"\n";
		prec=prec0;
		constant=constant0;
		vplot=vplot0;
		multfacttime=multfacttime0;
	}

	infile2>> prec>>constant>>vplot>>multfacttime;
	#if MSini ==1
	_chdir(mainpath);			
	#endif

	cout<<"\n_______________________________________________________";
	cout<< "\nGrowth simulation \n\n";
	cout << "0) Exit\t\t\t\t2) View parameters\n";
	cout<<"1) Start growth\t\t\t3) Change parameters\n";
	cout<<" \t\t\t\t4) About & Manual\n";
	
	type=ReadKey();

	if((type!=1) & (type!=2) & (type!=3) & (type!=4) & (type!=5) & (type!=6) & (type!=7) & (type!=8))
		{cout<<"Bye Bye\n";
		return 0;}

	if(type==4)
		{cout<<"\nGrowth simulation (v. "<<version<<")\n\n";
		cout<<" 1. Start the growth. The program asks for:\n\n";
        cout<<"    a. number of atoms in a 1D periodic substrate.\n\n";
		cout<<"    b. The number of incoming atoms (or the time for the deposition);\n";
		cout<<"    c. Growth mode: \n\n";
		cout<<"       i. Ballistic Growth: New atoms stick either to the first or second NN; \n";
		cout<<"       ii. Random Growth: New atoms sit on the position thet are deposited; \n\n";
		cout<<" 3. Relaxation: after deposition, atoms can move to near column\n"; 
		cout<<"    with lowest height. \n"; 
		cout<<"\nComment or suggestion: Nicola Ferralis <ferralis@mit.edu>\n";
		return main();}

	if(type==3)
		{
		int tmp, preftype;

		cout<<"\n1: Restore default,  2: custom parameters,  3: no change ";
		preftype=ReadKey();
				
		if(preftype==1)	
			{prec=prec0;
			constant=constant0;
			vplot=vplot0;
			multfacttime=multfacttime0;
			cout<<"\n Default parameters succesfully restored!\n\n"; }

		if(preftype==2)	
		{
		cout<<"Actual values:\n\n1) Precision: "<<prec<<"\n";
		cout<<"   New precision: (suggested: 0.0005) ";
		prec=ReadKeyF();		
		cout<<"\n2) Lattice constant: "<<constant<<"\n";
		cout<<"   New lattice constant: (suggested: 1) ";
		constant=ReadKeyF();		
		cout<<"\n3) Do you want the curve to be plotted? ";
		cout<<"\n   (Gnuplot needs to be installed to take advantage of this feature)";
		cout<<"\n   Curve is currently: ";
		
			if(vplot ==1)
				{cout <<"plotted. \n";}
			if (vplot ==2)
				{cout<<"NOT plotted.\n";}

			cout<<"   New value (1: plotted,  2: not plotted,  other: no change): ";
			//cin>>vplot;
			tmp=vplot;
			vplot=ReadKey();
			if(vplot!=1 && vplot!=2)
				{cout<<" Value not updated\n\n";
				vplot=tmp;}
		cout<<"\n4) Multiplication factor for the growth time: "<<multfacttime<<"\n";
		cout<<"   New multipication factor: (suggested: 1000) ";
		multfacttime=ReadKey();
		}
		if(preftype!=1 && preftype!=2)
			{cout<<"\n Parameters not updated!\n\n"; 
			return main();}

		
		ofstream outfile2(ini);
		outfile2<<prec<<"\n"<<constant<<"\n"<<vplot<<"\n"<<multfacttime<<"\n";
		outfile2.close();
		cout<< "\nParameters succesfully changed\n";
		return main();}

	if(type==2)
		{
		cout<<"\n Precision: "<<prec;
		cout<<"\n Lattice constant: "<<constant;
		cout<<"\n Curve is currently: ";
		
			if(vplot ==1)
				{cout <<"plotted.";}
			if (vplot ==2)
				{cout<<"NOT plotted.";}
		cout<<" \n Multiplication factor for the growth time: "<<multfacttime<<"\n";
		return main();}

//***********************Substrate***************************

	if(type==1)
	{    int imax, maxtime;

        cout <<"\n Number of substrate atoms to be created (suggested 200): ";
        imax=ReadKey();
        
		cout<<" Ballistic deposition (1) or Random deposition (2): ";
			tdep=ReadKey();		
			if (tdep==1)
				{cout<<" First NN (1) or Second NN (2): ";
				NN=ReadKey();}
			 
		cout <<" Number of atoms to be deposited (x1000, suggested '25' for 25000): ";
			maxtime=ReadKey();
		cout<<" Relaxation? (1: no;  2: yes): ";
			rel=ReadKey();
		
		sprintf (mxtc, "%d", maxtime);
		
        sprintf(nameout, "%d", imax);
        
		if(tdep==1)
			{strcat(nameout, "_BD");}
		else
			{strcat(nameout, "_RD");}

		if(rel==2)
			{strcat(nameout, "r");}

		strcat(nameout, mxtc);
		strcpy(nameout2, nameout);
		strcpy(nameout3, nameout);
		strcat(nameout, "_z");
		strcat(nameout2, "_h");
		strcat(nameout3, "_w");
		strcat(nameout, datextension);
		strcat(nameout2, datextension);
		strcat(nameout3, datextension);
		ofstream outfile(nameout);
		ofstream outfile2(nameout2);
		ofstream outfile3(nameout3);
			
        double * Xi = new double[imax] ();
        double * Hi = new double[imax] ();
        double * h = new double[maxtime*multfacttime] ();
        double * w = new double[maxtime*multfacttime] ();
        
#ifdef OPENMP
#pragma omp parallel for shared(Xi, Hi, imax)
#endif
		for(int t=1; t<(maxtime*multfacttime); t++)
			{
			//Define the x-coordinate of the random particle at time t			
			int u = random(imax);
			Xi[i]=Per(u, constant);
            
			deposition(u, Xi, Hi);

            relaxation(u, imax, Xi, Hi);
		
			//Averaged height at time t
            h[t]= AvHeight(imax, Hi);
				
			//Averaged height at time t
			w[t]=IntWidth(imax, h[t], Hi);
			}
        
        for(int t=1; t<(maxtime*multfacttime); t++)
        {   outfile<<Xp<<"\t"<<Hp<<"\n";
            outfile2<<t<<"\t"<<h[t]<<"\n";
            outfile3<<t<<"\t"<<w[t]<<"\n";
        }
        
        delete [] Xi;
        delete [] Hi;
        delete [] h;
        delete [] w;
        
		outfile.close();
		outfile2.close();
		outfile3.close();
		
        IniPlot(nameout, "X [arb. units]", "Z [arb. units]", 13);
		IniPlot(nameout2, "time [arb. units]", "average Z [arb. units]", 0);
        IniPlot(nameout3, "time [arb. units]", "interface width w [arb. units]", 0);
        
        if(vplot==1)
        {
            SavePlot(nameout);
            SavePlot(nameout2);
            SavePlot(nameout3);
		}
    }


return main();
}




//subroutines

//***************************************

double Per(int n, double s)
	{
	double a;
	a=n*s;

		return a;
	}

int random(int MaxValue)
	{int a;
	double b;

	b= ((double)rand() / ((double)(RAND_MAX)+(double)(1))) *MaxValue;
	a= (int) b;	
	return a;
	}

double AvHeight(int steps, double * Hi)
	{
	double a=0.0;
	for (int i=0; i<steps; i++)
		{a+=Hi[i];}
	return a/steps;
	}

double IntWidth(int steps, double av, double * Hi)
	{
	double a=0.0;

	for (int i=0; i<steps; i++)
		{a+=pow((Hi[i]-av),2);}
	return sqrt(a/steps);
	}

void deposition(int i, double * Xi, double * Hi)
	{
	if(tdep==1)			//Ballistic deposition	
		{if (NN==1)
			{Hi[i]=Maxim(Hi[i-1], Maxim((Hi[i])+1, Hi[i+1]));}  //First NN
		if (NN==2)
			{Hi[i]=Maxim(Hi[i-1]+1, Maxim((Hi[i])+1, Hi[i+1]+1));}  //Second NN
		}
	else				//Random deposition	
        { Hi[i] ++ ; }
       Xp=Xi[i];
       Hp=Hi[i];
        }

void relaxation(int i, int imax, double * Xi, double * Hi)
	{
	if(rel==2)			
		{for (int j=-1; j<=1; j+=2)
			{
			if((i+j)<0 ||(i+j)>=imax)
				{break;}

			if(Hi[i+j] < Hi[i]-1)
				{Hi[i+j]++;
				Hi[i]--;
				Xp=Xi[i+j];
				Hp=Hi[i+j]; }
			else
				{}
			}
		}
	else
		{}
	}



//**************** Plotting Routines ************************************

void IniPlot(char* name, string xlabel, string ylabel, int pt)
{
	if(vplot==1)	
	{   command = popen("gnuplot","w");
		fprintf(command,"set mouse\n");
		fprintf(command,"set autoscale\n");
		fprintf(command,"set grid\n");
		fprintf(command,"set xlabel '%s'\n", xlabel.c_str());
		fprintf(command,"set ylabel '%s'\n", ylabel.c_str());
		fprintf(command,"set arrow 1\n");
		fprintf(command,"set title 'Use the right button of your mouse to zoom. Type p to unzoom. Middle click to lock cursor.'\n");
		fprintf(command,"plot \"%s\" w p pt %d  \n", name, pt);
		fflush(command);
	}
	else
	{}
}


void Plot(char* name)
{	
	if(vplot==1)	
	{
	fprintf(command,"plot \"%s\" with points\n", name);
	fflush(command);
	}
	else
	{}
}


void SavePlot(char *name)
{	
	cout<<" \n Save the plot (PostScript) from the '"<<name<<"' file? \n (1: YES, 0: NO) ";
	if(ReadKey()==1)
		{
		fprintf(command,"set term postscript eps enhanced color\n");
      		//fprintf(command,"set title '%s   (%d peak-fit)' \n", name, npeaks);
		fprintf(command,"set out '%s.ps' \n", name);
		fflush(command);
		fprintf(command,"plot \"%s\" w p pt 13 \n", name);
		fflush(command);
		}
}


//************** I/O Routines **************************************

int ReadKey()
{	char tkc[10];
	int tk;
	cin>>tkc;
		
	tk=(int) atof(tkc);
	if(tk<0)
		{return 10;}
	else
		{}

	return tk;
}

float ReadKeyF()
{	char tkc[10];
	float tk;
	cin>>tkc;
		
	tk= (float) atof(tkc);
	if(tk<0)
		{return 10;}
	else
		{}

	return tk;
}

///////////////////////////////////////////////////
//********* Testing only *************************
/////////////////////////////////////////////////


