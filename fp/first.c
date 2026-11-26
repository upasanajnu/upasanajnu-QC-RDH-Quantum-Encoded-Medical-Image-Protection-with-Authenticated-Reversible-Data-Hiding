#include <stdio.h>
int main(void){ volatile double a=-2.8,b=1.0,c=-0.1,d=-0.1; double x=7,y=0;
 for(int i=0;i<1256;i++){double xn=x+a*x/(1+y*y)+b,yn=y+c*x+d;x=xn;y=yn; printf("%d %.17g %.17g\n",i+1,x,y);} }
