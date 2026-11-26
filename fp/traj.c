/* Same trajectory as core.chaos_digest: prints the 256 post-transient states as hex of IEEE-754 doubles. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
int main(void){
  volatile double a=-2.8,b=1.0,c=-0.1,d=-0.1;   /* volatile: no constant folding of the parameters */
  double x=7.0,y=0.0;
  for(int i=0;i<1000;i++){double xn=x+a*x/(1+y*y)+b, yn=y+c*x+d; x=xn; y=yn;}
  for(int i=0;i<256;i++){double xn=x+a*x/(1+y*y)+b, yn=y+c*x+d; x=xn; y=yn;
    uint64_t u,v; memcpy(&u,&x,8); memcpy(&v,&y,8); printf("%016llx %016llx\n",(unsigned long long)u,(unsigned long long)v);}
  return 0;}
