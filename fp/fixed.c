/* Integer-only chaos trajectory (Q32.32, floor division), as core.chaos_digest_fixed; prints the 256 states. */
#include <stdio.h>
#include <stdint.h>
typedef __int128 i128;
static i128 fdiv(i128 a, i128 b){ i128 q=a/b; if((a%b!=0)&&((a<0)!=(b<0))) q--; return q; }
int main(void){ const i128 one=(i128)1<<32;
  int64_t a=(int64_t)(-2.8*4294967296.0 - 0.5), b=1LL<<32, c=(int64_t)(-0.1*4294967296.0 - 0.5), d=c; /* rounded as Python round() */
  int64_t x=7LL<<32, y=0;
  for(int i=0;i<1256;i++){ i128 den=one+fdiv((i128)y*y,one);
    i128 xn=(i128)x+fdiv(fdiv((i128)a*x,one)*one,den)+b, yn=(i128)y+fdiv((i128)c*x,one)+d; x=(int64_t)xn; y=(int64_t)yn;
    if(i>=1000) printf("%lld %lld\n",(long long)x,(long long)y);} return 0;}
