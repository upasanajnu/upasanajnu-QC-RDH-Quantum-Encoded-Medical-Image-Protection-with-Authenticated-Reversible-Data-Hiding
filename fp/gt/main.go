package main
import ("fmt";"math")
func main(){a,b,c,d:=-2.8,1.0,-0.1,-0.1;x,y:=7.0,0.0
for i:=0;i<1256;i++{xn:=x+a*x/(1+y*y)+b;yn:=y+c*x+d;x,y=xn,yn;if i>=1000{fmt.Printf("%016x %016x\n",math.Float64bits(x),math.Float64bits(y))}}}
