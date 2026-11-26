public class T{public static void main(String[] s){double a=-2.8,b=1.0,c=-0.1,d=-0.1,x=7,y=0;StringBuilder o=new StringBuilder();
for(int i=0;i<1256;i++){double xn=x+a*x/(1+y*y)+b,yn=y+c*x+d;x=xn;y=yn;if(i>=1000)o.append(String.format("%016x %016x%n",Double.doubleToRawLongBits(x),Double.doubleToRawLongBits(y)));}
System.out.print(o);}}
