let a=-2.8,b=1.0,c=-0.1,d=-0.1,x=7.0,y=0.0;const f=()=>{const xn=x+a*x/(1+y*y)+b,yn=y+c*x+d;x=xn;y=yn;};
for(let i=0;i<1000;i++)f();const buf=new DataView(new ArrayBuffer(8));const h=v=>{buf.setFloat64(0,v);return buf.getBigUint64(0).toString(16).padStart(16,'0');};
let o="";for(let i=0;i<256;i++){f();o+=h(x)+" "+h(y)+"\n";}process.stdout.write(o);
