fn main(){let (a,b,c,d)=(-2.8f64,1.0f64,-0.1f64,-0.1f64);let (mut x,mut y)=(7.0f64,0.0f64);
for i in 0..1256{let xn=x+a*x/(1.0+y*y)+b;let yn=y+c*x+d;x=xn;y=yn;if i>=1000{println!("{:016x} {:016x}",x.to_bits(),y.to_bits());}}}
