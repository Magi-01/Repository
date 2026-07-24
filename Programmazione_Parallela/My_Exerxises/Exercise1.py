class A:
    def g ( self , x ) :
        self.f( x + 1)
    def f ( self , x ) :
        print ( f" A {x} " )
class B(A) :
    def g ( self , x ) :
        self.f( x + 2)
class C(B) :
    def f ( self , x ) :
        print ( f"C {x} " )
class D(C) :
    def g( self , x ) :
        super( ).g ( x + 1)
x = D( )
x . g ( 1 )