function [vc,k,semilunghezza]=bisezione(a,b,tol,f)
% [vc,k,semilunghezza]=bisezione(a,b,tol,maxit,f)
%----------------------------------------------------------------%
% Implementazione del Metodo di bisezione
%----------------------------------------------------------------%
% PARAMETRI
%----------------------------------------------------------------%
% IN OUTPUT:
% vc: vettore contenente la successione di punti medi
% K: numero di iterazioni effettuate
% semilunghezza: vettore contenente la successione di  semilunghezze
%----------------------------------------------------------------%
% IN INPUT:
% a,b : estremi dell'intervallo iniziale contente la radice cercata
% tol: tolleranza per il test di arresto basato sulla semilunghezza
% maxit: massimo numero di iterazioni permesse
% f: funzione di cui si cerca lo zero in [a,b]
%----------------------------------------------------------------%
vc=[];
semilunghezza=[];

%-------------------------%
% AGGIUSTA ERRORI UTENTE
%-------------------------%
if b < a
    s=b; b=a; a=s;
end

fa=feval(f,a); fb=feval(f,b);
if fa*fb > 0
   k=0;
   disp('L''intervallo iniziale non e'' accettabile')
   return
end

%----------------------------------------------------------------%
% CONTROLLA CHE LA RADICE SIA UNO DEI DUE ESTREMI DELL'INTERVALLO
%----------------------------------------------------------------%
if fa == 0
    vc=a; k=0; semilunghezza=(b-a)/2; 
    return;
end

if fb == 0
   vc=b; k=0; semilunghezza=(b-a)/2; 
   return;
end

%-------------------------%
% PARTE MODIFICATO: 
% Trovare il numero minimo di
% iterazioni necessari per
% garantire un errore in 
% valore assoluto inferiore
% a una tolleranza tol
%-------------------------%

maxit = ceil(log2((b-a)/tol)) - 1;

%-------------------------%
% INIZIA IL CICLO ITERATIVO
%-------------------------%
for index=1:maxit
%  calcola nuovo punto medio e il valore di f in esso
   c=(a+b)/2; fc=feval(f,c);
%  calcola semilunghezza del nuovo intervallo
   semilun=(b-a)/2; 
   vc=[vc;c];      
   semilunghezza=[semilunghezza;semilun];      
%  effettua TEST DI ARRESTO
   if (semilun < tol) | (fc == 0)
      k=index; fprintf('\n'); return;
   end
% calcolare il nuovo intervallo di lunghezza dimezzata
   if sign(fc) == sign(fa)
        a=c; fa=fc;  
            else
        b=c; fb=fc;  
   end
end
k=maxit; fprintf('\n');