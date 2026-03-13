(* ::Package:: *)

(* ::Input::Initialization:: *)
prec=15;


(* ::Input::Initialization:: *)
\[Chi][{L1_,L2_},r_][k_]:=Module[{},
If[r==1,
Return[HeavisideTheta[k-1/2]-HeavisideTheta[k-L1-1/2],Module];
];
If[r==2,
Return[HeavisideTheta[k-L1-1/2]-HeavisideTheta[k-L1-L2-1/2],Module];
];
];


(* ::Input::Initialization:: *)
J[{L1_,L2_},{p1_,p2_}][k_]:=Outer[Times, {(L2 p1- L1 p2)/(L1+L2)}//Flatten,(\[Chi][{L1,L2},1][k]/L1-\[Chi][{L1,L2},2][k]/L2)]


(* ::Input::Initialization:: *)
RedJ[{L1_,L2_},{p1_,p2_}][k_]:=Outer[Times, {(L2 p1- L1 p2)/(L1+L2)}//Flatten,{(\[Chi][{L1,L2},1][k]/L1-\[Chi][{L1,L2},2][k]/L2+1/L2)}//Flatten]


(* ::Input::Initialization:: *)
Jvector[{L1_,L2_},{p1_,p2_}]:=J[{L1,L2},{p1,p2}][Range[1,L1+L2]]


(* ::Input::Initialization:: *)
RedJvector[{L1_,L2_},{p1_,p2_}]:=RedJ[{L1,L2},{p1,p2}][Range[1,L1+L2-1]]


(* ::Input::Initialization:: *)
Mkernelfunc[L_,x_]:=-Sin[Pi/L](*+(-1)^x(Cos[Pi/L]-Exp[2Pi I *x/L])*)/(Cos[Pi/L]-Cos[2Pi x/L])


(* ::Input::Initialization:: *)
MMatrix[{L1_,L2_}]:=Module[{Mkernel,Mkernel1,Mkernel2,k1,k2,M,L=L1+L2},
Mkernel=Table[Mkernelfunc[L,x],{x,0,L-1}];
Mkernel1=Table[Mkernelfunc[L1,x],{x,0,L1-1}];
Mkernel2=Table[Mkernelfunc[L2,x],{x,0,L2-1}];
k1=Table[k*\[Chi][{L1,L2},1][k],{k,1,L}];
k2=Table[(k-L1)*\[Chi][{L1,L2},2][k],{k,1,L}];

M=Table[
(*multiply by Pi to kill exponential scaling of det*)
Pi(1/(Pi*L) Mkernel[[Mod[k-l,L]+1]]+(\[Chi][{L1,L2},1][k]\[Chi][{L1,L2},1][l])/(Pi*L1) Mkernel1[[Mod[k1[[k]]-k1[[l]],L1]+1]]+(\[Chi][{L1,L2},2][k]\[Chi][{L1,L2},2][l])/(Pi*L2) Mkernel2[[Mod[k2[[k]]-k2[[l]],L2]+1]])
,{k,1,L},
{l,1,L}];
Return[M,Module];
];


(* ::Input::Initialization:: *)
MMatrixN[{L1_,L2_}]:=Module[{Mkernel,Mkernel1,Mkernel2,k1,k2,M,L=L1+L2},
Mkernel=Table[N[Mkernelfunc[L,x],prec],{x,0,L-1}];
Mkernel1=Table[N[Mkernelfunc[L1,x],prec],{x,0,L1-1}];
Mkernel2=Table[N[Mkernelfunc[L2,x],prec],{x,0,L2-1}];
k1=Table[k*\[Chi][{L1,L2},1][k],{k,1,L}];
k2=Table[(k-L1)*\[Chi][{L1,L2},2][k],{k,1,L}];

M=Table[
(*multiply by Pi to kill exponential scaling of det*)
Pi*N[1/(Pi*L) Mkernel[[Mod[k-l,L]+1]]+(\[Chi][{L1,L2},1][k]\[Chi][{L1,L2},1][l])/(Pi*L1) Mkernel1[[Mod[k1[[k]]-k1[[l]],L1]+1]]+(\[Chi][{L1,L2},2][k]\[Chi][{L1,L2},2][l])/(Pi*L2) Mkernel2[[Mod[k2[[k]]-k2[[l]],L2]+1]],prec]
,{k,1,L},
{l,1,L}];
Return[Developer`ToPackedArray[M,Real],Module];
];


(* ::Input::Initialization:: *)
RedMMatrixN[{L1_,L2_}]:=Module[{Mkernel,Mkernel1,Mkernel2,LastRow,k1,k2,M,L=L1+L2},
Mkernel=Table[N[Mkernelfunc[L,x],prec],{x,0,L-1}];
Mkernel1=Table[N[Mkernelfunc[L1,x],prec],{x,0,L1-1}];
Mkernel2=Table[N[Mkernelfunc[L2,x],prec],{x,0,L2-1}];
k1=Table[k*\[Chi][{L1,L2},1][k],{k,1,L}];
k2=Table[(k-L1)*\[Chi][{L1,L2},2][k],{k,1,L}];
LastRow=Table[Pi*N[1/(Pi*L) Mkernel[[Mod[k-L,L]+1]]+(\[Chi][{L1,L2},1][k]\[Chi][{L1,L2},1][L])/(Pi*L1) Mkernel1[[Mod[k1[[k]]-k1[[L]],L1]+1]]+(\[Chi][{L1,L2},2][k]\[Chi][{L1,L2},2][L])/(Pi*L2) Mkernel2[[Mod[k2[[k]]-k2[[L]],L2]+1]],prec],{k,1,L}];
M=Table[
(*multiply by Pi to kill exponential scaling of det*)
Pi*N[1/(Pi*L) Mkernel[[Mod[k-l,L]+1]]+(\[Chi][{L1,L2},1][k]\[Chi][{L1,L2},1][l])/(Pi*L1) Mkernel1[[Mod[k1[[k]]-k1[[l]],L1]+1]]+(\[Chi][{L1,L2},2][k]\[Chi][{L1,L2},2][l])/(Pi*L2) Mkernel2[[Mod[k2[[k]]-k2[[l]],L2]+1]],prec]-LastRow[[k]]-LastRow[[l]]+LastRow[[L]]
,{k,1,L-1},
{l,1,L-1}];
Return[Developer`ToPackedArray[M,Real],Module];
];


(* ::Input::Initialization:: *)
(* Fast circulant builder: given kernel vector of length n, build n\[Times]n circulant matrix *)
circulantMatrix[kernel_,n_]:=Table[kernel[[Mod[k-l,n]+1]],{k,n},{l,n}]



(* ::Input::Initialization:: *)
(* Optimized MMatrixN: build as sum of three circulant blocks *)
MMatrixNFast[{L1_,L2_}]:=Module[{Mkernel,Mkernel1,Mkernel2,M,L=L1+L2},
Mkernel=Table[N[Mkernelfunc[L,x],prec],{x,0,L-1}];
Mkernel1=Table[N[Mkernelfunc[L1,x],prec],{x,0,L1-1}];
Mkernel2=Table[N[Mkernelfunc[L2,x],prec],{x,0,L2-1}];

(* Full L x L circulant (1/L factor) *)
M=circulantMatrix[Mkernel/L,L];
(* Add L1 x L1 circulant in top-left block *)
M[[1;;L1,1;;L1]]+=circulantMatrix[Mkernel1/L1,L1];
(* Add L2 x L2 circulant in bottom-right block *)
M[[L1+1;;L,L1+1;;L]]+=circulantMatrix[Mkernel2/L2,L2];
Return[Developer`ToPackedArray[M,Real],Module];
];



(* ::Input::Initialization:: *)
(* Optimized RedMMatrixN: build full matrix then reduce *)
RedMMatrixNFast[{L1_,L2_}]:=Module[{M,lastRow,corner,L=L1+L2},
M=MMatrixNFast[{L1,L2}];
lastRow=M[[L,1;;L-1]];
corner=M[[L,L]];
(* RedM[k,l] = M[k,l] - M[k,L] - M[L,l] + M[L,L] *)
Return[Developer`ToPackedArray[M[[1;;L-1,1;;L-1]]-ConstantArray[lastRow,L-1]-Transpose[ConstantArray[lastRow,L-1]]+corner,Real],Module];
];



(* ::Input::Initialization:: *)
RedJMJ[{L1_,L2_}]:=Module[{RedM,MInv,L=L1+L2},
RedM = Developer`ToPackedArray[RedMMatrixNFast[{L1,L2}],Real];
MInv = Inverse[RedM];
Return[1/L^2 Total[Total[MInv[[1;;L1,1;;L1]]]]]
]



(* ::Input::Initialization:: *)
RedJMJFast[{L1_,L2_}]:=Module[{RedM,A,B,Cblk,Y,S,x,L=L1+L2},
RedM=RedMMatrixNFast[{L1,L2}];
(* Block decomposition: A is L1xL1, Cblk is (L2-1)x(L2-1) *)
A=RedM[[1;;L1,1;;L1]];
B=RedM[[1;;L1,L1+1;;L-1]];
Cblk=Developer`ToPackedArray[RedM[[L1+1;;L-1,L1+1;;L-1]],Real];
(* Schur complement: (RedM^-1)_{11} = (A - B Cblk^-1 B^T)^-1 *)
Y=LinearSolve[Cblk,Developer`ToPackedArray[Transpose[B],Real]]; (* Cblk^-1 B^T *)
S=Developer`ToPackedArray[A-B . Y];
(* ones^T S^-1 ones = Total[x] where S x = ones *)
x=LinearSolve[S,Developer`ToPackedArray[ConstantArray[1.0,L1]],Real];
Return[1/L^2 Total[x],Module];
]


(* ::Input::Initialization:: *)
(*DO NOT USE NOT WORKING YET, Need the _{12} block too*)
RedvMJFast[{L1_,L2_},v_]:=Module[{RedM,A,B,Cblk,Y,S,x,L=L1+L2},
RedM=RedMMatrixNFast[{L1,L2}];
(* Block decomposition: A is L1xL1, Cblk is (L2-1)x(L2-1) *)
A=RedM[[1;;L1,1;;L1]];
B=RedM[[1;;L1,L1+1;;L-1]];
Cblk=Developer`ToPackedArray[RedM[[L1+1;;L-1,L1+1;;L-1]],Real];
(* Schur complement: (RedM^-1)_{11} = (A - B Cblk^-1 B^T)^-1 *)
Y=LinearSolve[Cblk,Developer`ToPackedArray[Transpose[B],Real]]; (* Cblk^-1 B^T *)
S=Developer`ToPackedArray[A-B . Y,Real];
(* ones^T S^-1 ones = Total[x] where S x = ones *)
x=LinearSolve[S,Developer`ToPackedArray[ConstantArray[1.0,L1],Real]];
Return[1/L^2 v . x,Module];
]


(* ::Input::Initialization:: *)
RedvMJ[{L1_,L2_},v_]:=Module[{RedMInv,x,L=L1+L2},
RedMInv=Inverse[Developer`ToPackedArray[RedMMatrixNFast[{L1,L2}],Real]];
x=Total[#[[1;;L1]]]&/@RedMInv;
Return[1/L^2 v . x,Module];
]
