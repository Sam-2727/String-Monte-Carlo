from __future__ import annotations

"""Stored genus-3 ribbon graphs and a compact-boson T-duality scan.

Usage reminders:
  Full default scan over all 818 stored topologies:
    ./.venv/bin/python "covariant formalism/python/genus3_t_duality.py"

  Quick check of one stored topology:
    ./.venv/bin/python "covariant formalism/python/genus3_t_duality.py" --topology 541

  Full scan with custom edge lengths:
    ./.venv/bin/python "covariant formalism/python/genus3_t_duality.py" --lengths 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134

  Quick check of one topology with custom edge lengths:
    ./.venv/bin/python "covariant formalism/python/genus3_t_duality.py" --topology 541 --lengths 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134

The check performed is:
  R^3 Theta(4 i R^2 T') - R^(-3) Theta(4 i R^(-2) T').
"""

import argparse
import base64
import json
import math
import os
import sys
import time
import zlib
from functools import lru_cache
from typing import Sequence

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import compact_partition as cp

GENUS3_GRAPH_COUNT = 818
GENUS3_DEFAULT_EDGE_LENGTHS = tuple(100 + i for i in range(15))
GENUS3_DEFAULT_R_VALUES = tuple(round(1.0 + 0.1 * i, 1) for i in range(1, 11))
GENUS3_DEFAULT_THETA_CUTOFF = 4

_GENUS3_F1_GRAPH_DATA_B85 = (
    "c-qyyOOEVF4yC=98pj~?a!I|FLJfNhKYpOmd+%_`<uW7BA*lqq$eu)H){TrC9}izfd-"
    "%Wq*T4Sfzy8O+|2_Q7|M~g*`Z~UDe_zL+>G--FU)R5{)A2pi&;IvyKE87~{{Q7?`2G8z*YOAI`7hh?FWdWH{`2#%|L^aw`1>1v{{"
    "Gva*WdsC{XKvG^#0SI<>&8*|Ga<1-"
    "~asmoqvD&`up?0|Nifv|Ni!$_YZvkLjU^zj~}wW|FiM$Plvxh{riXE@2CC!*Tdg;zTV&Y_fNxg{3Yw(H@$!0-*;?(|9bote}DA%-"
    "P3>m_fIyr0o$0w<+f2ft0ZxID&KbDdPg*FjZ;DIEey-`CK^-J3VN&f#-4r{=OmBC_uqG6S3jKL_?up(;`<9{m>>|}4Vi6(-1zo9-"
    "PD_F3|zg4-oDDf1SGy4jj<6DxUH%^9KZD$2}b?iy`uf*+^=vbQomKzHeq2t-"
    "$BuKQM2>8tjb?r&jkdx(C>8V132IXb2@h`XlsIJ+(OLW2vxZ6%r!Z&PUE{xT3+{7q+f&F6qdv{;kHoo<9m3ZacdyS{Y+qCrabPD#"
    "t!!bf%CaZJJu)mXkOEun9S_oUZzSu*C`w{Z~<<ezi|&&A)hNY%L%Ob?x=h>C=uU8UgZx~d~<1`PvminY(1GNzFV?3>k^r%w|gmqv"
    "%FfkW^#+fJ?U&8;MQroH$p*fx!|;wxAg8;o4)p?^d=^7xJj9YM#$&ZX?sGs-"
    "t=>rv!&XfR9D5%aY(BK?{UL;Mk&4!Hz?CE(5avE8cibown4VuG^By@RBtPuiS}uI>fL|_CoiB+y*tlmlb2%L_T(kX9hkhN_vf%V1"
    "O1%LYmv?j_jHMQ4R&O96UVtZDsJOa^SQMjDyA*NC-"
    "WVS|6o59%+r3(?03zl#sd4HeA?o2b5uD`E5iK+3Mvn{8<Z!?*Wr#R?{bHgPvpLxS4FcC{AeE-ZheJLy_Dfrfi3euj2px?JOY#F=R"
    "01D2lbvWq>Zh{r*e94p`rO;>W#VSYJ94r_pB$Y=3Ldw=gp>nV7^`=pTm<6a<1y-"
    "b9V9p58LzooUO&BFW7!=t;T^z)ZkVjjU2$`M)6FKzm%At`z>7W=XiepS#fx6+?M3$_WYc5bCjFS&yf*|aC<eo4T#y)d+;T0;v(D6"
    "y_y~34pnT2dpU^L<LlmVugBNzAYL2p<x4-GsLSyOuj5-"
    "^7k$gU9KO6=8Rj5<BfduhXP#RR;J@GafZtB0H|O5YnOdzc<TGWxf7+SqKJDyq82U_Y^ZQ$m7?>(?Bih>-"
    "ac)X`8zhO_*0{Bbp|>Y+Z<Z2zBaJbwp*I%TqD|LZ`#JRP+c052XU^T2C*99om2K_@)EBt|cPNd|r%5&z-"
    "*YgBpT?Lk2g!_YBaJIs8sD>-!!d1ge1Dw!%jjlzs<%K4WcYNwA-ATr^`_jIwu0Vm8fV^Z<+WMoaLc*N<Ta)D_z$?D-"
    "caOh?j9*r%_X;Fc~s9UaXZyK${nibb-3MVk8ua%y%@I@?GL#9c+cd9@#_V>;3kn93;R@VBbHGDz}!wClPpkVJ=S+<ZSVGM-Xa>bw"
    "s(6~S$|us;kkniT5j$Tk~z2@b2YBE9_uo=qkeo{;(9<9_pWD$av?^VxHhj%U~DK?VfOejkDKIkWCPA^<a1&Mz<ur;EZ^OOi~B|P4"
    "fbZ=VEY2E4e-yucd#$`-XHkv9h{V}ev5xUjj<fN2It<wCiVqCCvh)9KbwPN*RSlCS%%~1Hv0=BI=A<KpSweBxI2s3-"
    "2Lf=4twd1QivPs-BW7ygL*4Q<N%fQR@_4UnBE9mJ3zADB${>D8}ZFU83lfNd@D|A3x}IvDsz<L26{K%tsUQr?^f@rb8hJTQ+j-"
    ">DCPlA;~U1X8T8fS`{T<u=1JYUzyrNSGk47t8#kzy*EV2;Ft>yRc1Y!R^@Xm_{oIn5xPNv&$NI*&L-H2mUS--"
    "^?ADd$HIcCfTExC4NUz_tScAe&>$h2}VR6r{-"
    ">O`z1n<{xRI5>vuT;mdyu=J9H_+BFL_YVm3>oMG+^b`E68r5loGNc(>6RPceE@SS82S9m6SYKo-vSre9eYS`yuKFXh4dzf&?Z!G3"
    "y_$mfZn8hH|cH*$2Tb7T{f@KpKCD35=Nhjno`pmRpPt#bP(&Y_?DiI<GoOP>zGD%@ZuZAGc!dJx9YydpfNB<2938IHP)LW!;ONW&"
    "F7Q1Z?NNSf^M+9lPO#QZa6sR>SJy;H?|$17PqZ_oZEMRu~K}~1^MtKzRe|h;yTn@Y=iar;KessY2!6|32q7I=+FFIK?Pi(8n>nY6"
    "K>B1Qr6#N+jq}x{toX0chS8y&G>mptL8qisNQN@II!=psyEmeF1PPbtGBH_;>Ok6cYsRVp88SlhyxUgZ&JPS9GS+qQN6e9IWn)_V"
    "x$=2hO<;TQ7pu*!5k03eh#1lPEd{8@_;AYo)e_>EnN;%;T~Fl_qiIxd(HK?T5_q9?ykSVY6}$gQtNN1H!0_2{mt~|^}NLTo9InTI"
    "_S?0y{)329N!ztUqhGcjB{MZVYRj0<=?dX*NE@YIL&5qD?oajtq`|7t;Uo#*5Bx`nj!U2Wc{u57NbR%8|po&(PAy`z-"
    "cw;0!(lFuo`iL+`Pt4t3|kjht=wEQ$OdY*(A5^=k{^7O5EOQwh(vlI9sNlo6Bw3vXgvnEw)+PKIHZ-v_*jp4e49V4uv!|*?kKrTq"
    "5NCI)<y34*N>=n@`Mv8BA_6E+>YVv;AB;GCs#;*0FAE=%3>%QO68YdBBb4bB7Y0a?9;o=s`7gj?2*7KBl(nb6iFAR!eMlq|MSBt*"
    ")&@Z58VGhGQI8I#$0)g|^Fjf%q1b=x|#gz7-)m;Z`BOQB2EoTz-7xct+>AD#f?v)W_T0_{Na%klUJnkE)w6b-"
    "SV6$yE3LcQav{yS>n*KbBhL{<nSW`#z~?-Tw~U`o51T7UE`<(R<MSAPFGWyaz4(nK{0B4_Z4jcl|2&fnS_cn#18XD78t7-"
    "3Mll|MWhv;ru219AEss=S+Qy59D)}y>*_?zZa{7B{yW9d$B-OenWOCa$j)eReTBG7ku5*2UNv}{CwWcS%BW_<9o4CZJF)s-"
    "49iZ&uDx@y|ubU;=5gsSy!ie2QcnYdH1HY6BN8h#T@_fJu1ujtN1zc?S7>e@pJ6gda;&vpNQQlk)4Zizr?yxq8pbY_e(%4KIlX~;"
    "r@K*eK=~`cU16G^?tECW6%|t-"
    "XoM0?1OHZ`P^#FU@xcpHLoqB{aj|hh8ncNjCL!&g@@bGUMap;=Ta5FPh>V`q@_#0Ph>P`RD7m4YtKv%$kqE5xJb2m@$GV!sNa)ur"
    "=+jm=KT`uMv1K6W$u@#>P<#&biM|wHy^syn6LHK+n&CV^_Z&OfdNeVdTjI<kx@olZ(Cg84QYBy7w7TL<6C*S9q)DGdvz{V@%uzZ^"
    "@dk!O57)+)!P&w>y4|o>j9a1zd{$OHaorpoJH2(ddrjS3+{dT2lq=tw}|d-"
    "??U&9dQQDjIz{di2}rnCn$Jy}v+41DBGj4TZm!gQBA_gG0rP$dl9f3@&HE+H1)kn7G2Eb#-"
    "cVib1diTRTjU6J@)}BHBH0D<8qZ@g<7MBgnQ&~QAh7P$0AIFl*eqqvpRcT*Ef3D0i>AHUd<}HqcGK?Ovw&)&w!Pkc1EXcP;+yErn"
    ")V9uP0!^r7kGRw*K&jObGg=0T*{3It#i1=xIvLdmRKJ6Ir8fw^HlG1dYY#Vx;p>tb9&^v2C~{GpJy8E2jANh>ehLt$@tWg&JL;Eg"
    "6I|dI1{z+FRz3>`aqK!=M@brU~YU~(Y8bl?w)Jt>$WmWB)%uH#zvpcG-"
    "Yzn>doiK2e|-0mybO^_$Ze`pTpHTx=!%H&*AF2@V?eHS-wtQb3f)Gl-"
    "I%^^iU|iWk8F7kHt3%W)Sq1;@f!$J{aG@r{L?ohwb)jq4%(>^=sBU!LoRhe$9I)7_@Nt{aU180~bKte!WI+kW{}>!L|>vQN5L--"
    "Wj%62|s5An*}_7Cz#c1BkIQ*?<2En-pcR4k8BlmdB7>t&z*o~^_$dd9?TW$cc5UCaHrfOedV2p{8;Zj?7+Mw@E&%(d5dvGO+3Rz="
    "Pk2;KCOR*tHQj+uHM82G`Cm1nH!|7zelJmKgLG%28w#8*j@$vd?Bk)9`Nj)V73r7qkgRMJ~Cg8!}S<>AK6}zXRp7<ki_QtTUBp6m"
    "}{)R&FXE0JK-LUSKfKZkM-"
    "We4y?Ze?_t+lf75bJZ9uqLv9|0`fgAHNYb=5DOnl!!J)oPlA4*B-^SNq2)S`4B&@B<)cKbFEy~T@5f&MLd-ZF#Nw*04Lf!_L=t?*"
    ";AW!}4IZ3~*W@b~V`{w;gn;!Rlav!bE+4nHhfCcbq*^LY!6Zyd~3<}Eb7%_sQ6@f~`EujL*O;_djlH{2)da(vwm;@9!@Bz}GAUnl"
    "Apd{^AfCoa5y(Z65v@0WT1vh#7;=JU|p-|u(Qp7WwN&hf)O+}FX=@y)Ow$A88T_jaYR>GSB^aiHgL_oH)t+k1Wg|5F-"
    "wgGWu^a(8T&5$bT?uu9s~m8BcldY}H@^Bo46p)2I-"
    "ZOE%!f$Qz4tKA`4ZzeAA1Fp9vE%t?Ey@jwhAG3nbE!ZqOByrCmbuS>lHTF^59pKH~BYweKBc*dw*e=;S+!m_H_sQI>Ow+m!xUCv("
    "758-R4%R8=lv|?u^}K}MJ(;aO*IQ8e^4HQE8o}X)dJ{WT;3j#SC3@)RG-aE)KG)Cv6yD|hFt^U%GV$ke-"
    "z}Qgsnh<v>CZx~dWrrVtTNR)bh7i>iq@BK9m)(-"
    "L+|Bbe~$H5^tJkP){=Wxu=IxVT6Rd+yLWXI?Sc4~zK+AaPJF9~wyZ$(#sTeFqJ-XRg|bVBkLV4SC?BaOX@Xq6C75Fis5d|gnIl<m"
    "5X`iNqk0c8lwW2WSBo1oY5_mTeytm|Dt>ONs|~~R+RN+EwRy7Ro2nb+4oF*r+uInq)+=3~+OJDu+amL4(IWcPIcZ%2$K?W6O<yc-"
    "3l)rC3UUij(Zz?kzsc>w)V=(Y?&rXviR+tw4&9kLf2Q8Ey*9JfWBo$IS)El`k0EY%ZN~H_%37<lOuYq>zl8$wnq*mnn<$xi4Q+@3"
    "6L{QcGXxr;3U|*L*gb)Dtb+`1pA_S^cTb|+f$ftT+?K%Y>ILO<oUI2^&*t7QDsWTO-"
    "U%FTi&OhSW<KY;BW(~k`P|tY=|Z7iK3B`K2zO+2qzren`0x&HwR6S&{THKiJyYn|R=Y2F1VR3}y&dqv!+h2`E^;`}IM0`Uu1B`11"
    "3cG5xqD5!7F<ym(VNMu{J_%NQWy9_s@_o9$tdSQ?*Xf-wI4uCWO9oRjTph<28Eg0A)DLT=0@{1$-"
    "Tz*xY2wqnH#g2o8pq2V+Hobv-L&_FK#|J^0&VaTuATbVLr!tbIua^Ty1QNi38jRrh1crHuiyQ=xvaSEpkI|n-"
    "Xo4=jp9>fd^|JnCOjmw(U7nCA}3R+#L7xMu2pOJU71iZs;KP1+DmYcDnm@XnNmb?}i4sBRk!txD}SaA5Q&z&?(H_5wFDU?1u-"
    "rV>{ww>b@Y`>el;$EqAC7XH9&TpVNKdU9pQT@z_2vnw6tX@$7xUP_NGP#=@H2dbZxP-K$&uIpj7u3-"
    "sshzJa+NruNV9zCmO!3~i#Nb5m5gWoB_(oOshrbD#I#k6*n9e>LEXxcL4${`m{O_4r*Kdo{Ocmd~TVXEj@nd-"
    "1RPQymVv>AdFttL98gH%6S!`|syTt&8;BMf!fO!!dQ5$iBgUgS+LFHi&X}jKBsJxm(r-"
    "?!$_Gz|nhd!G5pJLFNX<^kz1AJ=5E=K`Qqq9T|prZxNt(_n|BYCkydO*d?EP#e#?YT*`?xFYD(@`;gwk&yn`!^d{U?``06GQ~N}{"
    "`#R-Sb6m$ng^Jbl<T^%5)zrKEZu<F1tCYy&BdmG3+<=u=ILb{}%{rGH-"
    "=ah7VT1h|Y|sR3oqi3LtW{F4ejP*g>NG_MJvaLc`E_i?@+ueb^+w7<7bvkVp#pF5I%bNieImI}g+VO3)Yd-cx<rKUuWC|749rEiH"
    "6$>F$BhArInw=HAwrXQevbHvSqiMj;L>ZriQZV3u$32Hr)BpoZI@;y_}%PN+a_(Xb9`I-"
    "GW#{i$IQX{HOgk0Bt5Tng}QCw<_B}dU6g6j#_g5+wXS?imw(W&^#xozeR@9U{h6j_{hHmKLFn{;%@H`%uLZ;{+OLV`476{7ehsu{"
    "u!Sp;*CZ8+^-z3EA{_0-"
    "<GXisTW>4Am8Tcc8;WW;ffwH{XQjC!t=N71Pfowa%e6`Z73PYpP*2DTQEtfy%ur*VYHGX$`qZGpCE{BpYohTI?$2F?3vm1U{P!Nh"
    "g#COC?eVvJ2&?FAlem3Wxs#;1qpV;*sgm9s-"
    "Xezk2F`{r*(OfjH`p|*u|3MU0js@5T7X+_jP+iw_YDJj10RQbhlzf!Hvf#RLZ&y`_;a=k)AcqF@QD5I+c1nf#3`+lueaO)72Bmjz"
    "18L?*fOoAH_#Xv!_%8+j4e_uuXWAZ&?fyyd2N=>TDxw`<aQ+vaL4qm#oaH(#Dvw~|Au=6V8~i({~K-"
    "rj&c;Ccatn`$f;YmLfjkLIr8?2nBHphsM0<W&|7X10rdlV!#yJC0%`TT@$S9p(#cSL$91f6?z=?UbM?&mni`cG8)f$Td+XO6ySZW"
    "Vs^RB!zXT+Ame;KP5|Z7M^O_+pa0P$v0I88fq_4+dC4;d};~UP%t!S?t-"
    "y7XYdKBNNc9)nd(k2DEffwH%X0iBoV%pXl&Y4(5m(aUeix?HV4|!I<Lj_xgefoNA4_U!kkM-"
    "O8?umfhd5h{)XFQBB=MEcaRZ#*js<&r_sNNAnWO4g<1lyA!@E%q4hG2glB>5f{ye393l?AyaC%Riyhg*wkcmU&uf_hGn&uu6?J>Y"
    "f(mQInv-OJpbcnRK%UG7uXC)?5<c`14?R-L`>@!^G8(XTDw+_f+Oy)&D(ZxFc=iakN<9k~!n{UMusjW|8_a~{)%-"
    "zWPyOxz^VgS=)ZZ6n*~<~13)c5XMhA*bsJxp{4Y?U{e>9!(H+Z&vKw{R|+19b9ezNNiDu+dq4)oxrGL*2!z^2L<#d$aB1Jf%nH2e"
    "0n!GpKrEPgKEy}x8cyZoTu0CnV-rY;`N(Ki`}6_{Z_?##hWR1{RV|v=kh+;rS~m!;Dj>EYapDxL(0j1?i@g#BJmwLeyk2ISK|h8t"
    "qtLE<ESRK$elBpCwcbfK{{*BG`q)mng<D&5_6_m3|M35na~NQe#TnxJd+_VaD?&EIuxxv+Z9QLIwr{;oZDOaPls%n_PXcxz<La94"
    "4daX@OljG4Hr7+VQzIBHZbRFFOaq^JIvXBuFkKGf%{{^&*AwsL|#B|vsiD={utmUrJ9`&5b-"
    "@R?$_~kSKOPT$Ir)px!=66i+0q#_#62|zt*d#b;Q2lay$8Qz4U*f4Oay_-"
    "<>_)raju!!$!mBIo)jT6ObJ53Ue>rYsw2UL=A3BTjUFn+f^1jL_W77rRp2p2-"
    "DO*k2?^~W1rJk#T^djUhnA>A=%tLDbd^GR+wDxd~Tqr9rcJC%NYp#Y;HqgnF9=NPhXu0^6PgiuzGYaT)#Vc`x;iZehb=L{73a$!M"
    "5sGs^6q$L;YzTvr0DcfqZVHEcOJ8o9W8jp+NmcH5=${*D+eMk=`}xHw$K=x692#S?XP<eiMC{XZ4$Fi?cw5`mJbj+&lFf!Sc}muX"
    "Epx%JA*a=5MWP4?2zCog(4q>)oZa1LJqE@~u;65PU50vCZ04zyxlaQ(6UbhcK0KlDV1QKxACU9KB~K@sM1{ASV-<aWhOq;(Tr%-"
    ">ukv3_OePUThHs1>)Nh+vJ9rOzbM$W_)+P-Ru7-"
    "zNKqR7s!rpz{&Kt;+tS{7f8~33s2I)I<%djJeZ~Du0wl0S2<;zpZ6`Ih_`NEuy3Kw$|?tsM&DvID!>|bxHX(%3dRk&JZq%Kw~l3Z"
    "zXZiMjO0rDB~nS)ozCWd35ajjo2p=+Yv@hWx3yoQ^fr@sk^K@Kxg)c~9P3l9-"
    "V`MFRPOD;&WV~k5WNF?CuQais|1~&_o<?3!y#R!PgUDNbOQ<KZEOXZn@Fj8Ls1PUF#2=5AqS3-"
    "+n=*3Qy<f_)u);>rr$oIH_*+e=-"
    "rdqc3^#~gbJ7<eZEF^#I(q~Z`t^67?!xgd=2RhMK<PZOz)XUTw%V3w!@^@&U|enTVX)#I`cIg-"
    "@Vx0e2vGq6uZuRjZc{@{aJj&8I$J%wc;Dv0oQLczNsm)>o+ahEBX*s7~e+0R<o88^&1}Fp~rp5ToE7LF;Dzt&sz*}=jVM3(H40`f"
    "%)1NKG|f<*Z4#Yo^7h6H|bVp^IXqr&Czx-XA<%!?}u*SWK;M&6Ve;#TI)O$9;9vPq|+SfTkKv1gnNnjCc!-V+>{cI-"
    "S!gmHF#`^9(029HF{*pJnWP_Uo)H2m6?RpZ;OuGr9wU@89dgf5<p^^?78Ch;3ONJEAlZSJIj`OZcpro;>%*)4E+T@f2`XMH7<&Yo"
    "s=c-oF>`gZ<;sv-W=cF@4B~I_vQ>~p(9|uZDEN!B<tPtd?Dla1>0G<Qn>1UL2%EYm)6?{R(j7w)=K-pQg3*3aEfpPL}-di+-"
    "*zRiXGYq2JtP$uD1`|>rHMBP7!WQsjX3oJGKkldx8_+_Aan=hot&VN_O<#T-%R>ie2;GoEoIzYjRDm%KTiEXmUlaTcf7eZ#7B-"
    "#oVjkV2nnpd4BzF&$sX!7wu`~(LT@gvaeipXO>?z@0avvkayOn`u9uv1IVbmE1@?PIdWd{rLJ5+iObzDu|$4wzr>SxD0E)2y>TJM"
    "_V1UV_(oz^ykEk_essUY6uZ*>5)j|3?A4_H!T8pDz{mxP-"
    "7m59e|Epbae)GQi!$cV*Pz~_hLxMI5pFtZbMMuJXKnfWp}xw&zQq^0P~SrCmyogB!}}%1<gHqN4i25aWTp$pw>Wa%Z7LKve-"
    "7gN#Sf*A58_(_5?fR{f3Dh=*FQRc4%?OjddE}_)$fqVb?Udh3T>l+)^C3eDk7oA`g`jpFSZp%`@F-;JAK|5;#KDSxh-"
    "<={CPy;q0o7zom(NY{swx_RN_kK&n>aT=g(cSE3LnEd`qzd>u(s}TI~Ag&u#tV=g$K!P$|Bt6~<@rZ5rZL<NUdC>fEW{-"
    "de0&{U+zn_ebk*cK+PhIjgY#-q=oAWPgnKdB6dR^=sBnnS1^`yjPaL{wC*B_Xq25<9sSC-zC=H-"
    "fApV#{z4y>UB&V&Wx?UaeQ}2Gpnt?VSH2d$4BvPJHV^X`%Hs)&Y^q2<xcL-`%F{2C-"
    "UZ=y4PFeDz((RR3AuS2%j55B4cE73t@@;w%$@zDt{5Zp|s`;O^RE=OyJ2(_(W3`y_IM4w>7EWvvz9#am`}yX=+m-"
    "?!bGRz#hrmBECsX>(`5n__jitJg!;kJxy%~!tK4M35=1?t(;m&%ih_v<6AvM<h-<5q;FaAqhxJ$5A-"
    "c;w_mq0Me8@ZdGfq|TX#=ty))RS^EIybfZDQeHV2Om)~Vm#n}ebH9e#AM+&hDPQePu_^LUm%Unss6B71W%RKLTI4yNl(PB={Rf~="
    "|cWSnnM)vii>j{!r>|D=9fcTZCH{fF2KE}&0UYq8)HF3Ei=Sa5@)PS@Y)9h2$a36gbvFSGup@3#m?FxTJK`z_iV3h1qRP`h9I*aY"
    "48Hy-41-tsBp=PP>LqxXe?pX0qD<o3CGvz`>b-9z-Y`cdQYo=Wfa7=5d!_n?EA_OAJQ!-kZ--9zJ>G^6(8J;mPZVXdpFH;ZS|zSf"
    "9uB{F;2#foq6K#TLXOS;~0KWx(PC!2atbT<t2vWoj8ejQ);?+2cb2h4atPRI95pMO5M=Hrj&^Iw+ZPhQPOp}$+{n|nFP-"
    "#dYC%JIDU`g?rd`J3T@`TO0@2K=3fw&SDH-~4;Sy&Y^n<CUzo^OMjIBm7>Ue|~<5`oXz=Kj{7!ecpQ-cLUp);30R5+8Lo7w<WOf1"
    "<vgXEFGc<H-"
    "uFV;BgaF<OA8<Acltnm;^rLo?!rAM*w*AXLHkVRwgjG?Pvx@Na99p$_H${85Z(`GI~4Q$_J)@$c_A<ir!pbX%dUu)D~H#0yok38M"
    "mQr8E%r#Rmr~M;8{M`_4*ElV)<Mk;*##?b4^IgdZM3uS=`jSA@Ndp9las9q5Kr*cBRkK+fLjh+xK}blD2uyPs(c`Tl*;<b4O5hno"
    "q7{XeIXXgf;|LvMxCQ@iOe=3nB_WKRb0@GQwj0mHGj<D%7{kb-DMB%TwOayMu~|UrKLff&e#a%#;aAa4Ri&z%xHbGMX`8!OyLL7Q"
    "Wzd(}0!^QHY!B4Fi67t~ZVO3iX=?G*F(`Z#$ro_$l1&dV^0^PkK04lnag39Ib9AeD*xm;BH{ed8)(RqSl$GvdLk$G81Bp^_V#|T)"
    "V}Y5<6`@7U1rh^V<cUbBpG@b%T_=CbY$4MpJLCt1)OSqPL~3VFH6&>FOCFW$w@vtb3q@(m7w3w8eXCMoBoC7g?9!ZmlyxfEzb!y%"
    "8#LH}jUeK@pRe2lJMuH36fSy7Lw_Wl_e6%v;cq1z03|-qN~o*RQh;ac3WRz9;5~zt4O}yvY9dNY`icKjB6}KX-HV<atX!Ft#S?w;"
    "PDLgB}*o_AtG3g^Jut@aZ3NBgr%OOP1Jy7O_^-"
    "ejYx|0+ws3+z^o+cp==J&^+^^&+W~}*R|npkL{h@9qM{~J^tnO;od(7_TFDx?&&zB_db<3ALsVA!`IKFd50Z_^L47<!<F~T-"
    "QVU~+jD!zjnCsx<+1xuV2?kozK1bh$C<$4wA+-;EopB%JmJ=;oh8a~W1Yu?ev5KXRON&FI-gsO`{vZzKj7{tk^b4-"
    "I(P#SFSt?IhJ!wnn}jiqeaLOZu$2LF^#(y~L_X0Q2e1?Vxq4Gt81F=H6O~20;%@tv&$;^+d_9ooFZgu^fBQBpU;6zxchA<L^Vz#x"
    "tC53#IC$63P0G^-"
    "XS+ImE`;9e&`(S9d)=G5Th_Bn{kACWq~UcmLfc3DZ0;7#w1LCjb9q)+3h3PfHslZW7I+PKeZ37EJk;B^LIQUm)r}P%#J3FT&JIQ6"
    "n@6>61I^7sI$(u7y`e|<<i6g-"
    "nI(8ty+bze^?qQ5w7k~FuUE@iKJP;}RL#@#`4zJS@?(MCGK_mM0Pm5yuucNE*`_Q#??n0*jV91VG>BWYO95^S3Cv-"
    "0b3|y8<oLdw+m3WQz8mNEL%mvjcMgE;5sGi=0n8+6elA_R3vXBN9tEghP;bm_YrllKef6{B8!WxxGR>;raMi^ZY9;FTz!9N2C^x6"
    "$CP}Pg0FRo)sAHJUFid{m;xsG9`n#*&fo3It{Y~SWmh$mQeA}hGMExElm}3M#zJc2puirp>Q*T_yu=-"
    "W>rrd`1OPt$PKSysgUo*OOs5hRkdHs5Bzs4JI*{dyd9XguLx9-(O-"
    "O{J)?R4uwy(9g48NG?dtbZqZbA_!YKBbP4X$#m+;QK^$*g`fG^7o14u7EXHoWFKb`CRS@?Dkx7J^zl>b%|DUb#_US-ayA$>s@L6-"
    "Lp;s<xRaCF$2=4>b;|F&ke7}aGy#SW2@LYSh_nmznOD~6;LsLwja95TSR6utFRt3HoDck#mQ@AlY0mAmi)X%o!JP0#o*>}hDl1~b"
    "F~@^tiLzMm0@a6&uc~^+nEs}-"
    "1cH@O<sRfjVBiA(fD>sT7mlQYdk%TZ?3VnLxJ^}*~P@yV{RW)VSlW1<7)v%MvY_i9M}B4dBVnxa~bX)&ai@ej!U4KhR8hEqxkfx9"
    ";UY?F+S9juXj}C$hn@7#Kq3_7zbL!_9&m9JI7hn25G7DEtIvjPxzb%X6?+ByAKRzO(5M(jZu)^VAeE~gxB0M%=L46u;YQBo5wmz`"
    "ni9)WA1E3_De#CJ33!a-zPFcI)2W>j^~2sJj^2wUHkqZpQ|Nz-"
    "@=zX@1R(N`8>i}ET7XY@37w9j&~)!C()?U;^p+`eVb+%uNrejWBIjUUpMl3V6(qkJ|8$_bl=|y=34vy8&?}F%lZ17icJ>gHP+wYz"
    "QMpx@QHU0?E3~2yJFBhjWF@|9sY~wk=4xtd<VffkBqMs&}#?<&z}!++UV&`4{LQJwQRlJbJzjy*g5R~7r1+$2R3*;I)@E9^8PePd"
    "Y=kT;NsJ|cArX)-"
    "_rBC3G+4K$Nrfn>F3CUgJ+u3{Jc>Uonh0PC_wL4M{<S@@1qRqdTTjd7cjUjQgWoTz<Nxryzm;$S&!jm7hR^Y&sE%;k5!zTcd7Q~0"
    "~J5-TD)4~_4~TT`l0XNwK(ZMZ~v`F5_<o7pBeZ0CfR&hWIvv~SdKTV?sv%M!+ZmLfNA|c-q9qu@i5)d%>G^<u9?c+U@8-"
    "Yx%cj=EoFgH+)Cok0YZ9fb+rdn)qCy8ke@ehVEOsna{)i^p7dEO`MEGbn0psb$_WdU;=cTxdbJJ!?&qdU!yu>--"
    "#u}Y<(3xSj&B7#wRx`biu=Qov;5pHTdhYBy_<KJXvvy-ugKT?g|!y2efTr(5UorC#q?fLrssLhOx-Zqr|0t(z>@Uta06PN;>+Z7k"
    "-d1LGxNDlU1qY&?OVQdZl})s)P^k5sh8;2T5h{j4|4mRdKGRWE%XD*ZRyH<p#ryp#o{~Mkd}Htsr8r^+K6wq9>Wn|B)&m^{&LI^>"
    "(60Vu1t~Iw+L~0z?=13hp-v+1^N~q&?5Gu+$xws_}7VVy$aJ739QG!Dh$-"
    "Pit918FtOts_vdh3QY*fF*zkJHr!BG`b5LI(1f2eyJ2eP{B<`j>kL|GS!5n#SOLI3z+PzuKn8^+2X|k>x^EEz91C_nXe61g#Iny?"
    "r&*kteFmKDx=O&+0KB~811KQWoyH9DbuDg1Rj7BSa5xsqtt@^zaBB*ku`mLrcL$@9UoqBLTbZXP~dTtPdEz*knq0j`&<NZ)-"
    "fpqQ`O=CaMyXC^<RrEF-TA1L<g4TM>Uw6u-"
    "Y7uS#D;K8`ZcK|7s1?^^1B!y>@p^1(6L=Q~zx?yZwu5!~eqzgbf?X$iPW;#Lb#LhJk6l~O|3H3j%li4OmGLBdW4=F|WjhX4@1Kuw"
    "d7WgxpHph*2Qjsu+cI6Jf?g-"
    "n_8l^emsyS<_dSn&eVpus&S9r=3rcN=7`H}*rbys!03EeE8+F`sLl;o{Jd=8xYAM65C~dtNH$p6=_eG^Ve7YmxGy>(mfslu<SND2"
    "bp_-qM&=4*+U<8(^#NC81i|@ubIKHPYc8v(18XHxKZyL-X@R{CLAmh-__w!kya-TQ31*z@yiGJP(?$|hc<Kj&yjI3{IVzxyb@A`e"
    "1r;Ud^Kj+*6RM;WTEg^v+^8B2nZK?ib^(`aM*+~Cfy&EZwa;?fctlyDp4QhN^9fL&+6gRVuk%EP*TYepb+yW9TSR&lb^%-"
    ";3lBze)Wi#{!$Ze-GS>;OQHQuu<-g<28%Cgb4jCJ}o!QGRhb|%EFL2*-"
    ")na}4oo12}gmCxHyc6DEJ{oE9GeOAY)s549h>G6GwD&{M4c@~_vfboi2przNbu2hS4XM`KBL)Egg8n?X)nMJg-b&0zKxtyQ~x9-"
    "pTZal(``g7Ha*W(Ug-WG}V=Nz&%OI}`+gC&vYB6DOtR8pC)-W=H<Br;1h+@Ud|X^cd@QC1r|w-4yevRbza-"
    "Q*U_lFhA2(Nx0aMx10cQNYhNxaIZO&mp(#_+v8n{B>urG^OiMyx25Wrxn(rJBVOA-"
    "m#zS;0;84a(vtIETccd9q409^r?Hh*9~`o`$|vu{pVI+yaOzEky86qW3L-"
    "+jiWx*+vjHc<N11T+6E(<sjjBppK}~bKRbTGk!po=9%OL55@Xoh91>Zi$Uc$W6aur>z&;V!5`y#A)Om}!#Xssjl?BDOx4%ED_9l~"
    "?hA|EMmF<_jtbGv}`ib$apbYwz#P=*|JoskHSfY;Yu^FSqjf$2vWQ=gXz9%$K`sw(8PWburwL7i39$)thSiVCx9l!6@iLK`W`ri-"
    "W_&IVpUcdYM-M#5}aZGX_P+z&PM~^>(U&ru2z50BOS=Q?@GJNj-zs=pzb>z1>0^B{>BFGzyEJe7#fI#Ec2f}(|g?lfkruSFg#uPD"
    "b$SG`*%H4&p8|*N?`?!rpdvbiM;9dFvC+0xhoXoaJQhX~}TgYfWN1FDG<?=bm;gH<S=Qx8i@)G$xl*5%J4883Pj%<;d&vm_48Q!n"
    "oxLWHhpRPB61g0>#F(0!=J?@6}!q;hjvAJ8$n|=qGJ~^Lb?v_&9AjmBlp&gbQ`Mhn>L>=qg-"
    "^%AL(=VuEZFVbB#}uYAM1UJpYHOr&^R>wE;r#kITbFF+O606q+c)b6)+IObbx(3p&w#(aV~6Pkn~!grKg3&eO%iZlPmqjXX_pW6@"
    "#{^;g~!i@_uu3B@AFev-sZXA0ssAgA1I$si9EN$5BZn#(&_oZV@EmfKHWplnbNsCP+^28+!9pUA&tAySDN5~-"
    "i^B21SR!m`Wi;y+(=#94k_GTJj*yhdi${q0wGy%70(3Jif<6h7HEn1w)Ae%Ta!G}TY?L-"
    "l+fD%jqB}#*3`SDR@z7Pme4@|dU^|up&<f#OMR&=(zsy~Cq<p*bC$$uRWFdwPi_Me*o1Jqh|n-"
    ";+@xJmC64=f>&T#6nRKBm=I1D<3!nG>oac1u_oe*2^=)E@fS<R1Z5txl&uQ_B=^ZLwwe*(UruIy4#O>*yq<2@Q!P0wZ+I5!KhQ3D"
    "Qh4)X_u>WS~wU*E8)meW&ba12Jt^elt=agH4^8Vc5#-"
    "IiIbM4Q}?=3$E&dfc(oS&mKCh}Zk4VSymVZS2L^%lv?Yh&ZtAh%g_Ce@$u!Bd$zlWxB?C$@~>7M(X@6s`MjX5V6Jy0U?*pVxR2uh"
    "F*zG}Zxc#`hTQT9fte#tVI$GRD?pgPtwOE(H!A^Ygj#Z8&@y?7n5xo=xXZKQ;OmRjaJR#_AX-Q*LcbtYd+-"
    "NV|9$Xnc=>i0&6KKbh-Lyf3J>!<s(zuupJb5blO;cc1!sDLid|40GcBF~)6B(*3al+zzVPG-TX9r`|W*^5*C6HlXR}(x;tWz!HAm"
    "aDyyPXEBT$?S^c8-"
    "yoO!s^4TIbeF?(bDp~LZBV^qx)zut131GLe4dKH9CH+!r#4fFT@Do=%~RKV5TSR_vh@ZrEJ1}W;`3AlD$G%3&ID#lufD5^%$dk!s"
    "UNl!n=`#+^ET6bZq1n#BbqvGiSrgYPx0n2aNdH3DS;Wxmc0&zi?Pn@4L_G|>^gqFpLe;eg)?ly*JB9GF-"
    "J;&?(G9!x&8`ycwW)p1DwCfe)$IdEAzbK;6pn5f^zL`A=cOzT%2PaFKrjNAL^d!(4)pVdyZ_3(%|R~&DWec8XLbsKOYa`^Yr~H&f"
    "QN3-gV)9QP<wj&*wQ%*W*8c=kCYPh5pZ(`r+sN9baG34`I$L@XK|H{k)Q}HlNe;y(xDTsE?QIpNsD6sq%0?y?j4l@-"
    "1me+$|xtfzRFZ5hK**ZZ4TzyuuI>?y<dNa`Q@K%<0^%n!fmiTL>A{KQF!;jV~)`joV;H@!d-NwF4Ab#Cx*dloU8W)!XDG{-"
    "40D)0Ug*WL_gVE0XT=@)|8!lJsj6mDlWor7&-!eBX1))|;{_{Y||MRG@!;9s9=h#lN;8-"
    "32<YV|QF%_}$0N{a&AExYViNU8Pc?9qd~~mC|utqi+f5D@}v)T1Z`Opk(f~k7>%?SV(aXJ&eaawT8;fYt2KpG_i4R<<;5@wYcZjz"
    "59ChlBJvGwXO4RS<z&k+eCjiWpYD0ZVb%L$V_vj*6+@(wIyQpyZ32mj6(HWDrtz0<9ok9Vt5xxF0x?BT$iX41{)c>Z-"
    "FI@G&42&mTujxSZ_mowYS%yjI^=DodTKO=XCM-rF-"
    "yM;qT%tix<J~xYfMj%{3qIS6nY^eyuQnJmdV4vVI@L_ggN*Ih@@4H+ufQ>+<jQ-cCy5mQ0ofu9&C&T7xq<Hu#un$>BBu*VgOjCKC"
    "v5k8Qq9(Hm&4#;s91Q$)ETr?f@_HwfO*_H(T_3|lwSr#mrZaXXaSEPy-"
    "22#vFp$mhh>Z?gpRIrH_#Fm>{|q|>&D<a2~J0DENT^GFtNEduj-G=~eL%r&?zR6Cys{kmvA?;U+ae(vXU<>xT!i{x`7i)&-"
    "hd~WA(Y!Jx3H1Ez$U-%2VU6*`boZB?<-"
    "ah=kx7;6pySDp(ugIPcGpv_$4?pM8h3NyZH}B5P$I^N{&jvL6`tK$9>vIa=@aLs@dhzop<Mr@rB6kNBuma*%JW)&J>y6ZH$X};5Y"
    "`vMZbHhC}1B#^>cY~=+5a$+@zzj9wyMqK~@VONqvqS}M$E}q=c=64h8n{ITKfe|KB(_%Zpr2o#;&J@&@sro2<i-"
    "44feKTE{2Wngd(`mrR!v`i?B_zqs0ZY8V{w5uSlouP*e7blcbC{?IS<uuo!CG{uTj4xmuCnwpTq2J7$b$dsZOwdfhs4to2s<&>IG"
    "}vmdQ;Kp+O+G&Bx4=%>9DoT_S7NCPZ)bu@(!?-&?adN!J_TQDZQ@5tw6-"
    "0^HZVKe#4!)}h@}(_EAmT8DNST!b+Vcq_g^3|k?e9^ZCXewTUOr#@@Sn`)ogrv|%nhx<WOUWXfo@@%DdC!folZRPi5KWC&e@=b0|"
    "iGn|qyRB5sL<Mv%-"
    ")W4jV{2cex^W6@UGAC8E3J;fX)zxzqd7926<Y&lK1V)@U+Ui63O~0++?(5PQ(PD4^uED)3vRky{QTIO?Y=~}ULPKOI_U8EWYxL3H"
    "#dK*bzhY`e!|?|qMN@Lq{oj(j6YNJj7uJOhd5pS*IUEkUV6TZc>xwF!QIKL#1H8$)z$hJ)O*u}g8yq7i+cim(gMTxUTiY=^-"
    ";)OXwQ8BoVc?H`tBa|=jwYG+b3-"
    "s1=DqUv5A5+GBV@4Ra50PWf>gbz5k9Nul)k+3<bCmA~c4<&H0!`D&%vP!TUV#<#Q{4%Uplx{h$8>{sg9*"
)


@lru_cache(maxsize=1)
def _decode_payload() -> tuple[dict, ...]:
    raw = zlib.decompress(base64.b85decode(_GENUS3_F1_GRAPH_DATA_B85.encode("ascii")))
    payload = tuple(json.loads(raw.decode("ascii")))
    if len(payload) != GENUS3_GRAPH_COUNT:
        raise ValueError(
            f"Expected {GENUS3_GRAPH_COUNT} stored genus-3 graphs, got {len(payload)}"
        )
    return payload


def _payload_to_graph_data(entry: dict) -> dict:
    vertex_sequence = tuple(int(v) for v in entry["v"])
    edge_sequence = tuple(int(e) for e in entry["x"])
    n_segments = len(edge_sequence)
    if len(vertex_sequence) != n_segments:
        raise ValueError("Stored genus-3 payload has inconsistent boundary data")

    boundary = tuple(
        (vertex_sequence[i], vertex_sequence[(i + 1) % n_segments], edge_sequence[i])
        for i in range(n_segments)
    )

    edge_positions: dict[int, list[int]] = {}
    for pos, edge in enumerate(edge_sequence, start=1):
        edge_positions.setdefault(edge, []).append(pos)

    sewing_pairs = []
    for edge, positions in sorted(edge_positions.items()):
        if len(positions) != 2:
            raise ValueError(
                f"Stored genus-3 payload has edge {edge} appearing {len(positions)} times"
            )
        sewing_pairs.append((edge, positions[0], positions[1]))

    edges = tuple(
        (idx + 1, int(endpoints[0]), int(endpoints[1]))
        for idx, endpoints in enumerate(entry["e"])
    )
    return {
        "edges": edges,
        "boundary": boundary,
        "sewing_pairs": tuple(sewing_pairs),
    }


def get_stored_genus3_graph(topology: int) -> dict:
    if not 1 <= topology <= GENUS3_GRAPH_COUNT:
        raise ValueError(
            f"topology must be in 1..{GENUS3_GRAPH_COUNT}, got {topology}"
        )
    return _payload_to_graph_data(_decode_payload()[topology - 1])


def iter_stored_genus3_graphs() -> tuple[dict, ...]:
    return tuple(_payload_to_graph_data(entry) for entry in _decode_payload())


@lru_cache(maxsize=None)
def theta_lattice_points(dim: int, N: int) -> np.ndarray:
    side = np.arange(-N, N + 1, dtype=float)
    grids = np.meshgrid(*([side] * dim), indexing="ij")
    return np.stack(grids, axis=-1).reshape(-1, dim)


def theta_quadratic_form(T_reduced: np.ndarray, N: int = GENUS3_DEFAULT_THETA_CUTOFF) -> np.ndarray:
    points = theta_lattice_points(T_reduced.shape[0], N)
    return np.einsum("ni,ij,nj->n", points, T_reduced, points, optimize=True)


def genus3_t_duality_row(quad: np.ndarray, R: float) -> dict:
    inv_R = 1.0 / R
    theta_R = float(np.sum(np.exp(-4.0 * math.pi * (R ** 2) * quad)))
    theta_inv = float(np.sum(np.exp(-4.0 * math.pi * (inv_R ** 2) * quad)))
    lhs = (R ** 3) * theta_R
    rhs = (inv_R ** 3) * theta_inv
    residual = lhs - rhs
    relative = abs(residual) / max(abs(lhs), abs(rhs))
    return {
        "R": float(R),
        "lhs": lhs,
        "rhs": rhs,
        "residual": residual,
        "relative_residual": relative,
    }


def scan_genus3_t_duality(
    edge_lengths: Sequence[int] = GENUS3_DEFAULT_EDGE_LENGTHS,
    r_values: Sequence[float] = GENUS3_DEFAULT_R_VALUES,
    N: int = GENUS3_DEFAULT_THETA_CUTOFF,
    topologies: Sequence[int] | None = None,
    progress: bool = False,
) -> dict:
    edge_lengths = tuple(int(x) for x in edge_lengths)
    if len(edge_lengths) != 15:
        raise ValueError(f"Expected 15 edge lengths, got {len(edge_lengths)}")

    if topologies is None:
        topology_list = list(range(1, GENUS3_GRAPH_COUNT + 1))
    else:
        topology_list = [int(t) for t in topologies]
        for topology in topology_list:
            if not 1 <= topology <= GENUS3_GRAPH_COUNT:
                raise ValueError(
                    f"topology must be in 1..{GENUS3_GRAPH_COUNT}, got {topology}"
                )

    r_values = tuple(float(r) for r in r_values)
    total_L = 2 * sum(edge_lengths)
    Mat = cp.direct_mat_n_fast(total_L)

    start = time.time()
    results = []
    worst_by_R = {r: None for r in r_values}
    overall_worst = None

    for count, topology in enumerate(topology_list, start=1):
        geom = cp.compact_boson_graph_geometry(
            edge_lengths,
            get_stored_genus3_graph(topology),
            Mat=Mat,
        )
        if geom["T_reduced"].shape != (6, 6):
            raise ValueError(
                f"Topology {topology} produced T_reduced shape {geom['T_reduced'].shape}, expected (6, 6)"
            )
        quad = theta_quadratic_form(geom["T_reduced"], N=N)
        rows = []
        for r in r_values:
            row = genus3_t_duality_row(quad, r)
            row["topology"] = topology
            rows.append(row)
            current = worst_by_R[r]
            if current is None or row["relative_residual"] > current["relative_residual"]:
                worst_by_R[r] = row
            if overall_worst is None or row["relative_residual"] > overall_worst["relative_residual"]:
                overall_worst = row
        results.append({"topology": topology, "rows": tuple(rows)})

        if progress and (count % 100 == 0 or count == len(topology_list)):
            elapsed = time.time() - start
            print(
                f"Processed {count:>3d} / {len(topology_list)} genus-3 topologies "
                f"in {elapsed:>7.2f} s"
            )

    return {
        "edge_lengths": edge_lengths,
        "total_L": total_L,
        "theta_cutoff": int(N),
        "r_values": r_values,
        "topology_count": len(topology_list),
        "results": tuple(results),
        "worst_by_R": tuple(worst_by_R[r] for r in r_values),
        "overall_worst": overall_worst,
        "elapsed_seconds": time.time() - start,
    }


def quick_check_genus3_topology(
    topology: int,
    edge_lengths: Sequence[int] = GENUS3_DEFAULT_EDGE_LENGTHS,
    r_values: Sequence[float] = GENUS3_DEFAULT_R_VALUES,
    N: int = GENUS3_DEFAULT_THETA_CUTOFF,
) -> dict:
    """Run the genus-3 duality scan for a single stored topology."""
    summary = scan_genus3_t_duality(
        edge_lengths=edge_lengths,
        r_values=r_values,
        N=N,
        topologies=[topology],
        progress=False,
    )
    return {
        "topology": int(topology),
        "rows": summary["results"][0]["rows"],
        "overall_worst": summary["overall_worst"],
        "edge_lengths": summary["edge_lengths"],
        "total_L": summary["total_L"],
        "theta_cutoff": summary["theta_cutoff"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check the genus-3 compact-boson T-duality relation."
    )
    parser.add_argument(
        "--topology",
        type=int,
        help="Run only one stored genus-3 topology for a quick check.",
    )
    parser.add_argument(
        "--lengths",
        nargs=15,
        type=int,
        metavar="L",
        help="Override the 15 edge lengths used in the genus-3 check.",
    )
    args = parser.parse_args()

    edge_lengths = (
        tuple(args.lengths)
        if args.lengths is not None
        else GENUS3_DEFAULT_EDGE_LENGTHS
    )

    print(f"Stored genus-3 one-face ribbon graphs: {GENUS3_GRAPH_COUNT}")
    print(f"edge lengths = {list(edge_lengths)}")
    print(f"total L      = {2 * sum(edge_lengths)}")
    print(f"theta cutoff = {GENUS3_DEFAULT_THETA_CUTOFF}")
    print(
        "R scan       = "
        f"{GENUS3_DEFAULT_R_VALUES[0]:.1f} to {GENUS3_DEFAULT_R_VALUES[-1]:.1f}"
        " in steps of 0.1"
    )
    print("check        = R^3 Theta(4 i R^2 T') - R^(-3) Theta(4 i R^(-2) T')")

    if args.topology is not None:
        print(f"topology     = {args.topology}")
        quick = quick_check_genus3_topology(
            args.topology,
            edge_lengths=edge_lengths,
        )
        print("\nRows for selected topology:")
        for row in quick["rows"]:
            print(
                f"  R={row['R']:.1f}"
                f"  lhs={row['lhs']:>14.10f}"
                f"  rhs={row['rhs']:>14.10f}"
                f"  resid={row['residual']:>12.6e}"
                f"  rel={row['relative_residual']:>10.3e}"
            )

        worst = quick["overall_worst"]
        print("\nWorst case for selected topology:")
        print(
            f"  R={worst['R']:.1f}  topology={worst['topology']:>3d}"
            f"  resid={worst['residual']:>12.6e}"
            f"  rel={worst['relative_residual']:>10.3e}"
        )
        print(f"\nElapsed = {quick['elapsed_seconds']:.2f} s")
    else:
        summary = scan_genus3_t_duality(
            edge_lengths=edge_lengths,
            progress=True,
        )
        print("\nWorst relative residual by R:")
        for row in summary["worst_by_R"]:
            print(
                f"  R={row['R']:.1f}  topology={row['topology']:>3d}"
                f"  resid={row['residual']:>12.6e}"
                f"  rel={row['relative_residual']:>10.3e}"
            )

        worst = summary["overall_worst"]
        print("\nOverall worst case:")
        print(
            f"  R={worst['R']:.1f}  topology={worst['topology']:>3d}"
            f"  resid={worst['residual']:>12.6e}"
            f"  rel={worst['relative_residual']:>10.3e}"
        )
        print(f"\nElapsed = {summary['elapsed_seconds']:.2f} s")
