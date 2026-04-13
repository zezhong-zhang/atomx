# Gnuplot version 5.0 patchlevel 3                        
# Results from DHFS. Atomic potential and radial functions
                                                          
reset                                                     
                                                          
#Terminal to wxt, using Arial font                        
set terminal wxt size sqrt(2.)*750,750 enhanced \
  font 'Arial,14' dashlength 2.0                        
                                                          
set encoding utf8  # Character encoding                   
set mouse                                                 
set zero 1.0e-99                                          
                                                          
set linetype  1 lw 2 lc rgb 29390     # UB blue           
set linetype  2 lw 2 lc rgb 13382400  # soft red          
set linetype  3 lw 2 lc rgb 39168     # dark green        
set linetype  4 lw 2 lc rgb 10053171  # brown             
set linetype  5 lw 2 lc rgb 16724940  # pink              
set linetype 10 lw 2 lc rgb 0         # black             
set linetype 11 lw 2 lc rgb 8421504   # 50 percent black  
set xzeroaxis linestyle 11                                
set yzeroaxis linestyle 11                                
                                                          
set dashtype 1 (8,3)       # dashed                       
set dashtype 2 (12,3,1,3)  # dot-dashed                   
set dashtype 3 (4,3)       # dotted                       
                                                          
set tmargin at screen 0.90                                
set bmargin at screen 0.13                                
set rmargin at screen 0.94                                
set lmargin at screen 0.14                                
                                                          
set size 1,1  # The plot fills the available canvas       
set border lw 2.0                                         
set pointsize 0.7                                         
                                                          
# set format xy "%.1te%+-3T" or "10^{%T}"                 
# set format cb "%.1te%+-3T"                              
                                                          
#-----------------------------------------------------------------------

 set format xy '%.1tE%+-3T'
 set mxtics 10
 set mytics 10
 set xlabel 'r'
 set ylabel 'Z(r)=r*V(r)'
 set title 'Self-consistent potential'
 set key right center
 set logscale x
 unset logscale y
 set xrange [1.0e-7:*]
 plot'potden.dat' u 1:3 t 'r*Vnuc' w l ls 4,\
 'potden.dat' u 1:5 t 'r*Vex' w l ls 2,\
 'potden.dat' u 1:(-$4) t '-r*Vel' w l ls 3,\
 'potden.dat' u 1:($3+$4+$5) t 'r*V(r)' w l ls 1
 pause -1 'Press enter to continue'

 set logscale y
 set xrange [1.0e-5:*]
 set yrange [*:*]
 set title 'Local electron density'
 set ylabel 'rho(r)' 
 plot 'potden.dat' u 1:6 notitle w l ls 1
 pause -1 'Hit return to continue'

 set title 'Radial electron density, RED'
 set ylabel '4*pi*r^2*rho(r)'
 set logscale x
 unset logscale y
 plot 'potden.dat' u 1:($6*4.0*pi*$1**2) notitle w l ls 1
 pause -1 'Press enter to continue'

 set title 'Radial wave functions. Shell = 1s 1/2              '
 set key left top
 set yrange [*:*]
 set ylabel 'P(r), Q(r)'
 plot 'rwavefcts.dat' index  0 u 1:2 t 'P' w l ls 1 ,\
 'rwavefcts.dat' index  0 u 1:($3*5.0) t 'Q*5' w l ls 2 ,\
 'rwavefcts.dat' index  0 u 1:(($2**2+$3**2)*0.5) t '(P^2+Q^2)/2' w l ls 3 ,\
 'potden.dat' u 1:($6*4.*pi*$1**2/25.) t 'RED/25' w l ls 10
 pause -1 'Press enter to continue'

 set title 'Radial wave functions. Shell = 2s 1/2              '
 set key left top
 set yrange [*:*]
 set ylabel 'P(r), Q(r)'
 plot 'rwavefcts.dat' index  1 u 1:2 t 'P' w l ls 1 ,\
 'rwavefcts.dat' index  1 u 1:($3*5.0) t 'Q*5' w l ls 2 ,\
 'rwavefcts.dat' index  1 u 1:(($2**2+$3**2)*0.5) t '(P^2+Q^2)/2' w l ls 3 ,\
 'potden.dat' u 1:($6*4.*pi*$1**2/25.) t 'RED/25' w l ls 10
 pause -1 'Press enter to continue'

 set title 'Radial wave functions. Shell = 2p 1/2              '
 set key left top
 set yrange [*:*]
 set ylabel 'P(r), Q(r)'
 plot 'rwavefcts.dat' index  2 u 1:2 t 'P' w l ls 1 ,\
 'rwavefcts.dat' index  2 u 1:($3*5.0) t 'Q*5' w l ls 2 ,\
 'rwavefcts.dat' index  2 u 1:(($2**2+$3**2)*0.5) t '(P^2+Q^2)/2' w l ls 3 ,\
 'potden.dat' u 1:($6*4.*pi*$1**2/25.) t 'RED/25' w l ls 10
 pause -1 'Press enter to continue'

 set title 'Radial wave functions. Shell = 2p 3/2              '
 set key left top
 set yrange [*:*]
 set ylabel 'P(r), Q(r)'
 plot 'rwavefcts.dat' index  3 u 1:2 t 'P' w l ls 1 ,\
 'rwavefcts.dat' index  3 u 1:($3*5.0) t 'Q*5' w l ls 2 ,\
 'rwavefcts.dat' index  3 u 1:(($2**2+$3**2)*0.5) t '(P^2+Q^2)/2' w l ls 3 ,\
 'potden.dat' u 1:($6*4.*pi*$1**2/25.) t 'RED/25' w l ls 10
 pause -1 'Press enter to continue'

 set title 'Radial wave functions. Shell = 3s 1/2              '
 set key left top
 set yrange [*:*]
 set ylabel 'P(r), Q(r)'
 plot 'rwavefcts.dat' index  4 u 1:2 t 'P' w l ls 1 ,\
 'rwavefcts.dat' index  4 u 1:($3*5.0) t 'Q*5' w l ls 2 ,\
 'rwavefcts.dat' index  4 u 1:(($2**2+$3**2)*0.5) t '(P^2+Q^2)/2' w l ls 3 ,\
 'potden.dat' u 1:($6*4.*pi*$1**2/200.) t 'RED/200' w l ls 10
 pause -1 'Press enter to continue'

 set title 'Radial wave functions. Shell = 3p 1/2              '
 set key left top
 set yrange [*:*]
 set ylabel 'P(r), Q(r)'
 plot 'rwavefcts.dat' index  5 u 1:2 t 'P' w l ls 1 ,\
 'rwavefcts.dat' index  5 u 1:($3*5.0) t 'Q*5' w l ls 2 ,\
 'rwavefcts.dat' index  5 u 1:(($2**2+$3**2)*0.5) t '(P^2+Q^2)/2' w l ls 3 ,\
 'potden.dat' u 1:($6*4.*pi*$1**2/200.) t 'RED/200' w l ls 10
 pause -1 'Press enter to continue'

 set title 'Radial wave functions. Shell = 3p 3/2              '
 set key left top
 set yrange [*:*]
 set ylabel 'P(r), Q(r)'
 plot 'rwavefcts.dat' index  6 u 1:2 t 'P' w l ls 1 ,\
 'rwavefcts.dat' index  6 u 1:($3*5.0) t 'Q*5' w l ls 2 ,\
 'rwavefcts.dat' index  6 u 1:(($2**2+$3**2)*0.5) t '(P^2+Q^2)/2' w l ls 3 ,\
 'potden.dat' u 1:($6*4.*pi*$1**2/200.) t 'RED/200' w l ls 10
 pause -1 'Press enter to continue'
# END
