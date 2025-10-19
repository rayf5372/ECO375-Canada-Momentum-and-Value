cd "C:\Users\rfang\Documents\ECO375-Canada-Momentum-and-Value"
clear all
set more off

import delimited using "panel_momentum_ca.csv", clear varnames(1)
encode ticker, gen(id)
gen mdate = monthly(month,"YM")
format mdate %tm

areg excess_return_lead mom_12_2, absorb(mdate) vce(cluster id)

twoway (scatter excess_return_lead mom_12_2, msize(tiny) color(blue))