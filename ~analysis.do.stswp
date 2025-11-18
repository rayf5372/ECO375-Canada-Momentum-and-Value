clear all
set more off

************************************************************
* 1. Load factor-level data (LONG–SHORT RETURNS)
************************************************************
import delimited "factors_ca.csv", clear stringcols(_all)

* Make sure long–short factors are numeric
destring val_lh mom_lh, replace force

* Drop rows where both factors are missing
drop if missing(val_lh) & missing(mom_lh)

* Convert date string ("YYYY-MM-DD") to Stata date and year
gen date_s = date(date, "YMD")
format date_s %td
gen year   = year(date_s)

************************************************************
* 2. Summary statistics
************************************************************
summarize val_lh mom_lh

************************************************************
* 3. Simple regressions
************************************************************

* (i) Value only: mean of value L–H
reg val_lh

* (ii) Momentum only: mean of momentum L–H
reg mom_lh

* (iii) Joint regression: value L–H on momentum L–H
reg val_lh mom_lh

* (iv) Reverse joint: momentum L–H on value L–H
reg mom_lh val_lh

************************************************************
* 4. Correlation
************************************************************
corr val_lh mom_lh

************************************************************
* 5. Extensions: FE + interaction
************************************************************

* Year fixed effects (time dummies)
areg val_lh mom_lh, absorb(year)
areg mom_lh val_lh, absorb(year)

* Non-linear interaction term
gen val_mom = val_lh * mom_lh

reg val_lh mom_lh val_mom
reg mom_lh val_lh val_mom
