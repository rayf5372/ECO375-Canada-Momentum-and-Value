* Simple Linear Regression of Tilt Score vs Excess Return
/* The steps for approaching this linear regression were as follows. First, using data from yahoo finance, we compiled the prices of 30 Canadian equities and 30 Canadian bonds, as well as the risk free market rate (1 month Canadian treasury bond yield) over the span of about 10 years, compiled as panel data, from which we selected an abitrary month, Janaury of 2022, to slice a Cross-Sectional data set to run our calculations for constructing our dependent tilt variable and independent excess return variables */

cd "C:\Users\Michael\Desktop\SCHOOL\25-26\ECO375\ECO 375 Feasibility plan project\tilt regression"

import delimited using "cs_regression_2022-01.csv", clear varnames(1)

describe
summarize 

regress tilt_score excess_return

twoway (scatter tilt_score excess_return, msize(small) color(blue))
       
	   
display as txt "Regression complete"

